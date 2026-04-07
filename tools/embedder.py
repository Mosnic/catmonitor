import base64
import os
import requests
from pathlib import Path

EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8001/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")

# Resize target — keeps token count low (~238 tokens vs ~8000 at 4K)
# Must match the width used in frames.py FFmpeg scale filter
EMBED_WIDTH = 640


def embed_image(image_path: str) -> list[float]:
    """
    Embed a single image frame using Qwen3-VL-Embedding-2B.

    Uses the messages format required by vLLM's pooling endpoint for
    vision input — the standard embeddings 'input' field only accepts text.

    Args:
        image_path: Path to a JPEG frame (should already be 640px wide
                    from the FFmpeg extraction step in frames.py)

    Returns:
        Embedding vector as a list of floats.

    Raises:
        FileNotFoundError: if the image path does not exist
        requests.HTTPError: if the embedding endpoint returns an error
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": EMBED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post(
        EMBED_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )

    if not response.ok:
        print(f"Embed error {response.status_code}: {response.text[:300]}")
        response.raise_for_status()

    data = response.json()
    if "data" not in data or not data["data"]:
        raise ValueError(f"Unexpected embedding response format: {list(data.keys())}")
    embedding = data["data"][0].get("embedding")
    if not isinstance(embedding, list) or len(embedding) == 0:
        raise ValueError(f"Invalid embedding: expected non-empty list, got {type(embedding)}")
    return embedding


def average_embeddings(vec_a: list[float], vec_b: list[float]) -> list[float]:
    """
    Return the element-wise average of two equal-length embedding vectors.

    Used to update a cat's prototype after each confirmed visit:
    the averaged vector drifts toward a more representative centre
    as more confirmed visits accumulate.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Embedding length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    return [(a + b) / 2.0 for a, b in zip(vec_a, vec_b)]


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 tools/embedder.py <image_path>")
        sys.exit(1)

    vec = embed_image(sys.argv[1])
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 8 values: {vec[:8]}")
    print(json.dumps({"dimension": len(vec), "sample": vec[:8]}))
