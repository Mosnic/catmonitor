import base64
import json
import os
import re
import time
import requests
from pathlib import Path

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

VISION_PROMPT = """
You are analyzing a security camera image from a stray cat feeding station.
Examine the image carefully and return ONLY a valid JSON object — no other text,
no markdown, no explanation.

Use exactly this schema:

{
  "cat_present": boolean,
  "cat_count": integer,
  "cats": [
    {
      "coat_color": string,
      "coat_pattern": "solid | tabby | bicolor | tortoiseshell | calico | unknown",
      "coat_length": "short | medium | long | unknown",
      "size": "small | medium | large | unknown",
      "build": "lean | normal | stocky | unknown",
      "distinctive_markings": ["array of strings"],
      "eye_color": "yellow | green | blue | amber | unknown",
      "body_condition": "poor | fair | good | excellent",
      "behavior": "eating | resting | alert | grooming | fleeing | fighting | unknown",
      "health_flags": ["array of strings"],
      "confidence": "low | medium | high"
    }
  ],
  "camera": "platform_front | platform_right | unknown",
  "lighting": "good | partial | poor",
  "notes": string
}

Rules:
- If no cat is present return cat_present: false, cat_count: 0, empty cats array
- If multiple cats visible add one object per cat in the cats array
- Read camera name from overlay text in top-left corner of the image
- Set confidence to low if lighting is poor or cat is partially obscured
- health_flags and distinctive_markings should be empty arrays if nothing notable
- Return ONLY the JSON object, nothing else
IMPORTANT VALIDATION RULES:
- Only report cats (Felis catus). 
- If the visitor is a bird, raccoon, squirrel, opossum, or any 
  other non-cat animal, set cat_present to false and describe 
  what you actually see in the notes field.
- A cat has: pointed ears, whiskers, fur, a tail, and four legs.
  If these features are not clearly present, do not classify as cat.
- If you are uncertain whether the visitor is a cat, set 
  confidence to "low" and describe your uncertainty in notes.
"""

def load_image_as_base64(image_path: str) -> str:
    """Load an image file and return as base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_json(text: str) -> dict:
    """Extract JSON from model response, handling any extra text."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in response: {text}")

def analyze_cat_image(image_path: str) -> dict:
    """
    Send a camera frame to the vision model for analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Structured dict describing any cats present
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_b64 = load_image_as_base64(str(image_path))

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": VISION_PROMPT
                    }
                ]
            }
        ],
        "max_tokens": 800
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                VLLM_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Vision request attempt {attempt + 1} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    content = response.json()["choices"][0]["message"]["content"]
    return extract_json(content)


if __name__ == "__main__":
    # Quick test
    import sys
    
    image = sys.argv[1] if len(sys.argv) > 1 else "~/cat_test.jpg"
    image = str(Path(image).expanduser())
    
    print(f"Analyzing: {image}")
    result = analyze_cat_image(image)
    print(json.dumps(result, indent=2))