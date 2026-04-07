import json
import re
import base64
import asyncio
import aiohttp
import tempfile
from pathlib import Path

import os

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
TOP_FRAMES = 4
MAX_CONCURRENT = int(os.getenv("VLLM_MAX_CONCURRENT", "1"))

SCORING_PROMPT = """
You are evaluating a single frame from a security camera at a stray cat feeding station.
Rate this frame for usefulness in identifying and analyzing a cat.

Consider:
- Is a cat clearly visible?
- Is the cat in focus and well lit?
- Is the cat's face or body clearly distinguishable?
- Is the frame free of motion blur?
- Does the frame show useful detail like coat pattern, markings, or behavior?

Return ONLY a JSON object, nothing else:
{
  "score": integer from 1 to 10,
  "cat_visible": boolean,
  "reason": string (one sentence explaining the score)
}

Score guide:
1-3: No cat visible, completely blurry, or too dark to see anything useful
4-6: Cat present but partially obscured, blurry, or poor angle
7-8: Cat clearly visible with good detail
9-10: Excellent frame — cat face or full body clearly visible with sharp detail
"""


import subprocess

def extract_frames_at_1fps(video_path: str) -> list[tuple[int, str]]:
    """
    Extract one frame per second using FFmpeg.
    More reliable than OpenCV for security camera formats.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    temp_dir = Path(tempfile.mkdtemp())

    # Get duration using ffprobe
    probe = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path)
    ], capture_output=True, text=True)

    import json as _json
    probe_data = _json.loads(probe.stdout)
    
    # Find video stream duration
    duration = 0
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            duration = int(float(stream.get("duration", 0)))
            fps = stream.get("r_frame_rate", "12/1")
            print(f"Video: {duration}s, fps: {fps}")
            break

    if duration == 0:
        raise ValueError(f"Could not determine video duration: {video_path}")

    # Extract 1 frame per second using FFmpeg
    output_pattern = str(temp_dir / "frame_%04d.jpg")
    
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", "fps=1,scale=640:-1",  # 1fps AND resize in one step
        "-q:v", "2",                   # high quality JPEG
        output_pattern,
        "-y", "-loglevel", "error"
    ], check=True)

    # Collect extracted frames
    frames = sorted(temp_dir.glob("frame_*.jpg"))
    result = [(i, str(f)) for i, f in enumerate(frames)]

    print(f"Extracted {len(result)} frames to {temp_dir}")
    return result


def encode_image(image_path: str) -> str:
    """Encode image file as base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def score_frame(
    session: aiohttp.ClientSession,
    second: int,
    image_path: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """
    Score a single frame using the vision model.
    Semaphore limits concurrent calls to MAX_CONCURRENT.
    """
    async with semaphore:
        image_b64 = encode_image(image_path)

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
                            "text": SCORING_PROMPT
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        try:
            async with session.post(
                VLLM_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                content = data["choices"][0]["message"]["content"]

                # Extract JSON from response
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group())
                    except json.JSONDecodeError as e:
                        print(f"  Frame {second:3d}s → JSON parse error: {e}")
                        return {
                            "second": second,
                            "image_path": image_path,
                            "score": 0,
                            "cat_visible": False,
                            "reason": f"JSON parse error: {e}"
                        }
                    result["second"] = second
                    result["image_path"] = image_path
                    print(f"  Frame {second:3d}s → score:{result.get('score', 0)} "
                          f"cat:{result.get('cat_visible', False)} "
                          f"— {result.get('reason', '')[:60]}")
                    return result

        except Exception as e:
            print(f"  Frame {second:3d}s → error: {e}")

        return {
            "second": second,
            "image_path": image_path,
            "score": 0,
            "cat_visible": False,
            "reason": "error during scoring"
        }


async def score_all_frames(
    frames: list[tuple[int, str]]
) -> list[dict]:
    """
    Score all frames concurrently with MAX_CONCURRENT limit.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        tasks = [
            score_frame(session, second, image_path, semaphore)
            for second, image_path in frames
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


def select_best_frames(
    scored_frames: list[dict],
    top_n: int = TOP_FRAMES,
    min_spacing: int = 3
) -> list[dict]:
    """
    Select top N frames by score.
    min_spacing ensures frames are at least N seconds apart
    so we get temporal diversity not just 4 frames from the same moment.
    """
    # Only consider frames where a cat is visible
    cat_frames = [f for f in scored_frames if f.get("cat_visible")]

    if not cat_frames:
        # Fall back to top scored frames regardless
        cat_frames = scored_frames

    # Sort by score descending
    cat_frames.sort(key=lambda x: x.get("score", 0), reverse=True)

    selected = []
    for frame in cat_frames:
        # Check spacing from already selected frames
        too_close = any(
            abs(frame["second"] - s["second"]) < min_spacing
            for s in selected
        )
        if not too_close:
            selected.append(frame)

        if len(selected) == top_n:
            break

    # Return sorted by time for chronological analysis
    selected.sort(key=lambda x: x["second"])
    return selected


async def extract_best_frames_async(video_path: str) -> list[str]:
    """
    Full pipeline: extract → score → select.
    Returns list of image paths for the best frames.
    """
    print(f"\n=== Extracting best frames from: {video_path} ===")

    # Step 1 - extract 1fps frames
    frames = extract_frames_at_1fps(video_path)

    # Step 2 - score all frames concurrently
    print(f"\nScoring {len(frames)} frames "
          f"({MAX_CONCURRENT} concurrent calls)...")
    scored = await score_all_frames(frames)

    # Step 3 - select best with temporal spacing
    best = select_best_frames(scored, top_n=TOP_FRAMES)

    # Best frame is the highest-scored one — capture before time-sort reorders
    best_frame = best[0]["image_path"]

    # Re-sort chronologically for the agent's narrative pass
    best.sort(key=lambda x: x["second"])

    print(f"\n=== Selected {len(best)} best frames ===")
    for frame in best:
        marker = " <- best" if frame["image_path"] == best_frame else ""
        print(f"  {frame['second']}s → score:{frame['score']} "
              f"— {frame['reason'][:70]}{marker}")

    paths = [frame["image_path"] for frame in best]
    return paths, best_frame


def extract_best_frames(video_path: str) -> tuple[list[str], str]:
    """
    Synchronous wrapper — this is what the agent calls.

    Returns:
        (paths, best_frame) where paths is the list of selected frame
        image paths in chronological order, and best_frame is the single
        highest-scored path for use as the identity embedding source.
    """
    return asyncio.run(extract_best_frames_async(video_path))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 frames.py <video_path>")
        sys.exit(1)

    video = sys.argv[1]
    paths, best_frame = extract_best_frames(video)

    print(f"\nBest frames (chronological):")
    for path in paths:
        print(f"  {path}")
    print(f"\nBest frame for embedding:")
    print(f"  {best_frame}")