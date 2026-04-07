import asyncio
import aiohttp
import base64
import cv2
import tempfile

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"

async def test():
    image_path = "/tmp/tmp24wvnid8/frame_0000.jpg"
    
    # Check original size
    img = cv2.imread(image_path)
    print(f"Original size: {img.shape}")
    
    # Resize to 640px wide
    height, width = img.shape[:2]
    scale = 640 / width
    new_size = (640, int(height * scale))
    img = cv2.resize(img, new_size)
    print(f"Resized to: {img.shape}")
    
    # Save resized to temp file
    tmp = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(tmp, img)
    
    with open(tmp, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

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
                        "text": "What do you see in this image?"
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            VLLM_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Prompt tokens: {data['usage']['prompt_tokens']}")
            print(f"Completion tokens: {data['usage']['completion_tokens']}")
            print(f"Finish reason: {data['choices'][0]['finish_reason']}")
            print(f"Response: {data['choices'][0]['message']['content']}")

asyncio.run(test())