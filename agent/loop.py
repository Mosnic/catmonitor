import json
import time
import requests
from pathlib import Path
from datetime import datetime
import sys

# Add parent to path so we can import our tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.frames import extract_best_frames
from tools.vision import analyze_cat_image
from db.store import process_visit, init_sqlite
# ANSI color codes
class Color:
    TOOL_NAME   = "\033[38;5;214m"  # orange
    TOOL_ARGS   = "\033[38;5;244m"  # grey
    TOOL_RESULT = "\033[38;5;71m"   # green
    ITERATION   = "\033[38;5;39m"   # blue
    SUMMARY     = "\033[38;5;226m"  # yellow
    ERROR       = "\033[38;5;196m"  # red
    RESET       = "\033[0m"
import os
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE", "8192"))
MIN_OUTPUT_TOKENS = 256
DESIRED_OUTPUT_TOKENS = 1000


def _estimate_tokens(messages: list, tools: list) -> int:
    """
    Conservative token estimate. Uses chars/3 (not chars/4) to account
    for JSON structure, tool schema overhead, and message framing tokens.
    """
    total_chars = len(json.dumps(tools))
    for msg in messages:
        total_chars += 16  # per-message framing overhead
        content = msg.get("content") or ""
        total_chars += len(content)
        if msg.get("tool_calls"):
            total_chars += len(json.dumps(msg["tool_calls"]))
    return total_chars // 3


def _compute_max_tokens(messages: list, tools: list) -> int:
    """Return max_tokens capped to fit within the model's context window."""
    est_input = _estimate_tokens(messages, tools)
    available = MODEL_CONTEXT_SIZE - est_input
    if available < MIN_OUTPUT_TOKENS:
        return MIN_OUTPUT_TOKENS
    return min(available, DESIRED_OUTPUT_TOKENS)

# ─────────────────────────────────────────────
# Tool definitions
# These are what the agent sees and reasons about
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_best_frames",
            "description": (
                "Extract the best frames from a video clip for cat identification. "
                "Scores all frames and returns the top 4 most useful ones. "
                "Always call this first when given a video path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video clip file"
                    }
                },
                "required": ["video_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_cat_image",
            "description": (
                "Analyze a single image frame from the security camera. "
                "Returns structured data about any cats present including "
                "coat color, pattern, size, behavior, body condition, and health flags. "
                "Call this on each frame returned by extract_best_frames."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Full path to the image file to analyze"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_visit",
            "description": (
                "Record a cat visit to the database. "
                "Matches the cat against known profiles using image embedding, "
                "or creates a new profile. "
                "Call this once per cat detected after analyzing frames. "
                "Pass the best_frame value from extract_best_frames as best_frame_path. "
                "Returns whether this is a known, new, or uncertain cat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "object",
                        "description": "The structured analysis dict returned by analyze_cat_image"
                    },
                    "clip_path": {
                        "type": "string",
                        "description": "Path to the original video clip"
                    },
                    "best_frame_path": {
                        "type": "string",
                        "description": "Path to the best frame image — use the best_frame value returned by extract_best_frames"
                    }
                },
                "required": ["analysis"]
            }
        }
    }
]

SYSTEM_PROMPT = """
You are an autonomous monitoring agent for a stray cat feeding station.
You have access to security camera footage and tools to analyze it.

Your job when given a video clip:
1. Extract the best frames from the clip — note the best_frame value in the result
2. Analyze the frames to identify and describe any cats present
3. Record the visit to the database — pass best_frame as best_frame_path to process_visit
4. Report a clear summary including: cat description, known/new status, any health flags,
   and the weather conditions from the process_visit result (temperature_f, precipitation_mm,
   windspeed_mph, weather_code). If weather is null, note it was unavailable.

Be thorough but efficient. If multiple frames show the same cat, 
use the best quality frame for the final analysis.
If health flags or unusual behavior are detected, highlight them clearly.
If this is a new cat never seen before, note that prominently.

Always complete all steps — do not stop after extracting frames or analyzing.
Finish by calling process_visit and then provide your final summary.

Your final summary must include the weather conditions at the time of the visit.

"""


# ─────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict, clip_timestamp: datetime) -> str:
    """
    Execute a tool call requested by the agent.
    clip_timestamp is the mtime of the source clip — passed to process_visit
    so weather is fetched for the actual recording time, not processing time.
    Returns the result as a string for the agent to reason about.
    """
    print(f"\n  {Color.TOOL_NAME}→ Executing tool: {tool_name}{Color.RESET}")
    print(f"  {Color.TOOL_ARGS}  Args: {json.dumps(tool_args, indent=2)[:200]}{Color.RESET}")
    try:
        if tool_name == "extract_best_frames":
            paths, best_frame = extract_best_frames(tool_args["video_path"])
            return json.dumps({
                "frames": paths,
                "count": len(paths),
                "best_frame": best_frame,
            })

        elif tool_name == "analyze_cat_image":
            result = analyze_cat_image(tool_args["image_path"])
            # Log if something non-cat was detected
            if result.get("cat_present") and result.get("cats"):
                confidence = result["cats"][0].get("confidence", "high")
                if confidence == "low":
                    print(f"  {Color.ERROR}  ⚠ Low confidence detection — may not be a cat{Color.RESET}")
            return json.dumps(result)

        elif tool_name == "process_visit":
            result = process_visit(
                tool_args["analysis"],
                tool_args.get("clip_path", ""),
                tool_args.get("best_frame_path", ""),
                timestamp=clip_timestamp,  # recording time → correct weather lookup
            )
            return json.dumps(result)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        error = f"Tool error: {str(e)}"
        print(f"  {Color.ERROR}  ERROR: {error}{Color.RESET}")
        return json.dumps({"error": error})


# ─────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────

def run_agent(clip_path: str) -> str:
    """
    Run the agent loop for a single video clip.
    
    This is the core of the autonomous monitor:
    - The agent decides what tools to call
    - We execute them and return results
    - The agent reasons about results and decides next step
    - Loop continues until agent signals it is done
    
    Returns the agent's final summary.
    """
    print(f"\n{'='*60}")
    print(f"AGENT: Processing clip: {clip_path}")
    print(f"{'='*60}")

    # Derive the clip's recording time from its mtime.
    # This is what process_visit (and in turn get_weather_for_visit) uses
    # so that weather reflects conditions at the time of recording, not processing.
    try:
        clip_timestamp = datetime.fromtimestamp(Path(clip_path).stat().st_mtime)
        print(f"Clip timestamp (mtime): {clip_timestamp.isoformat()}")
    except Exception as e:
        clip_timestamp = datetime.now()
        print(f"{Color.ERROR}WARNING: could not read clip mtime ({e}), "
              f"falling back to now() — weather data will be inaccurate{Color.RESET}")

    # Conversation history — this is the agent's memory for this session
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"A new camera clip has arrived. Please analyze it and report what you find.\n\nClip path: {clip_path}"
        }
    ]

    max_iterations = 10  # safety limit — prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{Color.ITERATION}[Iteration {iteration}] Calling model...{Color.RESET}")


        # Call the model (retry on timeouts, fail fast if server is down)
        max_tok = _compute_max_tokens(messages, TOOLS)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    VLLM_URL,
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "tools": TOOLS,
                        "tool_choice": "auto",
                        "max_tokens": max_tok,
                    },
                    timeout=90
                )
                break
            except requests.ConnectionError:
                raise  # Server is down — no point retrying
            except requests.Timeout as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s
                    print(f"  Model request attempt {attempt + 1} timed out, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:500]}")
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]
        finish_reason = data["choices"][0]["finish_reason"]

        print(f"  Finish reason: {finish_reason}")

        # Add assistant message to history
        messages.append({
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls")
        })

        # ── Agent is done ──
        if finish_reason == "stop" or not message.get("tool_calls"):
            final_response = message.get("content", "No summary provided.")
            print(f"\n{Color.SUMMARY}{'='*60}")
            print("AGENT SUMMARY:")
            print(final_response)
            print(f"{'='*60}{Color.RESET}")
            return final_response

        # ── Agent wants to call tools ──
        tool_calls = message["tool_calls"]
        print(f"  {Color.TOOL_NAME}Tool calls requested: {[tc['function']['name'] for tc in tool_calls]}{Color.RESET}")


        # Execute each tool and add results to conversation
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            result = execute_tool(tool_name, tool_args, clip_timestamp)

            # Add tool result to conversation history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            })

    last_tool_calls = []
    for msg in reversed(messages):
        if msg.get("tool_calls"):
            last_tool_calls = [tc["function"]["name"] for tc in msg["tool_calls"]]
            break
    print(f"{Color.ERROR}Max iterations ({max_iterations}) reached. "
          f"Last tools attempted: {last_tool_calls}{Color.RESET}")
    return f"Agent reached maximum iterations without completing. Last tools: {last_tool_calls}"


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    init_sqlite()

    if len(sys.argv) < 2:
        print("Usage: python3 agent/loop.py <video_path>")
        sys.exit(1)

    clip = sys.argv[1]
    summary = run_agent(clip)
