"""
Reflective agent loop — reasons over cat history, detects patterns,
flags concerns, resolves identity links, and delegates statistics
to the coding model.

Mirrors the structure of agent/loop.py but operates on accumulated
records rather than a single video clip.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.reflective_tools import (
    get_cat_dossier,
    get_cat_history,
    get_absent_cats,
    get_health_trends,
    get_overnight_summary,
    get_uncertain_matches,
    get_provisional_links,
    resolve_link,
    recompute_prototype,
    generate_and_run_analysis,
)
from db.store import get_sqlite_connection, init_sqlite


# ANSI color codes
class Color:
    TOOL_NAME   = "\033[38;5;214m"  # orange
    TOOL_ARGS   = "\033[38;5;244m"  # grey
    TOOL_RESULT = "\033[38;5;71m"   # green
    ITERATION   = "\033[38;5;39m"   # blue
    SUMMARY     = "\033[38;5;226m"  # yellow
    ERROR       = "\033[38;5;196m"  # red
    ALERT       = "\033[38;5;201m"  # magenta
    RESET       = "\033[0m"


VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE", "8192"))
MIN_OUTPUT_TOKENS = 256
DESIRED_OUTPUT_TOKENS = 2000


# ─────────────────────────────────────────────
# Context window management
# ─────────────────────────────────────────────

def estimate_tokens(messages: list, tools: list) -> int:
    """
    Conservative token estimate. Uses chars/3 (not chars/4) to account
    for JSON structure, tool schema overhead, and message framing tokens
    that the API adds beyond raw content.
    """
    total_chars = len(json.dumps(tools))
    for msg in messages:
        total_chars += 16  # per-message framing overhead (role, delimiters)
        content = msg.get("content") or ""
        total_chars += len(content)
        if msg.get("tool_calls"):
            total_chars += len(json.dumps(msg["tool_calls"]))
    # chars/3 is intentionally conservative to avoid 400 errors
    return total_chars // 3


def compute_max_tokens(messages: list, tools: list) -> int:
    """Return max_tokens capped to fit within the model's context window."""
    est_input = estimate_tokens(messages, tools)
    available = MODEL_CONTEXT_SIZE - est_input
    if available < MIN_OUTPUT_TOKENS:
        return MIN_OUTPUT_TOKENS
    return min(available, DESIRED_OUTPUT_TOKENS)


def trim_messages(messages: list) -> list:
    """
    Drop the oldest assistant+tool turns (keeping system and first user
    message) when the conversation is approaching the context limit.
    """
    est = estimate_tokens(messages, TOOLS)
    # Leave room for DESIRED_OUTPUT_TOKENS plus a safety margin
    target = MODEL_CONTEXT_SIZE - DESIRED_OUTPUT_TOKENS - 512

    if est <= target:
        return messages

    # Always keep messages[0] (system) and messages[1] (initial user context).
    # Remove from index 2 onward, oldest first.
    trimmed = list(messages)
    while estimate_tokens(trimmed, TOOLS) > target and len(trimmed) > 3:
        # Remove the oldest non-system/non-initial message
        trimmed.pop(2)

    if len(trimmed) <= 3:
        # Can't trim further — just return what we have
        return trimmed

    return trimmed


# ─────────────────────────────────────────────
# Tool definitions (what the model sees)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cat_dossier",
            "description": (
                "Get the full context for a single cat: prototype embedding, "
                "confidence trajectory, gap periods, human corrections, and "
                "provisional links. Use this to deeply investigate a specific cat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cat_id": {
                        "type": "string",
                        "description": "The cat_id to look up"
                    }
                },
                "required": ["cat_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cat_history",
            "description": (
                "Return the most recent visit records for a cat, newest first. "
                "Use to review recent activity for a specific cat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cat_id": {
                        "type": "string",
                        "description": "The cat_id to look up"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max records to return (default 60)"
                    }
                },
                "required": ["cat_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_absent_cats",
            "description": (
                "Find active cats not seen in the last N days. "
                "Use to identify cats that may have stopped visiting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Absence threshold in days (default 7)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_trends",
            "description": (
                "Get body_condition and health_flags over time for a cat. "
                "Use to spot declining condition or recurring health issues."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cat_id": {
                        "type": "string",
                        "description": "The cat_id to look up"
                    }
                },
                "required": ["cat_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_overnight_summary",
            "description": (
                "Get all visits from the last 12 hours. "
                "Call this first to understand recent activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_uncertain_matches",
            "description": (
                "Find visits with confidence in the 0.25–0.45 uncertainty band "
                "that have not been resolved. These may represent identity "
                "confusion between similar-looking cats."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_provisional_links",
            "description": (
                "Get all unresolved identity links between cats, enriched with "
                "descriptions. Review these to decide which should be merged or rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_link",
            "description": (
                "Resolve a provisional identity link. If confirmed=true, the "
                "candidate cat is merged into the target (visits reassigned, "
                "candidate marked as merged). If confirmed=false, the link is "
                "rejected with no changes. After confirming a merge, call "
                "recompute_prototype on the surviving cat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "link_id": {
                        "type": "integer",
                        "description": "The link_id to resolve"
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "True to merge, false to reject"
                    }
                },
                "required": ["link_id", "confirmed"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recompute_prototype",
            "description": (
                "Re-average the ChromaDB embedding for a cat from all its visit "
                "frames. Call after merging cats to update the surviving cat's "
                "prototype. This is expensive — only call when needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cat_id": {
                        "type": "string",
                        "description": "The cat_id whose prototype to recompute"
                    }
                },
                "required": ["cat_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_and_run_analysis",
            "description": (
                "Delegate a statistical or data analysis task to the coding model. "
                "Describe the analysis in plain language — the coding model will "
                "generate and execute Python code against the database. "
                "Use for visit frequency analysis, correlation studies, or any "
                "question requiring computation beyond simple queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Plain-language description of the analysis to perform"
                    }
                },
                "required": ["task_description"]
            }
        }
    },
]


SYSTEM_PROMPT = """\
You are a reflective monitoring agent for a stray cat feeding station.
Your role is to reason over accumulated visit history, detect patterns,
flag health or identity concerns, and resolve pending identity questions.

You have access to tools that query the cat database and a coding model
for statistical analysis. Your workflow:

1. Start by calling get_overnight_summary to see recent activity.
2. Call get_uncertain_matches and get_provisional_links to find identity
   questions that need attention.
3. For each uncertain match or provisional link, investigate by pulling
   cat dossiers (get_cat_dossier) and health trends (get_health_trends).
4. Resolve provisional links when you have enough evidence — confirm
   merges when descriptions, patterns, and confidence align; reject when
   they clearly differ. After a merge, call recompute_prototype.
5. Check for absent cats (get_absent_cats) and investigate any that are
   concerning.
6. Use generate_and_run_analysis to delegate statistical questions
   (e.g., visit frequency trends, weather correlations).
7. There are potentially multiple cameras with MP4 clips for each cat visit. These should be treated as a single visit.
When you find something concerning, describe it clearly in your final
summary. Your findings will be saved as alerts.

Types of findings to report (use these exact alert_type values):
- health_decline: a cat's body condition is worsening over time
- extended_absence: an active cat hasn't been seen in 7+ days
- frequency_change: a cat's visit frequency has shifted significantly
  (e.g., daily visitor now appearing weekly, or sudden spike)
- new_health_flag: a health flag appeared for the first time on a cat
  (e.g., first observation of limping, eye discharge, weight loss)
- uncertain_identity: visits with low confidence that need human review
- link_candidate_ready: a provisional link has enough evidence for
  human review — describe both cats and your confidence assessment
- statistical_deviation: a metric (visit rate, time-of-day pattern,
  weather correlation) deviates significantly from historical baseline
- embedding_drift: a cat's recent embeddings are drifting away from
  its prototype, suggesting appearance change or identity confusion
- identity_merged: you confirmed a merge between two cat records
- identity_rejected: you rejected a proposed link between cats
- pattern_anomaly: unusual visit patterns or statistical outliers
- health_flag: recurring health flags on a cat
- prolonged_absence: (legacy) alias for extended_absence

Be thorough but efficient. End with a clear summary of all findings
and actions taken.
"""


# ─────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute a reflective tool call. Returns the result as a JSON string.
    """
    print(f"\n  {Color.TOOL_NAME}> Executing tool: {tool_name}{Color.RESET}")
    print(f"  {Color.TOOL_ARGS}  Args: {json.dumps(tool_args, indent=2)[:200]}{Color.RESET}")

    try:
        if tool_name == "get_cat_dossier":
            result = get_cat_dossier(tool_args["cat_id"])
            # Strip embedding vectors from output — too large for the model
            if isinstance(result, dict) and "prototype_embedding" in result:
                has_embedding = result["prototype_embedding"] is not None
                result["prototype_embedding"] = (
                    f"<{len(result['prototype_embedding'])}d vector>"
                    if has_embedding else None
                )
            return json.dumps(result, default=str)

        elif tool_name == "get_cat_history":
            result = get_cat_history(
                tool_args["cat_id"],
                tool_args.get("limit", 60),
            )
            return json.dumps(result, default=str)

        elif tool_name == "get_absent_cats":
            result = get_absent_cats(tool_args.get("days", 7))
            return json.dumps(result, default=str)

        elif tool_name == "get_health_trends":
            result = get_health_trends(tool_args["cat_id"])
            return json.dumps(result, default=str)

        elif tool_name == "get_overnight_summary":
            result = get_overnight_summary()
            return json.dumps(result, default=str)

        elif tool_name == "get_uncertain_matches":
            result = get_uncertain_matches()
            return json.dumps(result, default=str)

        elif tool_name == "get_provisional_links":
            result = get_provisional_links()
            return json.dumps(result, default=str)

        elif tool_name == "resolve_link":
            result = resolve_link(tool_args["link_id"], tool_args["confirmed"])
            return json.dumps(result, default=str)

        elif tool_name == "recompute_prototype":
            result = recompute_prototype(tool_args["cat_id"])
            return json.dumps(result, default=str)

        elif tool_name == "generate_and_run_analysis":
            result = generate_and_run_analysis(tool_args["task_description"])
            return result  # already a string

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        error = f"Tool error: {str(e)}"
        print(f"  {Color.ERROR}  ERROR: {error}{Color.RESET}")
        return json.dumps({"error": error})


# ─────────────────────────────────────────────
# Alert persistence
# ─────────────────────────────────────────────

def write_alert(cat_id: str | None, alert_type: str, detail: str):
    """Write a finding to the alerts table."""
    conn = get_sqlite_connection()
    conn.execute(
        "INSERT INTO alerts (cat_id, timestamp, alert_type, detail) "
        "VALUES (?, ?, ?, ?)",
        (cat_id, datetime.now().isoformat(), alert_type, detail),
    )
    conn.commit()
    conn.close()
    print(f"  {Color.ALERT}[!] Alert saved: [{alert_type}] {detail[:80]}{Color.RESET}")


def parse_and_save_findings(summary: str):
    """
    Parse the agent's final summary for structured findings and write
    each one to the alerts table.

    The agent is instructed to report findings with alert_type labels.
    We look for lines matching the pattern:
        - <alert_type>: <cat_id or 'general'>: <detail>
    Falls back to saving the entire summary as a single 'reflective_summary'
    alert if no structured findings are found.
    """
    import re
    VALID_TYPES = {
        "health_decline", "prolonged_absence", "identity_merged",
        "identity_rejected", "uncertain_identity", "pattern_anomaly",
        "health_flag", "reflective_summary",
        # Extended alert types for structured querying
        "extended_absence", "frequency_change", "new_health_flag",
        "link_candidate_ready", "statistical_deviation", "embedding_drift",
    }

    findings = []
    for line in summary.splitlines():
        line = line.strip().lstrip("- •*")
        # Match: alert_type: cat_id: detail  OR  alert_type: detail
        m = re.match(
            r"([\w_]+)\s*:\s*(?:([a-f0-9\-]{8,})\s*:\s*)?(.+)",
            line, re.IGNORECASE,
        )
        if m and m.group(1).lower() in VALID_TYPES:
            findings.append({
                "alert_type": m.group(1).lower(),
                "cat_id": m.group(2),
                "detail": m.group(3).strip(),
            })

    if not findings:
        # Save entire summary as a single alert
        write_alert(None, "reflective_summary", summary[:2000])
        return

    for f in findings:
        write_alert(f["cat_id"], f["alert_type"], f["detail"])


# ─────────────────────────────────────────────
# Build the starting user message
# ─────────────────────────────────────────────

def build_starting_context() -> str:
    """
    Construct the initial user message with current state:
    which cats are active, what the overnight summary shows,
    and what links are pending.
    """
    conn = get_sqlite_connection()

    # Active cats
    active = conn.execute(
        "SELECT cat_id, description, last_seen, visit_count "
        "FROM cats WHERE status = 'active' ORDER BY last_seen DESC"
    ).fetchall()

    # Pending links count
    pending_links = conn.execute(
        "SELECT COUNT(*) as cnt FROM links WHERE resolved = 0"
    ).fetchone()["cnt"]

    # Overnight visit count
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(hours=12)).isoformat()
    overnight_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM visits WHERE timestamp > ?",
        (cutoff,),
    ).fetchone()["cnt"]

    conn.close()

    lines = [
        "Time for your reflective review. Here is the current state:\n",
        f"**Active cats:** {len(active)}",
    ]
    for cat in active[:15]:  # cap to avoid overwhelming context
        lines.append(
            f"  - {cat['cat_id'][:8]}… : {cat['description'] or 'no description'} "
            f"(last seen {cat['last_seen']}, {cat['visit_count']} visits)"
        )
    if len(active) > 15:
        lines.append(f"  ... and {len(active) - 15} more")

    lines.append(f"\n**Overnight visits (last 12h):** {overnight_count}")
    lines.append(f"**Pending identity links:** {pending_links}")
    lines.append(
        "\nPlease review overnight activity, investigate uncertain matches "
        "and pending links, check for absent or declining cats, and report "
        "your findings."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────

def run_reflective_agent() -> str:
    """
    Run the reflective agent loop.

    The agent reasons over accumulated cat records, resolves identity
    questions, and flags concerns. Findings are written to the alerts
    table on completion.

    Returns the agent's final summary.
    """
    print(f"\n{'='*60}")
    print("REFLECTIVE AGENT: Starting review")
    print(f"{'='*60}")

    starting_context = build_starting_context()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": starting_context},
    ]

    max_iterations = 200  # reflective runs are more involved
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{Color.ITERATION}[Iteration {iteration}] Calling model...{Color.RESET}")

        # Trim old messages if context is getting too large
        messages = trim_messages(messages)
        max_tok = compute_max_tokens(messages, TOOLS)
        print(f"  Context estimate: ~{estimate_tokens(messages, TOOLS)} tokens, "
              f"max_tokens={max_tok}")

        # Call the model (retry on timeouts)
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
                    timeout=120,
                )
                break
            except requests.ConnectionError:
                raise  # Server is down — no point retrying
            except requests.Timeout:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
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
            "tool_calls": message.get("tool_calls"),
        })

        # ── Agent is done ──
        if finish_reason == "stop" or not message.get("tool_calls"):
            final_response = message.get("content", "No summary provided.")
            print(f"\n{Color.SUMMARY}{'='*60}")
            print("REFLECTIVE AGENT SUMMARY:")
            print(final_response)
            print(f"{'='*60}{Color.RESET}")

            # Persist findings as alerts
            parse_and_save_findings(final_response)
            return final_response

        # ── Agent wants to call tools ──
        tool_calls = message["tool_calls"]
        print(f"  {Color.TOOL_NAME}Tool calls requested: "
              f"{[tc['function']['name'] for tc in tool_calls]}{Color.RESET}")

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            result = execute_tool(tool_name, tool_args)

            print(f"  {Color.TOOL_RESULT}  Result: {result[:300]}{Color.RESET}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })

    # Max iterations reached
    last_tool_calls = []
    for msg in reversed(messages):
        if msg.get("tool_calls"):
            last_tool_calls = [tc["function"]["name"] for tc in msg["tool_calls"]]
            break
    print(f"{Color.ERROR}Max iterations ({max_iterations}) reached. "
          f"Last tools attempted: {last_tool_calls}{Color.RESET}")

    summary = (f"Reflective agent reached maximum iterations ({max_iterations}) "
               f"without completing. Last tools: {last_tool_calls}")
    write_alert(None, "reflective_summary", summary)
    return summary


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    init_sqlite()
    summary = run_reflective_agent()
