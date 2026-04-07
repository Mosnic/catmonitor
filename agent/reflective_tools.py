"""
Reflective agent tools — read-heavy queries and identity-resolution actions
used by the nightly / on-demand reflective agent to review cat records,
surface anomalies, and resolve provisional identity links.
"""

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta

import requests

from db.store import (
    get_sqlite_connection,
    get_chroma_collection,
    SQLITE_PATH,
)
from tools.embedder import embed_image


# ─────────────────────────────────────────────
# Read tools
# ─────────────────────────────────────────────

def get_cat_dossier(cat_id: str) -> dict:
    """
    Assemble the full context object for a single cat:
      - prototype embedding (from ChromaDB)
      - best frames path
      - visit summary (count, first/last seen, description)
      - confidence trajectory (per-visit confidence over time)
      - gap periods (intervals > 48 h between consecutive visits)
      - human corrections (resolved links involving this cat)
      - provisional links (unresolved links involving this cat)
    """
    conn = get_sqlite_connection()

    # ── Cat record ──
    cat_row = conn.execute(
        "SELECT * FROM cats WHERE cat_id = ?", (cat_id,)
    ).fetchone()
    if not cat_row:
        conn.close()
        return {"error": f"Cat {cat_id} not found"}

    dossier = {
        "cat_id": cat_id,
        "first_seen": cat_row["first_seen"],
        "last_seen": cat_row["last_seen"],
        "visit_count": cat_row["visit_count"],
        "description": cat_row["description"],
        "status": cat_row["status"],
        "health_flags": json.loads(cat_row["health_flags"] or "[]"),
    }

    # ── Visits: confidence trajectory + gap detection ──
    visits = conn.execute(
        "SELECT timestamp, confidence, best_frame_path FROM visits "
        "WHERE cat_id = ? ORDER BY timestamp ASC",
        (cat_id,),
    ).fetchall()

    confidence_trajectory = [
        {"timestamp": v["timestamp"], "confidence": v["confidence"]}
        for v in visits
    ]
    dossier["confidence_trajectory"] = confidence_trajectory

    best_frames = [
        v["best_frame_path"] for v in visits if v["best_frame_path"]
    ]
    dossier["best_frames"] = best_frames

    # Gap periods: consecutive visits > 48 hours apart
    gaps = []
    for i in range(1, len(visits)):
        try:
            t_prev = datetime.fromisoformat(visits[i - 1]["timestamp"])
            t_curr = datetime.fromisoformat(visits[i]["timestamp"])
            delta = t_curr - t_prev
            if delta > timedelta(hours=48):
                gaps.append({
                    "from": visits[i - 1]["timestamp"],
                    "to": visits[i]["timestamp"],
                    "gap_hours": round(delta.total_seconds() / 3600, 1),
                })
        except (ValueError, TypeError):
            continue
    dossier["gap_periods"] = gaps

    # ── Human corrections (resolved links) ──
    corrections = conn.execute(
        "SELECT * FROM links WHERE resolved = 1 "
        "AND (candidate_cat_id = ? OR linked_to_cat_id = ?)",
        (cat_id, cat_id),
    ).fetchall()
    dossier["human_corrections"] = [dict(r) for r in corrections]

    # ── Provisional links (unresolved) ──
    provisional = conn.execute(
        "SELECT * FROM links WHERE resolved = 0 "
        "AND (candidate_cat_id = ? OR linked_to_cat_id = ?)",
        (cat_id, cat_id),
    ).fetchall()
    dossier["provisional_links"] = [dict(r) for r in provisional]

    conn.close()

    # ── ChromaDB prototype embedding ──
    collection = get_chroma_collection()
    chroma_result = collection.get(
        ids=[cat_id], include=["embeddings", "metadatas"]
    )
    if chroma_result["ids"]:
        dossier["prototype_embedding"] = chroma_result["embeddings"][0]
        dossier["chroma_metadata"] = chroma_result["metadatas"][0]
    else:
        dossier["prototype_embedding"] = None
        dossier["chroma_metadata"] = None

    return dossier


def get_cat_history(cat_id: str, limit: int = 20) -> list[dict]:
    """
    Return the most recent visit records for a cat, newest first.
    """
    conn = get_sqlite_connection()
    rows = conn.execute(
        "SELECT * FROM visits WHERE cat_id = ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (cat_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_absent_cats(days: int = 7) -> list[dict]:
    """
    Return cats whose last_seen is older than *days* days ago.
    Only includes cats with status = 'active'.
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = get_sqlite_connection()
    rows = conn.execute(
        "SELECT cat_id, first_seen, last_seen, visit_count, description "
        "FROM cats WHERE last_seen < ? AND status = 'active' "
        "ORDER BY last_seen ASC",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_health_trends(cat_id: str) -> list[dict]:
    """
    Extract body_condition and health_flags across visits, ordered by time.
    Useful for spotting declining condition or recurring flags.
    """
    conn = get_sqlite_connection()
    rows = conn.execute(
        "SELECT timestamp, body_condition, health_flags "
        "FROM visits WHERE cat_id = ? ORDER BY timestamp ASC",
        (cat_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "timestamp": r["timestamp"],
            "body_condition": r["body_condition"],
            "health_flags": json.loads(r["health_flags"] or "[]"),
        }
        for r in rows
    ]


def get_overnight_summary() -> list[dict]:
    """
    All visits in the last 12 hours, newest first.
    """
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    conn = get_sqlite_connection()
    rows = conn.execute(
        "SELECT v.*, c.description FROM visits v "
        "JOIN cats c ON v.cat_id = c.cat_id "
        "WHERE v.timestamp > ? ORDER BY v.timestamp DESC",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_uncertain_matches() -> list[dict]:
    """
    Visits with low confidence that have not yet been resolved via a link.

    The confidence column stores text labels ('low', 'medium', 'high').
    'low' corresponds to the uncertainty band where identity matching was
    not confident enough to be definitive.

    Joins against the links table to exclude visits whose candidate link
    has already been resolved.
    """
    conn = get_sqlite_connection()
    rows = conn.execute(
        """
        SELECT v.visit_id, v.cat_id, v.timestamp, v.confidence,
               v.best_frame_path, v.camera, v.behavior,
               c.description
        FROM visits v
        JOIN cats c ON v.cat_id = c.cat_id
        WHERE v.confidence = 'low'
          AND NOT EXISTS (
              SELECT 1 FROM links l
              WHERE l.candidate_cat_id = v.cat_id
                AND l.resolved = 1
          )
        ORDER BY v.timestamp DESC
        """,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_provisional_links() -> list[dict]:
    """
    All unresolved rows from the links table, enriched with cat descriptions.
    """
    conn = get_sqlite_connection()
    rows = conn.execute(
        """
        SELECT l.*,
               c1.description AS candidate_description,
               c2.description AS linked_to_description
        FROM links l
        JOIN cats c1 ON l.candidate_cat_id = c1.cat_id
        JOIN cats c2 ON l.linked_to_cat_id = c2.cat_id
        WHERE l.resolved = 0
        ORDER BY l.created_at DESC
        """,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Write / action tools
# ─────────────────────────────────────────────

def resolve_link(link_id: int, confirmed: bool) -> dict:
    """
    Resolve a provisional link.

    If confirmed=True:
      - Sets resolution to 'merged'
      - Reassigns visits from candidate_cat_id to linked_to_cat_id
      - Triggers prototype recompute for the surviving cat
    If confirmed=False:
      - Sets resolution to 'rejected'
      - No visit reassignment

    Returns a summary of what was done.
    """
    conn = get_sqlite_connection()

    link = conn.execute(
        "SELECT * FROM links WHERE link_id = ?", (link_id,)
    ).fetchone()
    if not link:
        conn.close()
        return {"error": f"Link {link_id} not found"}

    if link["resolved"]:
        conn.close()
        return {"error": f"Link {link_id} already resolved as '{link['resolution']}'"}

    candidate_id = link["candidate_cat_id"]
    target_id = link["linked_to_cat_id"]
    resolution = "merged" if confirmed else "rejected"

    conn.execute(
        "UPDATE links SET resolved = 1, resolution = ? WHERE link_id = ?",
        (resolution, link_id),
    )

    result = {
        "link_id": link_id,
        "resolution": resolution,
        "candidate_cat_id": candidate_id,
        "linked_to_cat_id": target_id,
    }

    if confirmed:
        # Reassign all visits from candidate to target
        cursor = conn.execute(
            "UPDATE visits SET cat_id = ? WHERE cat_id = ?",
            (target_id, candidate_id),
        )
        visits_moved = cursor.rowcount

        # Absorb visit count into target, mark candidate inactive
        conn.execute(
            "UPDATE cats SET visit_count = visit_count + "
            "(SELECT visit_count FROM cats WHERE cat_id = ?), "
            "last_seen = MAX(last_seen, "
            "(SELECT last_seen FROM cats WHERE cat_id = ?)) "
            "WHERE cat_id = ?",
            (candidate_id, candidate_id, target_id),
        )
        conn.execute(
            "UPDATE cats SET status = 'merged' WHERE cat_id = ?",
            (candidate_id,),
        )

        result["visits_moved"] = visits_moved
        result["note"] = (
            f"Merged {candidate_id} into {target_id}. "
            f"Call recompute_prototype('{target_id}') to update the embedding."
        )

    conn.commit()
    conn.close()
    return result


def recompute_prototype(cat_id: str) -> dict:
    """
    Re-average confirmed visit embeddings in ChromaDB for a cat.

    Gathers all best_frame_path entries from the cat's visits, embeds each,
    and stores the mean vector as the new prototype.

    This is expensive (one embed call per visit with a frame) — call only
    after merges or when drift is suspected.
    """
    conn = get_sqlite_connection()

    cat = conn.execute(
        "SELECT * FROM cats WHERE cat_id = ?", (cat_id,)
    ).fetchone()
    if not cat:
        conn.close()
        return {"error": f"Cat {cat_id} not found"}

    frame_rows = conn.execute(
        "SELECT best_frame_path FROM visits "
        "WHERE cat_id = ? AND best_frame_path IS NOT NULL "
        "ORDER BY timestamp ASC",
        (cat_id,),
    ).fetchall()
    conn.close()

    frame_paths = [r["best_frame_path"] for r in frame_rows]
    if not frame_paths:
        return {"error": f"No frames found for {cat_id} — cannot recompute"}

    # Embed all frames and average
    vectors = []
    errors = []
    for path in frame_paths:
        try:
            vectors.append(embed_image(path))
        except Exception as e:
            errors.append({"path": path, "error": str(e)})

    if not vectors:
        return {
            "error": "All frame embeddings failed",
            "frame_errors": errors,
        }

    dim = len(vectors[0])
    n = len(vectors)
    averaged = [sum(v[i] for v in vectors) / n for i in range(dim)]

    # Update ChromaDB
    collection = get_chroma_collection()
    existing = collection.get(ids=[cat_id], include=["metadatas"])

    if existing["ids"]:
        metadata = existing["metadatas"][0]
        collection.update(
            ids=[cat_id],
            embeddings=[averaged],
            metadatas=[metadata],
        )
    else:
        # Cat was merged in but never had a ChromaDB entry — create one
        cat_conn = get_sqlite_connection()
        cat_row = cat_conn.execute(
            "SELECT * FROM cats WHERE cat_id = ?", (cat_id,)
        ).fetchone()
        cat_conn.close()
        collection.add(
            ids=[cat_id],
            embeddings=[averaged],
            metadatas={
                "first_seen": cat_row["first_seen"] if cat_row else "",
                "coat_color": "unknown",
                "eye_color": "unknown",
                "coat_pattern": "unknown",
                "best_frame_path": frame_paths[-1],
            },
        )

    return {
        "cat_id": cat_id,
        "frames_embedded": len(vectors),
        "frames_failed": len(errors),
        "frame_errors": errors if errors else None,
        "embedding_dimension": dim,
    }


# ─────────────────────────────────────────────
# Coding-model analysis bridge
# ─────────────────────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
ANALYSIS_TIMEOUT = int(os.getenv("ANALYSIS_TIMEOUT", "30"))

# Schema description injected into the coding-model prompt so it knows the
# table structure without needing to introspect the database at runtime.
_SCHEMA_HINT = textwrap.dedent("""\
    Tables in the SQLite database:

    cats(cat_id TEXT PK, first_seen TEXT, last_seen TEXT, visit_count INTEGER,
         description TEXT, status TEXT, health_flags TEXT)

    visits(visit_id INTEGER PK, cat_id TEXT FK->cats, timestamp TEXT,
           confidence TEXT ('low','medium','high'), best_frame_path TEXT,
           camera TEXT, behavior TEXT, body_condition TEXT, health_flags TEXT,
           clip_path TEXT, temperature_f REAL, precipitation_mm REAL,
           windspeed_mph REAL, weather_code INTEGER, notes TEXT)

    alerts(alert_id INTEGER PK, cat_id TEXT, timestamp TEXT,
           alert_type TEXT, detail TEXT, resolved INTEGER DEFAULT 0)

    links(link_id INTEGER PK, candidate_cat_id TEXT, linked_to_cat_id TEXT,
          link_confidence REAL, resolved INTEGER, resolution TEXT, created_at TEXT)
""")


def generate_and_run_analysis(task_description: str) -> str:
    """
    Bridge to the local coding model (Ollama qwen2.5-coder:7b).

    Accepts a plain-language analysis task, asks the coding model to produce
    runnable Python (pandas / numpy / scipy) that queries the catmonitor
    SQLite database, executes the code in a sandboxed subprocess, and returns
    the captured stdout.

    On any failure (timeout, syntax error, runtime exception, Ollama
    unreachable) a descriptive error string is returned so the reasoning
    model can decide how to proceed.
    """

    # ── 1. Ask the coding model for Python code ──
    prompt = textwrap.dedent(f"""\
        You are a data-analysis code generator.

        {_SCHEMA_HINT}

        The database file is located at: {SQLITE_PATH}

        Write a single Python script that answers the following task.
        Use only the standard library plus pandas, numpy, and scipy.
        Connect to the SQLite database with sqlite3 (use the path above).
        Print results to stdout — do NOT write files or show plots.
        Return ONLY the Python code with no markdown fences or explanation.

        Task: {task_description}
    """)

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
    except (requests.ConnectionError, requests.exceptions.InvalidURL):
        return f"Error: cannot reach Ollama at {OLLAMA_URL}. Is it running?"
    except requests.Timeout:
        return "Error: Ollama request timed out after 120 s."
    except requests.HTTPError as e:
        return f"Error: Ollama returned HTTP {resp.status_code}: {e}"
    except requests.RequestException as e:
        return f"Error: Ollama request failed: {e}"

    code = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if the model included them despite instructions
    if code.startswith("```"):
        lines = code.splitlines()
        # Remove opening fence (```python or ```)
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)

    # ── 2. Execute the generated code in a subprocess ──
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=ANALYSIS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: analysis code exceeded {ANALYSIS_TIMEOUT}s timeout.\n"
            f"Generated code:\n{code}"
        )

    if result.returncode != 0:
        return (
            f"Error: analysis code failed (exit {result.returncode}).\n"
            f"stderr:\n{result.stderr.strip()}\n"
            f"Generated code:\n{code}"
        )

    output = result.stdout.strip()
    if not output:
        return (
            "Error: analysis code produced no output.\n"
            f"Generated code:\n{code}"
        )

    return output
