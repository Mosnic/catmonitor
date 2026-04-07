import sqlite3
import json
import os
import chromadb
from datetime import datetime
from pathlib import Path

from tools.embedder import embed_image
from tools.weather import get_weather_for_visit

# Paths
DB_DIR = Path.home() / "catmonitor" / "data"
DB_DIR.mkdir(exist_ok=True)

SQLITE_PATH = DB_DIR / "catmonitor.db"
CHROMA_PATH = DB_DIR / "chromadb"

# Similarity thresholds (configurable via env vars)
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.25"))
UNCERTAIN_THRESHOLD = float(os.getenv("UNCERTAIN_THRESHOLD", "0.45"))


# ─────────────────────────────────────────────
# SQLite - visits and facts
# ─────────────────────────────────────────────

def get_sqlite_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite():
    """Create tables if they don't exist, and migrate existing schema."""
    conn = get_sqlite_connection()

    # --- Base tables ---
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cats (
            cat_id      TEXT PRIMARY KEY,
            first_seen  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            visit_count INTEGER DEFAULT 1,
            description TEXT,
            health_flags TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS visits (
            visit_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            cat_id           TEXT NOT NULL,
            timestamp        TEXT NOT NULL,
            camera           TEXT,
            clip_path        TEXT,
            behavior         TEXT,
            body_condition   TEXT,
            temperature_f    REAL,
            precipitation_mm REAL,
            windspeed_mph    REAL,
            weather_code     INTEGER,
            health_flags     TEXT DEFAULT '[]',
            lighting         TEXT,
            confidence       REAL,
            notes            TEXT,
            raw_json         TEXT,
            FOREIGN KEY (cat_id) REFERENCES cats(cat_id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            cat_id      TEXT,
            timestamp   TEXT NOT NULL,
            alert_type  TEXT NOT NULL,
            detail      TEXT,
            resolved    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS links (
            link_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_cat_id  TEXT NOT NULL,
            linked_to_cat_id  TEXT NOT NULL,
            link_confidence   REAL NOT NULL,
            created_at        TEXT NOT NULL,
            resolved          INTEGER DEFAULT 0,
            resolution        TEXT,
            FOREIGN KEY (candidate_cat_id) REFERENCES cats(cat_id),
            FOREIGN KEY (linked_to_cat_id) REFERENCES cats(cat_id)
        );
    """)

    # --- Migrations: add new columns to existing tables if absent ---
    existing_cats_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(cats)")
    }
    if "status" not in existing_cats_cols:
        conn.execute(
            "ALTER TABLE cats ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
        )
        print("Migration: added cats.status")

    existing_visits_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(visits)")
    }
    if "best_frame_path" not in existing_visits_cols:
        conn.execute(
            "ALTER TABLE visits ADD COLUMN best_frame_path TEXT"
        )
        print("Migration: added visits.best_frame_path")

    for col, col_type in [
        ("temperature_f",    "REAL"),
        ("precipitation_mm", "REAL"),
        ("windspeed_mph",    "REAL"),
        ("weather_code",     "INTEGER"),
    ]:
        if col not in existing_visits_cols:
            conn.execute(f"ALTER TABLE visits ADD COLUMN {col} {col_type}")
            print(f"Migration: added visits.{col}")

    # --- Indexes for alert querying ---
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_alerts_type
            ON alerts(alert_type);
        CREATE INDEX IF NOT EXISTS idx_alerts_resolved
            ON alerts(resolved);
        CREATE INDEX IF NOT EXISTS idx_alerts_cat_timestamp
            ON alerts(cat_id, timestamp);
    """)

    conn.commit()
    conn.close()
    print(f"SQLite initialized at {SQLITE_PATH}")


# ─────────────────────────────────────────────
# ChromaDB - cat identity vectors
# ─────────────────────────────────────────────

def get_chroma_collection():
    """Get or create the cats vector collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name="cats",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ─────────────────────────────────────────────
# Core identity matching
# ─────────────────────────────────────────────

def find_matching_cat(best_frame_path: str, analysis: dict) -> tuple[str | None, float, str]:
    """
    Search ChromaDB for the nearest matching cat profile using image embedding.

    Embeds best_frame_path with Qwen3-VL-Embedding-2B, then queries by vector.
    Metadata hard-filters on coat_color; eye_color filter is relaxed when unknown.

    Returns (cat_id, distance, state) where state is one of:
      "known"     — distance < MATCH_THRESHOLD
      "uncertain" — MATCH_THRESHOLD <= distance < UNCERTAIN_THRESHOLD
      "new"       — distance >= UNCERTAIN_THRESHOLD or no candidates found
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        return None, 1.0, "new"

    cat = analysis["cats"][0] if analysis.get("cats") else {}
    coat_color = cat.get("coat_color", "unknown")
    eye_color = cat.get("eye_color", "unknown")

    # Embed the best frame image
    query_vector = embed_image(best_frame_path)

    # Build metadata filter — relax eye_color constraint when unknown
    if eye_color != "unknown":
        where_filter = {
            "$and": [
                {"coat_color": {"$eq": coat_color}},
                {"eye_color": {"$eq": eye_color}}
            ]
        }
    else:
        where_filter = {"coat_color": {"$eq": coat_color}}

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            where=where_filter,
            n_results=1
        )
    except Exception as e:
        print(f"ChromaDB query failed: {e}. Treating as new cat.")
        return None, 1.0, "new"

    if not results["ids"][0]:
        return None, 1.0, "new"

    distance = results["distances"][0][0]
    cat_id = results["ids"][0][0]

    if distance < MATCH_THRESHOLD:
        return cat_id, distance, "known"
    elif distance < UNCERTAIN_THRESHOLD:
        return cat_id, distance, "uncertain"
    else:
        return None, distance, "new"


def create_cat_profile(cat_id: str, best_frame_path: str, analysis: dict, timestamp: datetime):
    """
    Create a new cat profile in both ChromaDB and SQLite.

    Embeds best_frame_path to produce the initial prototype vector.
    Metadata fields are stored alongside for hard-filter queries.
    """
    ts = timestamp.isoformat()
    cat = analysis["cats"][0] if analysis.get("cats") else {}

    # Embed the best frame — this becomes the initial prototype
    embedding = embed_image(best_frame_path)

    collection = get_chroma_collection()
    collection.add(
        ids=[cat_id],
        embeddings=[embedding],
        metadatas=[{
            "first_seen": ts,
            "coat_color": cat.get("coat_color", "unknown"),
            "eye_color": cat.get("eye_color", "unknown"),
            "coat_pattern": cat.get("coat_pattern", "unknown"),
            "best_frame_path": best_frame_path,
        }]
    )

    # Add to SQLite — description stored as human-readable coat summary
    description = (
        f"{cat.get('coat_color', 'unknown')} "
        f"{cat.get('coat_pattern', 'unknown')} "
        f"{cat.get('coat_length', 'unknown')}-haired"
    )
    conn = get_sqlite_connection()
    conn.execute("""
        INSERT INTO cats (cat_id, first_seen, last_seen, visit_count, description)
        VALUES (?, ?, ?, 1, ?)
    """, (cat_id, ts, ts, description))
    conn.commit()
    conn.close()

    print(f"New cat profile created: {cat_id}")


def update_cat_profile(cat_id: str, best_frame_path: str, timestamp: datetime):
    """
    Update an existing cat's profile after a confirmed visit.

    SQLite: bumps last_seen and visit_count.
    ChromaDB: updates the prototype using a cumulative moving average so each
              new confirmed frame contributes 1/n weight, making the prototype
              increasingly stable over time rather than bouncing with every visit.
    """
    ts = timestamp.isoformat()

    # ── SQLite update — fetch visit_count after incrementing ──
    conn = get_sqlite_connection()
    conn.execute("""
        UPDATE cats
        SET last_seen = ?, visit_count = visit_count + 1
        WHERE cat_id = ?
    """, (ts, cat_id))
    conn.commit()
    row = conn.execute(
        "SELECT visit_count FROM cats WHERE cat_id = ?", (cat_id,)
    ).fetchone()
    conn.close()

    # ── Embedding averaging ──
    collection = get_chroma_collection()
    existing = collection.get(ids=[cat_id], include=["embeddings", "metadatas"])

    if not existing["ids"]:
        # Profile missing from ChromaDB — nothing to average against
        print(f"Warning: no ChromaDB entry found for {cat_id}, skipping averaging")
        return

    n = row["visit_count"]  # already incremented — new frame gets weight 1/n
    stored_vector = existing["embeddings"][0]
    new_vector = embed_image(best_frame_path)
    averaged = [((n - 1) * s + v) / n for s, v in zip(stored_vector, new_vector)]

    # Upsert the averaged prototype back; preserve existing metadata
    metadata = existing["metadatas"][0]
    if best_frame_path:
        metadata["best_frame_path"] = best_frame_path
    collection.update(
        ids=[cat_id],
        embeddings=[averaged],
        metadatas=[metadata],
    )
    print(f"Updated prototype for {cat_id} (embedding averaged)")


def log_visit(
    cat_id: str,
    analysis: dict,
    timestamp: datetime,
    clip_path: str = "",
    best_frame_path: str = "",
    weather: dict | None = None,
) -> int:
    """
    Write a visit record to SQLite.
    Returns the visit_id.
    """
    ts = timestamp.isoformat()
    cat = analysis["cats"][0] if analysis.get("cats") else {}

    conn = get_sqlite_connection()
    cursor = conn.execute("""
        INSERT INTO visits 
        (cat_id, timestamp, camera, clip_path, behavior, body_condition,
         temperature_f, precipitation_mm, windspeed_mph, weather_code,
         health_flags, lighting, confidence, notes, raw_json, best_frame_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        cat_id,
        ts,
        analysis.get("camera", "unknown"),
        clip_path,
        cat.get("behavior", "unknown"),
        cat.get("body_condition", "unknown"),
        weather.get("temperature_f")    if weather else None,
        weather.get("precipitation_mm") if weather else None,
        weather.get("windspeed_mph")    if weather else None,
        weather.get("weather_code")     if weather else None,
        json.dumps(cat.get("health_flags", [])),
        analysis.get("lighting", "unknown"),
        cat.get("confidence"),
        analysis.get("notes", ""),
        json.dumps(analysis),
        best_frame_path or None,
    ))
    conn.commit()
    visit_id = cursor.lastrowid
    conn.close()
    return visit_id


# ─────────────────────────────────────────────
# Main entry point - process one analysis result
# ─────────────────────────────────────────────

def log_uncertain_link(candidate_cat_id: str, linked_to_cat_id: str, distance: float, timestamp: datetime):
    """
    Write a provisional link record when a visit falls in the uncertainty band.
    """
    ts = timestamp.isoformat()
    # link_confidence expressed as similarity (1 - distance) for readability
    confidence = round(1.0 - distance, 4)
    conn = get_sqlite_connection()
    conn.execute("""
        INSERT INTO links (candidate_cat_id, linked_to_cat_id, link_confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (candidate_cat_id, linked_to_cat_id, confidence, ts))
    conn.commit()
    conn.close()


def process_visit(
    analysis: dict,
    clip_path: str = "",
    best_frame_path: str = "",
    timestamp: datetime | None = None,
) -> dict:
    """
    Main entry point — process one structured vision analysis result.

    timestamp should be derived from the clip file's mtime so it reflects
    the actual recording time rather than processing time. Falls back to
    datetime.now() if not provided.

    Determines cat identity, logs the visit, and for uncertain matches
    also writes a provisional link record for later review.

    best_frame_path is required for identity matching. If absent, the visit
    is still logged but matching is skipped and the cat is treated as new.

    Returns a dict with keys:
      status        — "known_cat" | "new_cat" | "uncertain_cat" | "no_cat"
      cat_id        — assigned cat ID
      visit_id      — SQLite visit row ID
      distance      — ChromaDB cosine distance (lower = more similar)
      candidate_id  — nearest candidate cat_id (uncertain only)
    """
    import uuid

    if timestamp is None:
        timestamp = datetime.now()

    if not analysis.get("cat_present"):
        return {"status": "no_cat", "message": "No cat detected in image"}

    # Fetch weather for the clip timestamp — fails gracefully to None
    weather = None
    try:
        weather = get_weather_for_visit(timestamp)
    except Exception as e:
        print(f"Weather fetch failed: {e}")

    link_candidate_id = None

    if not best_frame_path:
        # No frame to embed — log as new cat without matching
        print("Warning: no best_frame_path provided, skipping identity match")
        match_state = "new"
        distance = 1.0
        cat_id = None
    else:
        cat_id, distance, match_state = find_matching_cat(best_frame_path, analysis)

    if match_state == "known":
        status = "known_cat"
        update_cat_profile(cat_id, best_frame_path, timestamp)
        print(f"Known cat: {cat_id} (distance: {distance:.3f})")

    elif match_state == "uncertain":
        # Log the visit against the candidate — don't mint a new cat ID.
        # The visit's confidence field records the uncertainty; low-confidence
        # visits can be queried for human review without fragmenting the cat population.
        link_candidate_id = cat_id
        status = "uncertain_cat"
        print(
            f"Uncertain match (distance: {distance:.3f}) — "
            f"logging against candidate {cat_id} for review"
        )

    else:  # "new"
        status = "new_cat"
        cat_id = f"cat_{uuid.uuid4().hex[:8]}"
        if best_frame_path:
            create_cat_profile(cat_id, best_frame_path, analysis, timestamp)
        else:
            # Minimal SQLite-only profile — no ChromaDB entry until a frame exists
            ts = timestamp.isoformat()
            cat = analysis["cats"][0] if analysis.get("cats") else {}
            description = (
                f"{cat.get('coat_color', 'unknown')} "
                f"{cat.get('coat_pattern', 'unknown')} "
                f"{cat.get('coat_length', 'unknown')}-haired"
            )
            conn = get_sqlite_connection()
            conn.execute("""
                INSERT INTO cats (cat_id, first_seen, last_seen, visit_count, description)
                VALUES (?, ?, ?, 1, ?)
            """, (cat_id, ts, ts, description))
            conn.commit()
            conn.close()
        print(f"New cat detected (distance from nearest: {distance:.3f})")

    visit_id = log_visit(cat_id, analysis, timestamp, clip_path, best_frame_path, weather)

    result = {
        "status": status,
        "cat_id": cat_id,
        "visit_id": visit_id,
        "distance": distance,
        "weather": weather,  # None if fetch failed — agent should handle gracefully
    }
    if link_candidate_id:
        result["candidate_id"] = link_candidate_id

    return result


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    init_sqlite()

    # These paths must exist to run the test — point at real frames
    TEST_FRAME_1 = "/tmp/test_frame_orange.jpg"
    TEST_FRAME_2 = "/tmp/test_frame_black.jpg"

    test_timestamp = datetime.now()

    base_cat = {
        "coat_color": "orange",
        "coat_pattern": "solid",
        "coat_length": "short",
        "size": "medium",
        "build": "normal",
        "distinctive_markings": [],
        "eye_color": "yellow",
        "body_condition": "good",
        "behavior": "eating",
        "health_flags": [],
        "confidence": "high"
    }

    test_analysis = {
        "cat_present": True,
        "cat_count": 1,
        "cats": [dict(base_cat)],
        "camera": "platform_front",
        "lighting": "good",
        "notes": ""
    }

    print("\n--- Visit 1 (new cat) ---")
    r1 = process_visit(test_analysis, clip_path="/clips/clip1.mp4", best_frame_path=TEST_FRAME_1, timestamp=test_timestamp)
    print(r1)

    print("\n--- Visit 2 (same cat — prototype averaged) ---")
    r2 = process_visit(test_analysis, clip_path="/clips/clip2.mp4", best_frame_path=TEST_FRAME_1, timestamp=test_timestamp)
    print(r2)

    print("\n--- Visit 3 (different cat) ---")
    test_analysis["cats"][0] = dict(base_cat, coat_color="black", eye_color="green")
    r3 = process_visit(test_analysis, clip_path="/clips/clip3.mp4", best_frame_path=TEST_FRAME_2, timestamp=test_timestamp)
    print(r3)
