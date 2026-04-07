import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.loop import run_agent
from db.store import init_sqlite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("catmonitor.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def process_existing(folder: Path, dry_run: bool = False):
    """
    Find and process all existing clips in a folder.
    """
    clips = sorted([
        f for f in folder.rglob("*")
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not clips:
        log.info(f"No video clips found in {folder}")
        return

    log.info(f"Found {len(clips)} clips to process")

    for i, clip in enumerate(clips, 1):
        log.info(f"[{i}/{len(clips)}] {clip.name}")

        if dry_run:
            log.info(f"  DRY RUN — skipping")
            continue

        try:
            run_agent(str(clip))
        except Exception as e:
            log.error(f"  Failed: {e}")
            continue

    log.info("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process existing clips in FTP folder"
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing video clips"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List clips without processing them"
    )
    args = parser.parse_args()

    init_sqlite()
    process_existing(Path(args.folder), dry_run=args.dry_run)