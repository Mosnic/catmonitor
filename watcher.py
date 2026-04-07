import os
import random
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

sys.path.insert(0, str(Path(__file__).parent))

from agent.loop import run_agent
from agent.reflective_loop import run_reflective_agent
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
FTP_FOLDER = Path("/mnt/newdrive/srv/files/uploads")  # ← change to your FTP drop folder

# Reflective agent configuration
REFLECTIVE_MIN_HOURS = float(os.getenv("REFLECTIVE_MIN_HOURS", "4"))
REFLECTIVE_MAX_HOURS = float(os.getenv("REFLECTIVE_MAX_HOURS", "6"))
REFLECTIVE_QUIET_MINUTES = float(os.getenv("REFLECTIVE_QUIET_MINUTES", "15"))


class NewClipHandler(FileSystemEventHandler):
    """
    Watches for new video files arriving from the Reolink cameras.
    Triggers the agent loop when a new clip is detected.

    Uses on_moved instead of on_created because the FTP server writes
    a temp file first then renames it to the final filename. The
    MOVED_TO event fires on the final, complete file.
    """

    def __init__(self):
        self.processing = set()  # track files being processed

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return

        if path in self.processing:
            return

        if not path.exists():
            log.warning(f"File not found on create, skipping: {path.name}")
            return

        log.info(f"New clip detected (created): {path.name}")
        self.processing.add(path)
        self._wait_for_file(path)
        self._process_clip(path)

    def on_moved(self, event):
        if event.is_directory:
            return

        path = Path(event.dest_path)  # use dest_path — this is the final filename

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return

        if path in self.processing:
            return

        if not path.exists():
            log.warning(f"File not found after move, skipping: {path.name}")
            return

        log.info(f"New clip detected: {path.name}")
        self.processing.add(path)

        # Wait briefly for file to finish writing before processing
        self._wait_for_file(path)
        self._process_clip(path)

    def _wait_for_file(self, path: Path, timeout: int = 30):
        """
        Wait until file size stops growing — ensures upload is complete
        before we try to process it.
        """
        log.info(f"Waiting for upload to complete: {path.name}")
        previous_size = -1
        stable_count = 0

        for _ in range(timeout):
            if not path.exists():
                log.warning(f"File disappeared while waiting: {path.name}")
                return
            current_size = path.stat().st_size
            if current_size == previous_size:
                stable_count += 1
                if stable_count >= 3:
                    log.info(f"Upload complete: {path.name} ({current_size} bytes)")
                    return
            else:
                stable_count = 0
            previous_size = current_size
            time.sleep(1)

        log.warning(f"Timeout waiting for {path.name} — processing anyway")

    def _process_clip(self, path: Path):
        """Hand the clip to the agent and log the outcome."""
        try:
            log.info(f"Starting agent for: {path.name}")
            summary = run_agent(str(path))
            log.info(f"Agent complete for {path.name}")
        except Exception as e:
            log.error(f"Agent failed for {path.name}: {e}")
        finally:
            self.processing.discard(path)


class ReflectiveScheduler:
    """
    Runs the reflective agent periodically during idle periods.

    Integrated into the watcher so it can directly check whether clip
    processing is active, rather than inferring it from DB timestamps.
    """

    def __init__(self, clip_handler: NewClipHandler):
        self.clip_handler = clip_handler
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="reflective-scheduler"
        )
        self._thread.start()
        log.info(
            f"Reflective scheduler started — "
            f"interval {REFLECTIVE_MIN_HOURS}-{REFLECTIVE_MAX_HOURS}h, "
            f"quiet period {REFLECTIVE_QUIET_MINUTES}m"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _is_idle(self) -> bool:
        """True when no clips are currently being processed."""
        return len(self.clip_handler.processing) == 0

    def _wait_for_quiet(self, timeout_minutes: float = 30) -> bool:
        """Wait until no clips have been processed for REFLECTIVE_QUIET_MINUTES."""
        quiet_needed = timedelta(minutes=REFLECTIVE_QUIET_MINUTES)
        deadline = datetime.now() + timedelta(minutes=timeout_minutes)
        idle_since = None

        while datetime.now() < deadline:
            if self._stop_event.is_set():
                return False

            if self._is_idle():
                if idle_since is None:
                    idle_since = datetime.now()
                elif datetime.now() - idle_since >= quiet_needed:
                    return True
            else:
                idle_since = None  # reset — a clip is being processed

            self._stop_event.wait(30)

        return False

    def _run_loop(self):
        # Wait one interval before the first run so the watcher can
        # process any backlog of clips first.
        interval = random.uniform(REFLECTIVE_MIN_HOURS, REFLECTIVE_MAX_HOURS)
        if self._stop_event.wait(interval * 3600):
            return

        while not self._stop_event.is_set():
            try:
                log.info("Reflective scheduler: waiting for quiet period...")
                if not self._wait_for_quiet():
                    if self._stop_event.is_set():
                        return
                    log.warning(
                        "Reflective scheduler: could not confirm idle, running anyway"
                    )

                log.info("Reflective scheduler: starting reflective agent")
                start = datetime.now()
                summary = run_reflective_agent()
                elapsed = (datetime.now() - start).total_seconds()
                log.info(f"Reflective scheduler: agent finished in {elapsed:.0f}s")

                if summary:
                    first_line = summary.strip().splitlines()[0][:120]
                    log.info(f"Reflective findings: {first_line}")

            except Exception as e:
                log.error(f"Reflective scheduler: agent failed: {e}", exc_info=True)

            # Sleep until next run
            interval = random.uniform(REFLECTIVE_MIN_HOURS, REFLECTIVE_MAX_HOURS)
            next_run = datetime.now() + timedelta(hours=interval)
            log.info(f"Next reflective run scheduled at ~{next_run.strftime('%H:%M')}")
            if self._stop_event.wait(interval * 3600):
                return


def process_existing_clips(folder: Path):
    """Process all existing video files in the folder (recursively)."""
    clips = sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not clips:
        log.info(f"No existing clips found in {folder}")
        return

    log.info(f"Processing {len(clips)} existing clip(s) in {folder}")
    consecutive_failures = 0
    max_consecutive_failures = 5
    for i, clip in enumerate(clips, 1):
        log.info(f"[{i}/{len(clips)}] Processing: {clip.name}")
        try:
            summary = run_agent(str(clip))
            log.info(f"[{i}/{len(clips)}] Complete: {clip.name}")
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            log.error(f"[{i}/{len(clips)}] Failed: {clip.name}: {e}")
            if consecutive_failures >= max_consecutive_failures:
                log.warning(f"{max_consecutive_failures} consecutive failures — server may be down. Waiting 60s...")
                time.sleep(60)
                consecutive_failures = 0  # give it another round of attempts


def start_watcher(folder: Path, enable_reflective: bool = True):
    """Start watching the FTP drop folder."""
    folder.mkdir(parents=True, exist_ok=True)
    if not os.access(folder, os.R_OK):
        log.error(f"No read permission on watch folder: {folder}")
        sys.exit(1)

    log.info(f"Cat Monitor starting...")
    log.info(f"Watching folder: {folder}")
    log.info(f"Waiting for clips from Reolink cameras...")

    handler = NewClipHandler()
    observer = Observer()
    observer.schedule(handler, str(folder), recursive=True)
    observer.start()

    reflective = None
    if enable_reflective:
        reflective = ReflectiveScheduler(handler)
        reflective.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down watcher...")
        if reflective:
            reflective.stop()
        observer.stop()
        observer.join()
        log.info("Cat Monitor stopped.")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cat Monitor — watch for camera clips")
    parser.add_argument("folder", nargs="?", default=str(FTP_FOLDER),
                        help="Folder to watch (default: %(default)s)")
    parser.add_argument("--process-existing", action="store_true",
                        help="Process all existing clips in the folder before watching")
    parser.add_argument("--no-reflective", action="store_true",
                        help="Disable the periodic reflective agent")
    args = parser.parse_args()

    folder = Path(args.folder)

    while True:
        try:
            init_sqlite()
            if args.process_existing:
                process_existing_clips(folder)
                args.process_existing = False  # only on first loop, not on restart
            start_watcher(folder, enable_reflective=not args.no_reflective)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception(f"Watcher error, restarting loop in 10 seconds: {e}")
            time.sleep(10)
