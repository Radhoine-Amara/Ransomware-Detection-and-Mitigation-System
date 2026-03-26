"""
System and file-system monitoring using psutil and watchdog.

Collects:
- CPU, memory, disk I/O, and network stats via psutil
- File-system create/modify/delete/rename events via watchdog
"""

import csv
import logging
import os
import time
from datetime import datetime

import psutil
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process / system metrics collector
# ---------------------------------------------------------------------------

class SystemMonitor:
    """Collect system-level metrics at a fixed interval and write to a CSV."""

    DEFAULT_FIELDS = [
        "timestamp",
        "cpu_percent",
        "mem_percent",
        "disk_read_bytes",
        "disk_write_bytes",
        "net_bytes_sent",
        "net_bytes_recv",
        "num_processes",
    ]

    def __init__(self, output_path: str, interval: float = 1.0):
        self.output_path = output_path
        self.interval = interval
        self._running = False

    def _snapshot(self) -> dict:
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "mem_percent": psutil.virtual_memory().percent,
            "disk_read_bytes": disk.read_bytes if disk else 0,
            "disk_write_bytes": disk.write_bytes if disk else 0,
            "net_bytes_sent": net.bytes_sent,
            "net_bytes_recv": net.bytes_recv,
            "num_processes": len(psutil.pids()),
        }

    def run(self, duration: float | None = None):
        """Collect metrics, writing rows to *output_path*.

        Args:
            duration: How many seconds to run. ``None`` runs until stopped.
        """
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self._running = True
        start = time.monotonic()

        with open(self.output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.DEFAULT_FIELDS)
            writer.writeheader()
            while self._running:
                row = self._snapshot()
                writer.writerow(row)
                fh.flush()
                logger.debug("snapshot: %s", row)
                time.sleep(self.interval)
                if duration is not None and (time.monotonic() - start) >= duration:
                    break

        self._running = False

    def stop(self):
        """Signal the monitoring loop to stop."""
        self._running = False


# ---------------------------------------------------------------------------
# File-system event collector
# ---------------------------------------------------------------------------

class FileEventHandler(FileSystemEventHandler):
    """Log file-system events to a CSV file for later feature extraction."""

    FIELDS = ["timestamp", "event_type", "src_path", "dest_path", "is_directory"]

    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._fh = open(output_path, "w", newline="")  # noqa: WPS515
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def _write(self, event_type: str, event):
        dest = getattr(event, "dest_path", "")
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "src_path": event.src_path,
            "dest_path": dest,
            "is_directory": event.is_directory,
        }
        self._writer.writerow(row)
        self._fh.flush()
        logger.debug("file event: %s", row)

    def on_created(self, event):
        self._write("created", event)

    def on_modified(self, event):
        self._write("modified", event)

    def on_deleted(self, event):
        self._write("deleted", event)

    def on_moved(self, event):
        self._write("moved", event)

    def close(self):
        self._fh.close()

    def __del__(self):
        try:
            if not self._fh.closed:
                self._fh.close()
        except Exception:
            pass


class FileSystemMonitor:
    """Watch a directory tree for file-system events."""

    def __init__(self, watch_path: str, output_path: str, recursive: bool = True):
        self.watch_path = watch_path
        self.output_path = output_path
        self.recursive = recursive
        self._handler = None
        self._observer = None

    def start(self):
        self._handler = FileEventHandler(self.output_path)
        self._observer = Observer()
        self._observer.schedule(self._handler, self.watch_path, recursive=self.recursive)
        self._observer.start()
        logger.info("Watching %s (recursive=%s)", self.watch_path, self.recursive)

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
        if self._handler:
            self._handler.close()
        logger.info("File-system monitor stopped.")
