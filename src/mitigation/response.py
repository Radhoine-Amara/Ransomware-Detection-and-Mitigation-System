"""
Automated mitigation and response actions.

When the ML model detects a ransomware-like process this module provides
helpers to:
- Kill the offending process
- Quarantine (move) suspicious files to an isolated directory
- Snapshot (copy) critical files before they can be encrypted
- Revoke write permissions on protected directories
"""

import logging
import os
import shutil
import signal
import stat
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process isolation
# ---------------------------------------------------------------------------

def kill_process(pid: int, force: bool = False) -> bool:
    """Terminate a process by PID.

    Args:
        pid: Target process identifier.
        force: If ``True`` send SIGKILL instead of SIGTERM (Unix only).

    Returns:
        ``True`` if the process was successfully terminated, ``False`` if it
        was not found.
    """
    try:
        proc = psutil.Process(pid)
        if force:
            proc.kill()
        else:
            proc.terminate()
        proc.wait(timeout=5)
        logger.warning("Terminated process PID=%d (force=%s)", pid, force)
        return True
    except psutil.NoSuchProcess:
        logger.info("Process PID=%d not found (already exited?).", pid)
        return False
    except psutil.AccessDenied:
        logger.error("Access denied when trying to terminate PID=%d.", pid)
        return False


# ---------------------------------------------------------------------------
# File quarantine
# ---------------------------------------------------------------------------

def quarantine_file(src_path: str, quarantine_dir: str) -> str:
    """Move a suspicious file into a quarantine directory.

    The file is renamed with a timestamp prefix to avoid collisions.

    Args:
        src_path: Absolute path of the file to quarantine.
        quarantine_dir: Destination directory for quarantined files.

    Returns:
        Path of the file inside the quarantine directory.
    """
    os.makedirs(quarantine_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    basename = os.path.basename(src_path)
    dest_path = os.path.join(quarantine_dir, f"{timestamp}_{basename}")
    shutil.move(src_path, dest_path)
    logger.warning("Quarantined %s -> %s", src_path, dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# File snapshot / backup
# ---------------------------------------------------------------------------

def snapshot_directory(src_dir: str, snapshot_dir: str) -> str:
    """Create a timestamped backup copy of *src_dir*.

    Args:
        src_dir: Directory to back up.
        snapshot_dir: Parent directory where the snapshot folder will be
            created.

    Returns:
        Path to the created snapshot directory.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    dest = os.path.join(snapshot_dir, f"snapshot_{timestamp}")
    shutil.copytree(src_dir, dest)
    logger.info("Snapshot of %s created at %s", src_dir, dest)
    return dest


# ---------------------------------------------------------------------------
# Permission hardening
# ---------------------------------------------------------------------------

def revoke_write_permissions(path: str):
    """Recursively remove write permissions from *path*.

    This can be used to protect a directory of important files from being
    encrypted or deleted by a ransomware process.

    Args:
        path: File or directory whose write bits should be cleared.
    """
    for root, dirs, files in os.walk(path):
        for name in files + dirs:
            target = os.path.join(root, name)
            try:
                current = stat.S_IMODE(os.lstat(target).st_mode)
                new_mode = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                os.chmod(target, new_mode)
            except OSError as exc:
                logger.error("Could not chmod %s: %s", target, exc)
    logger.info("Revoked write permissions on %s", path)


# ---------------------------------------------------------------------------
# High-level response orchestrator
# ---------------------------------------------------------------------------

def respond(pid: int, suspicious_files: list[str], quarantine_dir: str):
    """Execute a full mitigation response.

    1. Kill the suspicious process.
    2. Quarantine each suspicious file.

    Args:
        pid: PID of the detected ransomware process.
        suspicious_files: List of file paths associated with the process.
        quarantine_dir: Directory to move suspicious files into.
    """
    logger.warning("Initiating mitigation response for PID=%d", pid)
    kill_process(pid)
    for path in suspicious_files:
        if os.path.isfile(path):
            quarantine_file(path, quarantine_dir)
    logger.warning("Mitigation response complete for PID=%d", pid)
