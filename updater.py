"""Replace a stopped worker installation with a downloaded GitHub Release."""

from __future__ import annotations

import shutil
import json
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path


def update(package_dir: Path, archive_urls: list[str], process_id: int) -> None:
    log_path = package_dir.parent / "football-worker-update.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Starting worker update\n")
            with tempfile.TemporaryDirectory(prefix="football-worker-update-") as temp:
                temp_dir = Path(temp)
                archive_path = temp_dir / "worker.zip"
                with archive_path.open("wb") as output:
                    for archive_url in archive_urls:
                        log.write(f"Downloading {archive_url}\n")
                        with urllib.request.urlopen(archive_url, timeout=300) as response:
                            shutil.copyfileobj(response, output)
                extracted = temp_dir / "extracted"
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(extracted)
                candidates = list(extracted.glob("**/football-worker.exe"))
                if len(candidates) != 1:
                    raise RuntimeError("Release must contain exactly one football-worker.exe")
                new_package = candidates[0].parent
                deadline = time.time() + 60
                while time.time() < deadline:
                    try:
                        import os
                        os.kill(process_id, 0)
                    except OSError:
                        break
                    time.sleep(1)
                backup = package_dir.with_name(f"{package_dir.name}.old-{int(time.time())}")
                package_dir.rename(backup)
                shutil.copytree(new_package, package_dir)
                log.write(f"Installed package at {package_dir}\n")
        executable = package_dir / "football-worker.exe"
        import subprocess
        subprocess.Popen([str(executable)], cwd=package_dir.parent, close_fds=True)
    except Exception:
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        raise


if __name__ == "__main__":
    update(Path(sys.argv[1]), json.loads(sys.argv[2]), int(sys.argv[3]))
