"""Replace a stopped worker installation with a downloaded GitHub Release."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


def update(package_dir: Path, archive_url: str, process_id: int) -> None:
    with tempfile.TemporaryDirectory(prefix="football-worker-update-") as temp:
        temp_dir = Path(temp)
        archive_path = temp_dir / "worker.zip"
        urllib.request.urlretrieve(archive_url, archive_path)
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
        backup = package_dir.with_name(package_dir.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        package_dir.rename(backup)
        shutil.copytree(new_package, package_dir)
        shutil.rmtree(backup)
    executable = package_dir / "football-worker.exe"
    import subprocess
    subprocess.Popen([str(executable)], cwd=package_dir, close_fds=True)


if __name__ == "__main__":
    update(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
