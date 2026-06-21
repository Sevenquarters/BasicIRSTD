import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    pythonw = root / "venv" / "Scripts" / "pythonw.exe"
    if not pythonw.exists():
        pythonw = root / "venv" / "Scripts" / "python.exe"

    cmd = [str(pythonw), "train.py"] + sys.argv[1:]
    process = subprocess.Popen(
        cmd,
        cwd=str(root),
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    print(process.pid)


if __name__ == "__main__":
    main()
