import time
from pathlib import Path


out = Path("tmp/background_persist_result.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("started\n", encoding="utf-8")
time.sleep(8)
out.write_text("finished\n", encoding="utf-8")
