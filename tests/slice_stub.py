"""Stand-in slice for tests/test_prepare_data.py; kept import-light because spawned
child processes import it."""

import json
import os
import time
from pathlib import Path


def fake_slice(name, rank, world, budget, a):
    """Stand-in for prepare_data._run_slice: logs its run interval, writes a finished state."""
    out = Path(a["out"])
    t0 = time.time()
    time.sleep(0.4)
    if name == "crash":
        os._exit(3)
    (out / "state").mkdir(parents=True, exist_ok=True)
    (out / "state" / f"{name}-{rank:03d}.json").write_text(json.dumps({"tokens": budget, "done": True}))
    with open(out / "intervals.txt", "a") as f:
        f.write(f"{name} {t0} {time.time()}\n")
    os._exit(0)
