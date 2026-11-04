#!/usr/bin/env python3
"""Generate deterministic 1 TiB virtual 3D shard plans; does NOT launch cloud jobs."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--start-group", type=int, default=0)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not (1 <= a.workers <= 4096): p.error("--workers must be 1..4096")
    if not (0 <= a.start_group < 1024): p.error("--start-group must be 0..1023")
    if not (1 <= a.groups <= 1024-a.start_group): p.error("invalid group count")
    rows = []
    for group in range(a.start_group, a.start_group + a.groups):
        start = group * 1024
        rows.append({
            "group": group,
            "worker_slot": group % a.workers,
            "first_tile_id": start,
            "tile_count": 1024,
            "virtual_input_bytes": 1024 * (2**20),
            "first_tile_coord_xyz": [start % 64, (start//64) % 64, start//4096],
            "command": ["./jx3d_1tib", "--start", str(start), "--tiles", "1024", "--out", f"group-{group:04d}.jxtb"]
        })
    body = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    if a.output:
        a.output.write_text(body)
        print(f"Planned {len(rows)} groups; no remote jobs launched")
    else:
        print(body, end="")

if __name__ == "__main__":
    main()
