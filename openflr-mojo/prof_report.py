#!/usr/bin/env python3
"""Turn an `nsys export --type sqlite` trace into the four tables this
project's optimization log actually reads.

    python3 prof_report.py v2prof.sqlite

Uses only the standard library -- no `sqlite3` CLI, which is not installed
on every cluster node, and no pandas.

The tables, and why each one is here:

  1. per-kernel aggregate, with the three occupancy limiters (`blockX`,
     `registersPerThread`, `staticSharedMemory`) that `nsys` gives away for
     free.  Reach for this before `ncu`.
  2. one steady-state iteration in dispatch order, so the pipeline is
     legible and the aggregate can be checked against a single iteration.
  3. an early-vs-late duration for each kernel.  `nsys` does not lock
     clocks; if these disagree by more than ~1% the aggregate spans a clock
     change and no ratio taken from it means anything.
  4. kernel time against wall span *within one iteration*, which is where
     dispatch overhead would show up if there were any.  Measured over the
     whole trace it would instead measure the host-side gaps between
     `ctx.synchronize()` and the next launch, which is not the question.
"""

import sqlite3
import sys
from collections import defaultdict

KERNEL_CHARS = 46


def table(rows, headers, aligns=None):
    if not rows:
        return "  (no rows)\n"
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows])))
    widths = [max(len(c) for c in col) for col in cols]
    aligns = aligns or ["<"] + [">"] * (len(headers) - 1)
    out = []
    out.append("  ".join(f"{h:{a}{w}}" for h, a, w in zip(headers, aligns, widths)))
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append("  ".join(f"{str(c):{a}{w}}" for c, a, w in zip(r, aligns, widths)))
    return "\n".join(out) + "\n"


def main(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    ks = con.execute("""
        select k.start, k.end, k.gridX, k.blockX, k.registersPerThread reg,
               k.staticSharedMemory shm, s.value name
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id = k.demangledName
        order by k.start
    """).fetchall()
    if not ks:
        sys.exit(f"{path}: no kernel records -- was the trace exported with --trace=cuda?")

    total_ns = sum(k["end"] - k["start"] for k in ks)
    lo, hi = ks[0]["start"], ks[-1]["start"]

    print(f"=== {path}: {len(ks)} dispatches, {total_ns/1e6:.2f} ms of kernel time ===\n")

    # 1. aggregate, keyed the way an occupancy question needs it
    agg = defaultdict(list)
    for k in ks:
        agg[(k["name"][:KERNEL_CHARS], k["blockX"], k["reg"], k["shm"])].append(
            k["end"] - k["start"])
    rows = []
    for (name, thr, reg, shm), ds in sorted(
            agg.items(), key=lambda kv: -sum(kv[1])):
        rows.append([name, thr, reg, shm, len(ds),
                     round(sum(ds) / len(ds) / 1000, 1),
                     round(min(ds) / 1000, 1),
                     round(100 * sum(ds) / total_ns, 1)])
    print("1. per-kernel aggregate")
    print(table(rows, ["kernel", "thr", "reg", "shm", "n", "us_avg", "us_min", "pct"]))

    # 2. one steady-state iteration.  The iteration boundary is wherever the
    # dispatch sequence repeats; find the period from the name sequence
    # rather than assuming a dispatch count, so this works for v1 and v2 and
    # survives a change to the pipeline.
    names = [k["name"] for k in ks]
    period = None
    for p in range(2, min(60, len(names) // 3 + 1)):
        tail = names[-3 * p:]
        if tail[:p] == tail[p:2 * p] == tail[2 * p:]:
            period = p
            break
    if period:
        it = ks[-period:]
        rows = [[k["name"][:KERNEL_CHARS], k["gridX"], k["blockX"],
                 round((k["end"] - k["start"]) / 1000, 1)] for k in it]
        itotal = sum(k["end"] - k["start"] for k in it)
        rows.append(["TOTAL", "", "", round(itotal / 1000, 1)])
        print(f"2. last complete iteration ({period} dispatches)")
        print(table(rows, ["kernel", "grid", "thr", "us"]))
    else:
        print("2. last complete iteration: no repeating dispatch period found\n")

    # 3. clock-stability control
    rows = []
    for name in sorted({k["name"] for k in ks},
                       key=lambda n: -sum(k["end"] - k["start"]
                                          for k in ks if k["name"] == n))[:10]:
        early = [k["end"] - k["start"] for k in ks
                 if k["name"] == name and k["start"] < lo + (hi - lo) * 0.2]
        late = [k["end"] - k["start"] for k in ks
                if k["name"] == name and k["start"] > lo + (hi - lo) * 0.8]
        if not early or not late:
            continue
        e, l = sum(early) / len(early) / 1000, sum(late) / len(late) / 1000
        rows.append([name[:KERNEL_CHARS], round(e, 1), round(l, 1),
                     f"{100 * (l / e - 1):+.1f}%"])
    print("3. clock-stability control: same kernel, first vs last 20% of trace")
    print("   (disagreement over ~1% means the aggregate spans a clock change)")
    print(table(rows, ["kernel", "early_us", "late_us", "drift"]))

    print("4. launch gaps within one iteration")
    if period:
        it = ks[-period:]
        ik = sum(k["end"] - k["start"] for k in it)
        ispan = it[-1]["end"] - it[0]["start"]
        print(table([[round(ik / 1000, 1), round(ispan / 1000, 1),
                      f"{100 * (1 - ik / ispan):.2f}%"]],
                    ["kernel_us", "span_us", "outside_kernels"]))
    else:
        print("  (needs an iteration period)\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
