"""
Tokenomics for The Front Porch Exchange, computed from runs_log.jsonl.

Run from the repo root:   python submission/tokenomics.py
Prints the tables used in report.html and slides.html (no API calls).

Definitions
- A "pipeline run" is a run_id that reached the final Agent 3 call (a full kit was built).
- An "interview run" is a pipeline run that also contains Agent 1 (Intake) calls.
- Costs are recomputed from token counts with the course prices (Lecture 11, gpt-6-luna,
  USD per 1M tokens): input 0.10, cached 0.01, cache write 0.125, output 0.50;
  billed uncached input = input - cached - cache_write. Output already includes reasoning tokens.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "runs_log.jsonl"
PRICE = {"input": 0.10, "cached": 0.01, "cache_write": 0.125, "output": 0.50}
GATHERINGS_PER_USER_PER_MONTH = 4  # ASSUMPTION (stated in the report)


def cost(c):
    uncached = max(0, c["input_tokens"] - c["cached_tokens"] - c["cache_write_tokens"])
    return (uncached * PRICE["input"] + c["cached_tokens"] * PRICE["cached"]
            + c["cache_write_tokens"] * PRICE["cache_write"] + c["output_tokens"] * PRICE["output"]) / 1e6


def stage(agent):
    if agent.startswith("Agent 1"):
        return "Agent 1 (Intake)"
    if agent.startswith("Agent 2"):
        return "Agent 2 (Blueprint)"
    return "Agent 3 (Connector)"


calls = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
by_run = defaultdict(list)
for c in calls:
    by_run[c["run_id"]].append(c)

pipeline = [r for r, cs in by_run.items() if any(c["agent"] == "Agent 3 · Connector" for c in cs)]
interview = [r for r in pipeline if any(c["agent"].startswith("Agent 1") for c in by_run[r])]

print(f"Calls logged: {len(calls)} | run_ids: {len(by_run)} | pipeline runs: {len(pipeline)} | "
      f"interview runs: {len(interview)} | sample-host runs: {len(pipeline) - len(interview)}")
print(f"Models: {sorted({c['model'] for c in calls})} | caching modes: {sorted({c['caching'] for c in calls})}")
print(f"Max |logged usd - recomputed| = {max(abs((c['usd'] or 0) - cost(c)) for c in calls):.2e}")
print(f"Calls with cached_tokens > 0: {sum(1 for c in calls if c['cached_tokens'] > 0)} | "
      f"calls with cache writes: {sum(1 for c in calls if c['cache_write_tokens'] > 0)}")

# ---- per-agent averages per run ------------------------------------------------
rows = []
for name, runs in (("Agent 1 (Intake)", interview), ("Agent 2 (Blueprint)", pipeline), ("Agent 3 (Connector)", pipeline)):
    per_run = []
    for r in runs:
        cs = [c for c in by_run[r] if stage(c["agent"]) == name]
        per_run.append({
            "calls": len(cs),
            "input": sum(c["input_tokens"] for c in cs),
            "uncached": sum(max(0, c["input_tokens"] - c["cached_tokens"] - c["cache_write_tokens"]) for c in cs),
            "cached": sum(c["cached_tokens"] for c in cs),
            "write": sum(c["cache_write_tokens"] for c in cs),
            "output": sum(c["output_tokens"] for c in cs),
            "usd": sum(cost(c) for c in cs),
        })
    avg = {k: statistics.mean(p[k] for p in per_run) for k in per_run[0]}
    rows.append((name, len(runs), avg))

print("\nPer agent, average per run")
print(f"{'stage':22} {'runs':>4} {'calls':>5} {'input':>7} {'uncached':>8} {'cached':>6} {'c.write':>7} {'output':>7} {'USD':>10}")
for name, n, a in rows:
    print(f"{name:22} {n:>4} {a['calls']:>5.1f} {a['input']:>7.0f} {a['uncached']:>8.0f} {a['cached']:>6.0f} "
          f"{a['write']:>7.0f} {a['output']:>7.0f} {a['usd']:>10.6f}")
full = {k: sum(a[k] for _, _, a in rows) for k in rows[0][2]}
print(f"{'FULL RUN (sum)':22} {'':>4} {full['calls']:>5.1f} {full['input']:>7.0f} {full['uncached']:>8.0f} "
      f"{full['cached']:>6.0f} {full['write']:>7.0f} {full['output']:>7.0f} {full['usd']:>10.6f}")
kit_only = rows[1][2]["usd"] + rows[2][2]["usd"]
print(f"Kit only (Agents 2+3, sample-host path): {kit_only:.6f}")

direct = [sum(cost(c) for c in by_run[r]) for r in interview]
print(f"Cross-check, direct mean of the {len(interview)} interview runs: {statistics.mean(direct):.6f} "
      f"(min {min(direct):.6f}, max {max(direct):.6f})")

# ---- scale -----------------------------------------------------------------------
per_run = full["usd"]
per_user = per_run * GATHERINGS_PER_USER_PER_MONTH
print("\nScale (assumption: 4 gatherings per user per month)")
print(f"cost per run            = {per_run:.6f}")
print(f"per user per month      = {per_run:.6f} x 4 = {per_user:.6f}")
print(f"100 users per month     = {per_user:.6f} x 100 = {per_user * 100:.4f}")
print(f"10,000 users per month  = {per_user:.6f} x 10,000 = {per_user * 10000:.2f}")

# ---- caching what-if (from the same logs) ------------------------------------------
writes = sum(c["cache_write_tokens"] for c in calls)
extra = writes * (PRICE["cache_write"] - PRICE["input"]) / 1e6
print(f"\nCache writes across all calls: {writes} tokens; premium paid vs uncached input: ${extra:.6f}")
print(f"All {len(calls)} logged calls total: ${sum(cost(c) for c in calls):.6f}")

# ---- time ---------------------------------------------------------------------------
from datetime import datetime
kit_secs = {r: sum(c["seconds"] for c in by_run[r] if not c["agent"].startswith("Agent 1")) for r in pipeline}
print("\nKit generation time (API seconds, Agents 2+3) per pipeline run:")
for r in pipeline:
    print(f"  {r}: {kit_secs[r]:.1f}s{'  (interview)' if r in interview else ''}")
vals = list(kit_secs.values())
print(f"  mean {statistics.mean(vals):.1f}s | median {statistics.median(vals):.1f}s | max {max(vals):.1f}s")
print("Interview wall-clock (first Agent 1 call start -> last Agent 3 call end), includes the person typing:")
for r in interview:
    cs = by_run[r]
    t0 = datetime.fromisoformat(cs[0]["ts"]).timestamp() - cs[0]["seconds"]
    t1 = datetime.fromisoformat(cs[-1]["ts"]).timestamp()
    a1 = sum(c["seconds"] for c in cs if c["agent"].startswith("Agent 1"))
    print(f"  {r}: {t1 - t0:.0f}s total, {sum(1 for c in cs if c['agent'].startswith('Agent 1'))} Agent 1 calls, "
          f"{a1:.1f}s of Agent 1 API time, run cost ${sum(cost(c) for c in cs):.6f}")
