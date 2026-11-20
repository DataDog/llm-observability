#!/usr/bin/env python3
"""
trace_scenarios.py — drive the stock-watchlist-agent through a set of scenarios
to produce a diverse corpus of LLM Observability traces.

Each scenario runs ``python -m src.main <tickers>`` in a fresh subprocess (the
real app entrypoint, so each run flushes its own trace on exit). Every run from a
single invocation shares a ``harness_run:<id>`` tag plus a per-scenario
``scenario:<name>`` tag, so you can filter/group the traces in LLM Obs.

Prereqs (real, paid OpenAI calls; ~30s–2min each):
    Set OPENAI_API_KEY, and DD_API_KEY/DD_SITE to emit traces, in .env
    (see .env.example) — loaded automatically via sitecustomize.py.

Usage:
    .venv/bin/python trace_scenarios.py                      # run all scenarios
    .venv/bin/python trace_scenarios.py --list               # list, don't run
    .venv/bin/python trace_scenarios.py --dry-run            # print commands, don't run
    .venv/bin/python trace_scenarios.py --only minimal etfs  # run a subset
    .venv/bin/python trace_scenarios.py --delay 5            # pause between runs
    .venv/bin/python trace_scenarios.py --version prompt-v2  # tag DD_VERSION on every run
    .venv/bin/python trace_scenarios.py --ml-app stock-watchlist-agent-dev
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCENARIOS = [
    {"name": "minimal", "tickers": ["NVDA"], "desc": "single ticker, smallest trace"},
    {"name": "same-sector", "tickers": ["NVDA", "AMD"], "desc": "one themed research batch"},
    {"name": "cross-sector", "tickers": ["AAPL", "JPM", "XOM"], "desc": "multiple themed batches"},
    {
        "name": "wide-watchlist",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        "desc": "parallel batches, deep trace",
    },
    {"name": "lowercase-input", "tickers": ["tsla"], "desc": "cli uppercasing path"},
    {"name": "etfs", "tickers": ["SPY", "QQQ"], "desc": "non-equity tickers"},
    {"name": "known-plus-bogus", "tickers": ["NVDA", "ZZZZ"], "desc": "partial / uncertain / error handling"},
    {"name": "mega-cap", "tickers": ["AAPL"], "desc": "lots of news, long tool-output spans"},
]


def _venv_python(script_dir: Path) -> str:
    cand = script_dir / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def _preflight(dry_run: bool) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set — the agent cannot run.")
        if not dry_run:
            sys.exit(1)
    if not os.environ.get("DD_API_KEY"):
        print("⚠️  DD_API_KEY not set — the app will run but emit NO traces to LLM Obs.")


def _build_env(scenario: dict, run_id: str, script_dir: Path, args: argparse.Namespace) -> dict:
    env = dict(os.environ)
    env.update(scenario.get("env", {}))

    # Tags so every trace is filterable/groupable in LLM Obs.
    tags = [f"harness_run:{run_id}", f"scenario:{scenario['name']}"]
    existing = env.get("DD_TAGS", "")
    env["DD_TAGS"] = ",".join(t for t in [existing, *tags] if t)

    if args.version:
        env["DD_VERSION"] = args.version
    if args.ml_app:
        env["DD_LLMOBS_ML_APP"] = args.ml_app

    # Ensure the local sitecustomize.py workaround loads even if the fixed ddtrace
    # isn't installed yet (harmless once it is). See sitecustomize.py.
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(script_dir), env.get("PYTHONPATH", "")] if p)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    parser.add_argument("--only", nargs="*", metavar="NAME", help="run only these scenario names")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to pause between runs")
    parser.add_argument("--version", help="set DD_VERSION on every run (for run-over-run comparison)")
    parser.add_argument("--ml-app", help="override DD_LLMOBS_ML_APP")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.list:
        for s in SCENARIOS:
            print(f"  {s['name']:<18} {' '.join(s['tickers']):<42} {s['desc']}")
        return

    scenarios = SCENARIOS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s["name"] for s in SCENARIOS}
        if unknown:
            print(f"unknown scenario(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"available: {', '.join(s['name'] for s in SCENARIOS)}", file=sys.stderr)
            sys.exit(2)
        scenarios = [s for s in SCENARIOS if s["name"] in wanted]

    _preflight(args.dry_run)

    py = _venv_python(script_dir)
    run_id = f"harness-{int(time.time())}"
    print(f"harness run id : {run_id}")
    print(f"python         : {py}")
    print(f"scenarios      : {len(scenarios)}\n")

    results: list[tuple[str, bool, float, int]] = []
    try:
        for idx, scenario in enumerate(scenarios, 1):
            cmd = [py, "-m", "src.main", *scenario["tickers"]]
            env = _build_env(scenario, run_id, script_dir, args)
            label = f"[{idx}/{len(scenarios)}] {scenario['name']} ({' '.join(scenario['tickers'])})"

            if args.dry_run:
                print(f"DRY-RUN {label}")
                print(f"        DD_TAGS={env['DD_TAGS']}")
                print(f"        {' '.join(cmd)}")
                continue

            print(f"▶  {label} … ", end="", flush=True)
            start = time.time()
            proc = subprocess.run(
                cmd, cwd=str(script_dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            dur = time.time() - start
            ok = proc.returncode == 0
            print(f"{'✅' if ok else '❌'}  ({dur:.0f}s, exit {proc.returncode})")
            if not ok:
                tail = "\n".join((proc.stdout or "").strip().splitlines()[-20:])
                print(f"   --- output tail ---\n{tail}\n   -------------------")
            results.append((scenario["name"], ok, dur, proc.returncode))

            if args.delay and idx < len(scenarios):
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\ninterrupted — printing partial summary…")

    if not args.dry_run and results:
        passed = sum(1 for _, ok, _, _ in results if ok)
        print("\n=== summary ===")
        for name, ok, dur, rc in results:
            print(f"  {'✅' if ok else '❌'} {name:<18} {dur:6.0f}s  exit={rc}")
        print(f"\n  {passed}/{len(results)} succeeded   |   filter traces by tag:  harness_run:{run_id}")
        sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
