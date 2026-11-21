"""Run controlled ablations and retain the official runner's per-question evidence."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from hw4 import ABLATION_STEPS, TIME_LIMIT_SECONDS


def parse_stdout(stdout: str) -> dict:
    def grab(pattern: str) -> str:
        m = re.search(pattern, stdout, flags=re.M)
        if not m:
            raise ValueError(f"Missing runner metric: {pattern}")
        return m.group(1)

    metrics = {
        "t1_1_pct": float(grab(r"Task 1\.1 table_to_text:\s*([\d.]+)%")),
        "t1_2_pct": float(grab(r"Task 1\.2 chunk_document:\s*([\d.]+)%")),
        "chunks": int(grab(r"(\d+)\s+chunks to index")),
        "t2_1_pct": float(grab(r"Task 2\.1 valid, non-empty retrieval on answerable questions:\s*([\d.]+)%")),
        "t2_2_pct": float(grab(r"Task 2\.2 parse_query fields right:\s*([\d.]+)%")),
        "t2_3_recall": float(grab(r"Task 2\.3 retrieval recall@5 on answerable questions:\s*([\d.]+)%")),
        "t3_kept": float(grab(r"Task 3\s+answerable questions with chunks:\s*([\d.]+)%")),
        "t3_rej": float(grab(r"unanswerable questions with \[\]:\s*([\d.]+)%")),
        "t4_acc": float(grab(r"Task 4\s+answer accuracy:\s*([\d.]+)%")),
        "pts_1_1": float(grab(r"Task 1\.1\s+([\d.]+)\s*/")),
        "pts_1_2": float(grab(r"Task 1\.2\s+([\d.]+)\s*/")),
        "pts_2_1": float(grab(r"Task 2\.1\s+([\d.]+)\s*/")),
        "pts_2_2": float(grab(r"Task 2\.2\s+([\d.]+)\s*/")),
        "pts_2_3": float(grab(r"Task 2\.3\s+([\d.]+)\s*/")),
        "pts_3": float(grab(r"Task 3\s+([\d.]+)\s*/")),
        "pts_4": float(grab(r"Task 4\s+([\d.]+)\s*/")),
        "total_pts": float(grab(r"Total\s+([\d.]+)\s*/\s*75")),
        "time_s": float(grab(r"Time:\s*([\d.]+)s")),
    }
    for cat in ("comparison", "fact", "table", "unanswerable"):
        m = re.search(rf"^\s*{cat}\s+([\d.]+)%\s+([\d.]+%|-)", stdout, flags=re.M)
        metrics[f"{cat}_acc"] = float(m.group(1)) if m else 0.0
        metrics[f"{cat}_rec"] = m.group(2) if m else "-"

    for key, label in (("your_usage", "Your usage"), ("gen_usage", "Generator usage")):
        m = re.search(rf"{label}:\s*(\{{.*?\}})", stdout)
        if not m:
            raise ValueError(f"Missing runner metric: {label}")
        metrics[key] = ast.literal_eval(m.group(1))
    return metrics


def inspect_csv(csv_path: Path) -> dict:
    wrong_by_cat: dict[str, list[dict]] = {}
    miss_recall: list[str] = []
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if len(rows) != 60 or len({row["id"] for row in rows}) != 60:
            raise ValueError(f"Expected 60 distinct public questions in {csv_path}")
        for row in rows:
            if row["retrieval_hit"] == "False":
                miss_recall.append(row["id"])
            if row["correct"] == "False":
                wrong_by_cat.setdefault(row["category"], []).append({
                    "id": row["id"],
                    "retrieved": row["retrieved"],
                    "hit": row["retrieval_hit"],
                    "answer": row["answer"],
                    "expected": row["expected"],
                })
    return {"miss_recall": miss_recall, "wrong_by_cat": wrong_by_cat}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_path", nargs="?", type=Path, help="optional extra copy of the summary JSON")
    parser.add_argument("--output-dir", type=Path, help="new directory for logs, results and provenance")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1, help="number of independent runs at once")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (args.output_dir or (repo / "ablation_runs" / stamp)).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    variants = [("B0", "Naive retrieval baseline", [])]
    for i, (_, description) in enumerate(ABLATION_STEPS, 1):
        variants.append((f"A{i:02}", description, [key for key, _ in ABLATION_STEPS[:i]]))

    inputs = [repo / "hw4.py", repo / "requirements.txt", Path(__file__).resolve()]
    inputs.extend(p for p in sorted((repo / "public_test").rglob("*")) if p.is_file())
    manifest = {
        "python": sys.version,
        "started_utc": stamp,
        "workers": args.workers,
        "files_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "variants": {name: features for name, _, features in variants},
        "default_row": variants[-1][0],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def run_variant(variant: str, desc: str, features: list[str]) -> dict:
        env = os.environ.copy()
        is_default = variant == manifest["default_row"]
        if is_default:
            env.pop("HW4_VARIANT", None)
        else:
            env["HW4_VARIANT"] = variant
        for attempt in range(1, 4):
            log = run_dir / f"{variant}.attempt{attempt}.log"
            out_csv = log.with_suffix(".csv")
            print(f"Running {variant}: {desc} ({log.name})", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    [sys.executable, "-u", "hw4.py", "--data-folder", "public_test", "--output", str(out_csv)],
                    cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
                    check=True, timeout=TIME_LIMIT_SECONDS,
                )
            output = log.read_text(encoding="utf-8")
            transport_error = re.search(
                r"^Generator failed on .*: (?:OpenAIConnectionError|APITimeoutError|RateLimitError|InternalServerError)\(",
                output, flags=re.M,
            )
            if not transport_error or attempt == 3:
                break
            print(f"  {variant} API transport failure; saved this attempt and retrying", flush=True)
        if any(problem in output for problem in ("Generator failed on", "Traceback (most recent call last)", "WARNING:")):
            raise RuntimeError(f"The runner reported a failure; see {log}")
        metrics = parse_stdout(output)
        results = run_dir / ("results.csv" if is_default else f"{variant}.csv")
        results.write_bytes(out_csv.read_bytes())
        metrics.update(variant=variant, desc=desc, features=features, log=log.name, results=results.name, errors=inspect_csv(results))
        print(f"  {variant} {metrics['total_pts']:.2f}/75 in {metrics['time_s']:.1f}s", flush=True)
        return metrics

    summary = run_dir / "summary.json"
    completed, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_variant, *variant): variant[0] for variant in variants}
        for future in as_completed(pending):
            variant = pending[future]
            try:
                completed[variant] = future.result()
            except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
                failures[variant] = str(error)
            all_results = [completed[name] for name, _, _ in variants if name in completed]
            summary.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    if failures:
        (run_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Incomplete ablation (see {run_dir / 'failures.json'})")

    print("\n=== MARKDOWN ABLATION TABLE ===")
    print("| Variant | Change (single component) | Chunks | T1.1 | T1.2 | T2.2 | Recall@5 | Kept / Rej | T4 Acc | Fact / Table / Comp / Unans Acc | Score (/75) | Calls (Ret+Gen) | Cost ($) | Time (s) |")
    print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: | ---: | ---: |")
    for r in all_results:
        calls = r["your_usage"]["llm_calls"] + r["gen_usage"]["llm_calls"]
        cost = r["your_usage"]["cost_usd"] + r["gen_usage"]["cost_usd"]
        cat_acc = f"{r['fact_acc']:.1f}% / {r['table_acc']:.1f}% / {r['comparison_acc']:.1f}% / {r['unanswerable_acc']:.1f}%"
        kept_rej = f"{r['t3_kept']:.1f}% / {r['t3_rej']:.1f}%"
        print(
            f"| {r['variant']} | {r['desc']} | {r['chunks']} | {r['t1_1_pct']:.1f}% | {r['t1_2_pct']:.1f}% | "
            f"{r['t2_2_pct']:.1f}% | {r['t2_3_recall']:.1f}% | {kept_rej} | {r['t4_acc']:.1f}% | "
            f"{cat_acc} | {r['total_pts']:.2f} | {calls} | ${cost:.4f} | {r['time_s']:.1f} |"
        )

    if args.summary_path:
        args.summary_path.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
