#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_seconds(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def fmt_score(value):
    return "n/a" if value is None else f"{value:.3f}"


def truncate_text(value, limit=120):
    if value is None:
        return "n/a"
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def percentile(values, ratio):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * ratio))
    return ordered[index]


def summarize_smoke(output_dir: Path):
    files = sorted((output_dir / "smoke").glob("*.json"))
    items = []
    for path in files:
        payload = load_json(path)
        comparison = payload.get("comparison", {})
        score_name = comparison.get("score_name")
        synth = payload.get("synth", {})
        items.append(
            {
                "name": path.stem,
                "endpoint": payload.get("endpoint"),
                "score_name": score_name,
                "score": comparison.get(score_name) if score_name else None,
                "total_seconds": synth.get("total_seconds"),
                "first_chunk_seconds": synth.get("first_chunk_seconds"),
                "status_code": payload.get("status_code") or synth.get("status_code"),
                "expected": comparison.get("expected"),
                "actual": comparison.get("actual"),
            }
        )
    return items


def summarize_longform(output_dir: Path):
    files = sorted((output_dir / "longform").glob("*.json"))
    items = []
    for path in files:
        payload = load_json(path)
        comparison = payload.get("comparison", {})
        score_name = comparison.get("score_name")
        items.append(
            {
                "name": path.stem,
                "score_name": score_name,
                "score": comparison.get(score_name) if score_name else None,
                "total_seconds": payload.get("synth", {}).get("total_seconds"),
                "first_chunk_seconds": payload.get("synth", {}).get("first_chunk_seconds"),
                "expected": comparison.get("expected"),
                "actual": comparison.get("actual"),
            }
        )
    return items


def summarize_concurrency(output_dir: Path):
    files = sorted((output_dir / "concurrency").glob("concurrency_*.json"))
    rows = []
    for path in files:
        payload = load_json(path)
        summary = payload.get("summary", {})
        results = payload.get("results", [])
        failures = payload.get("failures", [])
        cer_values = [item.get("comparison", {}).get("cer", 1.0) for item in results]
        first_chunk_values = [
            item.get("synth", {}).get("first_chunk_seconds") or item.get("synth", {}).get("total_seconds", 0.0)
            for item in results
        ]
        total_values = [item.get("synth", {}).get("total_seconds", 0.0) for item in results]
        rows.append(
            {
                "level": summary.get("concurrency_level"),
                "success_count": summary.get("success_count", 0),
                "total_requests": summary.get("total_requests", 0),
                "failure_count": summary.get("failure_count", len(failures)),
                "avg_first_chunk_seconds": summary.get("avg_first_chunk_seconds"),
                "p95_first_chunk_seconds": percentile(first_chunk_values, 0.95),
                "avg_total_seconds": summary.get("avg_total_seconds"),
                "p95_total_seconds": percentile(total_values, 0.95),
                "avg_cer": summary.get("avg_cer"),
                "max_cer": summary.get("max_cer"),
                "failure_examples": [item.get("error") for item in failures[:2]],
            }
        )
    rows.sort(key=lambda row: (row["level"] is None, row["level"]))
    return rows


def recommend_concurrency(rows):
    stable = None
    degraded = None
    unsafe = rows[-1]["level"] if rows else None
    for row in rows:
        success_rate = 0.0
        if row["total_requests"]:
            success_rate = row["success_count"] / row["total_requests"]
        if success_rate >= 0.95 and row["failure_count"] == 0:
            stable = row["level"]
        elif degraded is None:
            degraded = row["level"]
    return stable, degraded, unsafe


def read_tegrastats(output_dir: Path):
    log_path = output_dir / "tegrastats.log"
    if not log_path.exists():
        return []
    return [line.strip() for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def build_markdown_report(output_dir: Path):
    smoke = summarize_smoke(output_dir)
    longform = summarize_longform(output_dir)
    concurrency = summarize_concurrency(output_dir)
    stable, degraded, unsafe = recommend_concurrency(concurrency)
    tegra_lines = read_tegrastats(output_dir)

    lines = [
        "# GPT-SoVITS Jetson Test Report",
        "",
        "## Summary",
        f"- Smoke cases: {len(smoke)}",
        f"- Longform cases: {len(longform)}",
        f"- Concurrency levels tested: {', '.join(str(row['level']) for row in concurrency) or 'none'}",
        f"- Recommended stable concurrency: {stable if stable is not None else 'not reached'}",
        f"- First degraded concurrency: {degraded if degraded is not None else 'not observed'}",
        f"- Highest tested concurrency: {unsafe if unsafe is not None else 'n/a'}",
        "",
        "## Smoke",
    ]

    if not smoke:
        lines.append("- No smoke results found.")
    else:
        for item in smoke:
            if item["score_name"]:
                lines.append(
                    f"- `{item['name']}`: {item['score_name']}={fmt_score(item['score'])}, "
                    f"total={fmt_seconds(item['total_seconds'])}, first_chunk={fmt_seconds(item['first_chunk_seconds'])}"
                )
                lines.append(f"- expected: `{truncate_text(item['expected'])}`")
                lines.append(f"- asr: `{truncate_text(item['actual'])}`")
            else:
                lines.append(
                    f"- `{item['name']}`: endpoint={item['endpoint'] or 'n/a'}, status={item['status_code'] or 'n/a'}"
                )

    lines.extend(["", "## Longform"])
    if not longform:
        lines.append("- No longform results found.")
    else:
        for item in longform:
            lines.append(
                f"- `{item['name']}`: {item['score_name']}={fmt_score(item['score'])}, "
                f"total={fmt_seconds(item['total_seconds'])}, first_chunk={fmt_seconds(item['first_chunk_seconds'])}"
            )
            lines.append(f"- expected: `{truncate_text(item['expected'])}`")
            lines.append(f"- asr: `{truncate_text(item['actual'])}`")

    lines.extend(["", "## Concurrency"])
    if not concurrency:
        lines.append("- No concurrency benchmark results found.")
    else:
        for row in concurrency:
            lines.append(
                f"- `conc={row['level']}`: success={row['success_count']}/{row['total_requests']}, "
                f"failures={row['failure_count']}, "
                f"avg_first_chunk={fmt_seconds(row['avg_first_chunk_seconds'])}, "
                f"p95_first_chunk={fmt_seconds(row['p95_first_chunk_seconds'])}, "
                f"avg_total={fmt_seconds(row['avg_total_seconds'])}, "
                f"max_cer={fmt_score(row['max_cer'])}"
            )
            for error in row["failure_examples"]:
                lines.append(f"- error: `{truncate_text(error)}`")

    lines.extend(["", "## Resource Notes"])
    if tegra_lines:
        lines.append("- `tegrastats` captured during test run.")
        for index, line in enumerate(tegra_lines[:5], start=1):
            lines.append(f"- sample {index}: `{line}`")
    else:
        lines.append("- No tegrastats log found.")

    return "\n".join(lines) + "\n"


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test-results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_markdown_report(output_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
