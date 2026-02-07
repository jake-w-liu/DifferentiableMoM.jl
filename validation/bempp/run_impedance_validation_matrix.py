#!/usr/bin/env python3
"""Run and summarize a multi-case impedance-loaded external validation matrix.

This script orchestrates:
1) Julia reference generation for each case
2) Bempp impedance-loaded solve for each case
3) Bempp-vs-Julia comparison for each case
4) Matrix-level summary with acceptance-oriented gates
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    freq_ghz: float
    zs_imag_ohm: float
    theta_inc_deg: float
    phi_inc_deg: float


CASES: List[ValidationCase] = [
    ValidationCase("case01_z50_n0_f3p00", 3.00, 50.0, 0.0, 0.0),
    ValidationCase("case02_z100_n0_f3p00", 3.00, 100.0, 0.0, 0.0),
    ValidationCase("case03_z200_n0_f3p00", 3.00, 200.0, 0.0, 0.0),
    ValidationCase("case04_z300_n0_f3p00", 3.00, 300.0, 0.0, 0.0),
    ValidationCase("case05_z200_n5_f3p00", 3.00, 200.0, 5.0, 0.0),
    ValidationCase("case06_z200_n0_f3p06", 3.06, 200.0, 0.0, 0.0),
]


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    path = env.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if venv_bin not in entries:
        env["PATH"] = venv_bin + os.pathsep + path
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_case_pass_flags(metrics: Dict) -> Dict[str, bool]:
    rmse = float(metrics["global"]["rmse_db"])
    max_abs = float(metrics["global"]["max_abs_diff_db"])
    near_target = float(metrics["near_target"]["mean_abs_diff_db"])
    return {
        "pass_rmse_le_2p5": rmse <= 2.5,
        "pass_target_le_1p5": near_target <= 1.5,
        "pass_max_le_25": max_abs <= 25.0,
    }


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    path: Path,
    rows: List[Dict[str, object]],
    gates: Dict[str, object],
    config: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("# Impedance Validation Matrix Summary")
    lines.append("")
    lines.append("## Bempp Convention Configuration")
    lines.append(f"- op-sign: `{config['bempp_op_sign']}`")
    lines.append(f"- rhs-cross: `{config['bempp_rhs_cross']}`")
    lines.append(f"- rhs-sign: `{config['bempp_rhs_sign']}`")
    lines.append(f"- phase-sign: `{config['bempp_phase_sign']}`")
    lines.append(f"- zs-scale: `{config['bempp_zs_scale']}`")
    lines.append("")
    lines.append("## Case Results")
    lines.append(
        "| Case | f (GHz) | Zs imag (ohm) | theta_inc (deg) | RMSE (dB) | Mean |ΔD| near target (dB) | Max |ΔD| (dB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['freq_ghz']:.2f} | {row['zs_imag_ohm']:.1f} | "
            f"{row['theta_inc_deg']:.1f} | {row['rmse_db']:.3f} | "
            f"{row['near_target_mean_abs_db']:.3f} | {row['max_abs_db']:.3f} |"
        )
    lines.append("")
    lines.append("## Acceptance Gates")
    lines.append(
        f"- Cases with near-target mean |ΔD| <= 1.5 dB: {gates['count_target_le_1p5']}/{gates['num_cases']}"
    )
    lines.append(
        f"- Cases with global RMSE <= 2.5 dB: {gates['count_rmse_le_2p5']}/{gates['num_cases']}"
    )
    lines.append(
        f"- Cases with max |ΔD| <= 25 dB: {gates['count_max_le_25']}/{gates['num_cases']}"
    )
    lines.append("")
    lines.append(
        f"- Matrix gate status (>=4/6 for target and RMSE, all <=25 dB max): "
        f"{'PASS' if gates['matrix_gate_pass'] else 'FAIL'}"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/ and Project.toml",
    )
    parser.add_argument("--n-theta", type=int, default=180)
    parser.add_argument("--n-phi", type=int, default=72)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-julia", action="store_true")
    parser.add_argument("--skip-bempp", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--bempp-op-sign", choices=["minus", "plus"], default="minus")
    parser.add_argument("--bempp-rhs-cross", choices=["e_cross_n", "n_cross_e"], default="e_cross_n")
    parser.add_argument("--bempp-rhs-sign", type=float, default=1.0)
    parser.add_argument("--bempp-phase-sign", choices=["plus", "minus"], default="plus")
    parser.add_argument("--bempp-zs-scale", type=float, default=1.0)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for case in CASES:
        prefix = case.case_id
        print(f"\n=== {case.case_id} ===")

        if not args.skip_julia:
            run_cmd(
                [
                    "julia",
                    "--project=.",
                    "validation/bempp/run_impedance_case_julia_reference.jl",
                    "--freq-ghz",
                    str(case.freq_ghz),
                    "--theta-ohm",
                    str(case.zs_imag_ohm),
                    "--theta-inc-deg",
                    str(case.theta_inc_deg),
                    "--phi-inc-deg",
                    str(case.phi_inc_deg),
                    "--n-theta",
                    str(args.n_theta),
                    "--n-phi",
                    str(args.n_phi),
                    "--output-prefix",
                    prefix,
                ],
                cwd=project_root,
                dry_run=args.dry_run,
            )

        if not args.skip_bempp:
            run_cmd(
                [
                    sys.executable,
                    "validation/bempp/run_impedance_cross_validation.py",
                    "--freq-ghz",
                    str(case.freq_ghz),
                    "--zs-imag-ohm",
                    str(case.zs_imag_ohm),
                    "--theta-inc-deg",
                    str(case.theta_inc_deg),
                    "--phi-inc-deg",
                    str(case.phi_inc_deg),
                    "--n-theta",
                    str(args.n_theta),
                    "--n-phi",
                    str(args.n_phi),
                    "--op-sign",
                    args.bempp_op_sign,
                    "--rhs-cross",
                    args.bempp_rhs_cross,
                    "--rhs-sign",
                    str(args.bempp_rhs_sign),
                    "--phase-sign",
                    args.bempp_phase_sign,
                    "--zs-scale",
                    str(args.bempp_zs_scale),
                    "--output-prefix",
                    prefix,
                ],
                cwd=project_root,
                dry_run=args.dry_run,
            )

        if not args.skip_compare:
            run_cmd(
                [
                    sys.executable,
                    "validation/bempp/compare_impedance_to_julia.py",
                    "--output-prefix",
                    prefix,
                    "--target-theta-deg",
                    "30.0",
                ],
                cwd=project_root,
                dry_run=args.dry_run,
            )

        if args.dry_run:
            continue

        report_json = data_dir / f"bempp_{prefix}_cross_validation_report.json"
        if not report_json.exists():
            print(f"WARNING: missing report {report_json}, skipping summary row")
            continue

        metrics = load_json(report_json)
        flags = compute_case_pass_flags(metrics)
        row: Dict[str, object] = {
            "case_id": case.case_id,
            "freq_ghz": case.freq_ghz,
            "zs_imag_ohm": case.zs_imag_ohm,
            "theta_inc_deg": case.theta_inc_deg,
            "phi_inc_deg": case.phi_inc_deg,
            "rmse_db": float(metrics["global"]["rmse_db"]),
            "mean_abs_db": float(metrics["global"]["mean_abs_diff_db"]),
            "max_abs_db": float(metrics["global"]["max_abs_diff_db"]),
            "near_target_theta_deg": float(metrics["near_target"]["nearest_theta_deg"]),
            "near_target_mean_abs_db": float(metrics["near_target"]["mean_abs_diff_db"]),
            "near_target_max_abs_db": float(metrics["near_target"]["max_abs_diff_db"]),
            **flags,
        }
        summary_rows.append(row)

    if args.dry_run:
        print("\nDry run complete.")
        return

    summary_csv = data_dir / "impedance_validation_matrix_summary.csv"
    summary_md = data_dir / "impedance_validation_matrix_summary.md"
    summary_json = data_dir / "impedance_validation_matrix_summary.json"

    write_summary_csv(summary_csv, summary_rows)

    gates = {
        "num_cases": len(summary_rows),
        "count_target_le_1p5": sum(bool(r["pass_target_le_1p5"]) for r in summary_rows),
        "count_rmse_le_2p5": sum(bool(r["pass_rmse_le_2p5"]) for r in summary_rows),
        "count_max_le_25": sum(bool(r["pass_max_le_25"]) for r in summary_rows),
    }
    gates["matrix_gate_pass"] = (
        gates["count_target_le_1p5"] >= 4
        and gates["count_rmse_le_2p5"] >= 4
        and gates["count_max_le_25"] == gates["num_cases"]
    )

    config = {
        "bempp_op_sign": args.bempp_op_sign,
        "bempp_rhs_cross": args.bempp_rhs_cross,
        "bempp_rhs_sign": args.bempp_rhs_sign,
        "bempp_phase_sign": args.bempp_phase_sign,
        "bempp_zs_scale": args.bempp_zs_scale,
    }
    write_summary_md(summary_md, summary_rows, gates, config)
    summary_json.write_text(
        json.dumps({"config": config, "cases": summary_rows, "gates": gates}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved {summary_csv}")
    print(f"Saved {summary_md}")
    print(f"Saved {summary_json}")
    print(
        f"Matrix gate status: {'PASS' if gates['matrix_gate_pass'] else 'FAIL'} "
        f"(target<=1.5: {gates['count_target_le_1p5']}/{gates['num_cases']}, "
        f"rmse<=2.5: {gates['count_rmse_le_2p5']}/{gates['num_cases']})"
    )


if __name__ == "__main__":
    main()
