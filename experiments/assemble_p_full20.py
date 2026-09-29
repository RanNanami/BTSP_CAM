"""Add the already validated p=.5 full20 trajectory to the p audit table.

The lambda audit's fixed-.01 baseline used the exact same k, p, f_base, RNG,
writer and coordinate-aligned readout, and checked its WB against archived
checkpoints bitwise. Missing event-level fields remain blank, never zero.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run_lambda_state_dependence import read_rows, write_rows

SOURCE = ROOT / "diagnostics" / "lambda_family_full20.csv"
TARGET = ROOT / "diagnostics" / "p_full20.csv"
TASK10 = ROOT / "diagnostics" / "p_equal_budget_task10.csv"


def main() -> None:
    rows = read_rows(TARGET)
    archived = read_rows(SOURCE)
    short = read_rows(TASK10)
    for k in (128, 160):
        candidate = [r for r in rows if int(r["k"]) == k and float(r["p"]) == 1.0]
        if candidate and len(candidate) != 20:
            raise AssertionError(f"k={k} contrast is incomplete")
        if not candidate:
            continue
        base = [r for r in archived if r["family"] == "L_FIXED_01" and int(r["k"]) == k]
        if len(base) != 20:
            raise AssertionError(f"k={k} archived baseline incomplete")
        if k == 128:
            for prior in base[:10]:
                check = next(r for r in short if int(r["task"]) == int(prior["task"]) and
                             float(r["p"]) == 0.5)
                if abs(float(check["cumulative_accuracy"]) -
                       float(prior["cumulative_accuracy"])) > 1e-12:
                    raise AssertionError("archived baseline disagrees with newly run Task1-10")
        rows = [r for r in rows if not (int(r["k"]) == k and float(r["p"]) == 0.5)]
        for record in base:
            row = {
                "status": "DIAGNOSTIC_COUNTERFACTUAL",
                "source": "archived_bitwise_verified_baseline",
                "mode": "equal_budget",
                "k": k, "p": 0.5, "f_base": 0.005,
                "f_base_times_p": 0.0025,
                "task": int(record["task"]),
                "readout": "coordinate_aligned_tau0",
                "plateau_count": record["plateau_count_task"],
                "gross_flips": record["gross_flips_task"],
            }
            for field in ("cumulative_accuracy", "old_accuracy", "current_accuracy",
                          "current_5way_accuracy", "all_a_same_cos", "all_a_diff_cos",
                          "all_a_margin", "old_a_same_cos", "old_a_diff_cos",
                          "old_a_margin", "current_a_same_cos", "current_a_diff_cos",
                          "current_a_margin", "F_mean", "F_std", "N_mean", "N_std",
                          "I_mean", "I_std", "f_eff_mean", "WB_density",
                          "runtime_seconds"):
                row[field] = record[field]
            rows.append(row)
    rows.sort(key=lambda r: (int(r["k"]), float(r["p"]), int(r["task"])))
    for row in rows:
        row.setdefault("source", "new_experimental_run")
    write_rows(TARGET, rows)


if __name__ == "__main__":
    main()
