"""Mechanical CSV correction for the gamma=0 pseudoinverse edge case."""

import csv
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "diagnostics/gamma_fixed_task10.csv"


def main() -> None:
    with PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    zero_rows = [row for row in rows if float(row["gamma"]) == 0.0]
    if len(zero_rows) != 10:
        raise ValueError("expected one zero-gamma row per Task1–10")
    for row in zero_rows:
        rank = int(row["solve_status"].removeprefix("spectral_pinv_rank_"))
        eigen_min = float(row["min_eigenvalue_after_regularization"])
        eigen_max = float(row["lambda_max_A"])
        row["condition_A_plus_gamma_1norm_est"] = "nan"  # no Cholesky estimate
        row["condition_A_plus_gamma_spectral"] = (
            "inf" if rank < 4096 else str(eigen_max / eigen_min)
        )
    temp = PATH.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(PATH)


if __name__ == "__main__":
    main()
