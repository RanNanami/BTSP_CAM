"""Five-task representation/readout ablation for BTSP-CAM localization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml
from scipy.linalg import cho_solve

from btsp_cam.config import BTSPCAMConfig
from btsp_cam.model import BTSPCAM
from btsp_cam.protocol import pilot_class_order
from btsp_cam.ridge import StreamingRidge
from btsp_cam.rng import RNGProvenance
from btsp_cam.sparse_encoder import RandomSparseEncoder


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "btsp_cam_cifar100.yaml"
CACHE_DIR = ROOT / "data" / "feature_cache_btsp_cam"
OUTPUT_DIR = ROOT / "diagnostics"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cache(split: str, task: int) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE_DIR / f"{split}_task_{task:02d}.npz"
    with np.load(path, allow_pickle=False) as cache:
        features = cache["features"].copy()
        labels = cache["labels"].copy()
    if not np.all(np.isfinite(features)):
        raise ValueError(f"non-finite cached features: {path}")
    return features, labels


def update_sparse_ridge(
    ridge: StreamingRidge, codes: np.ndarray, labels: np.ndarray
) -> None:
    """Exact sparse form of A += S.T@S and B += S.T@Y."""

    labels = np.asarray(labels, dtype=np.int64)
    ridge.expand_classes(int(labels.max()) + 1)
    for code, label in zip(codes, labels):
        active = np.flatnonzero(code)
        ridge.a[np.ix_(active, active)] += 1.0
        ridge.b[active, int(label)] += 1.0


def representation_geometry(features: np.ndarray, labels: np.ndarray) -> dict:
    """Exact mean pairwise cosines without constructing an N x N matrix."""

    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    classes = np.unique(labels)
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    same_numerator = 0.0
    same_denominator = 0
    prototypes = []
    for label in classes:
        class_rows = normalized[labels == label]
        class_sum = class_rows.sum(axis=0)
        sums[int(label)] = class_sum
        counts[int(label)] = class_rows.shape[0]
        same_numerator += float(class_sum @ class_sum - np.einsum("ij,ij->", class_rows, class_rows))
        same_denominator += class_rows.shape[0] * (class_rows.shape[0] - 1)
        prototype = x[labels == label].mean(axis=0)
        prototype_norm = np.linalg.norm(prototype)
        prototypes.append(prototype / prototype_norm if prototype_norm > 0 else prototype)

    different_numerator = 0.0
    different_denominator = 0
    for left_index, left in enumerate(classes):
        for right in classes[left_index + 1 :]:
            different_numerator += float(sums[int(left)] @ sums[int(right)])
            different_denominator += counts[int(left)] * counts[int(right)]
    prototype_matrix = np.asarray(prototypes)
    prototype_values = []
    for left in range(len(classes)):
        for right in range(left + 1, len(classes)):
            prototype_values.append(float(prototype_matrix[left] @ prototype_matrix[right]))
    prototype_values = np.asarray(prototype_values, dtype=np.float64)
    return {
        "same_class_mean_cosine": same_numerator / same_denominator,
        "different_class_mean_cosine": different_numerator / different_denominator,
        "cosine_margin": same_numerator / same_denominator
        - different_numerator / different_denominator,
        "class_prototype_pairwise_cosine_mean": float(prototype_values.mean()),
        "class_prototype_pairwise_cosine_std": float(prototype_values.std()),
        "class_prototype_pairwise_cosine_min": float(prototype_values.min()),
        "class_prototype_pairwise_cosine_max": float(prototype_values.max()),
        "zero_row_fraction": float(np.mean(norms[:, 0] == 0)),
    }


def power_eigenvalue(matrix: np.ndarray, seed: int, iterations: int = 12) -> float:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(matrix.shape[0])
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        vector = matrix @ vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0.0
        vector /= norm
    return float(vector @ (matrix @ vector))


def inverse_min_eigenvalue(ridge: StreamingRidge, seed: int, iterations: int = 8) -> float:
    factor = ridge._last_cholesky
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(ridge.feature_dim)
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        vector = cho_solve(factor, vector, check_finite=False)
        vector /= np.linalg.norm(vector)
    regularized_value = float(vector @ (ridge.a @ vector) + ridge.gamma)
    return max(0.0, regularized_value - ridge.gamma)


def spectrum_summary(
    ridge: StreamingRidge, *, sample_count: int, seed: int
) -> dict:
    matrix = ridge.a
    dimension = ridge.feature_dim
    trace = float(np.trace(matrix))
    frobenius_squared = float(np.einsum("ij,ij->", matrix, matrix, optimize=True))
    effective_rank = trace * trace / frobenius_squared if frobenius_squared else 0.0
    active_dimensions = int(np.count_nonzero(np.diag(matrix) > 0))
    rank_upper = min(sample_count, active_dimensions, dimension)
    if dimension <= 1024:
        eigenvalues = np.linalg.eigvalsh(matrix)
        eig_max = float(eigenvalues[-1])
        tolerance = max(1.0, eig_max) * dimension * np.finfo(np.float64).eps
        rank_value = int(np.count_nonzero(eigenvalues > tolerance))
        eig_min = float(max(0.0, eigenvalues[0]))
        rank_is_lower_bound = False
        rank_lower = rank_value
    else:
        eig_max = power_eigenvalue(matrix, seed)
        # A zero diagonal in the PSD Gram matrix A implies an all-zero row and
        # column, and sample_count < dimension also proves rank deficiency.
        # In either case the true minimum eigenvalue is exactly zero.  Only use
        # inverse iteration when the simple rank bound does not settle it.
        eig_min = (
            0.0
            if rank_upper < dimension
            else inverse_min_eigenvalue(ridge, seed + 1)
        )
        subset_size = min(512, dimension)
        selected = np.linspace(0, dimension - 1, subset_size, dtype=np.int64)
        sub_eigenvalues = np.linalg.eigvalsh(matrix[np.ix_(selected, selected)])
        tolerance = max(1.0, float(sub_eigenvalues[-1])) * subset_size * np.finfo(np.float64).eps
        rank_lower = int(np.count_nonzero(sub_eigenvalues > tolerance))
        rank_value = rank_lower
        rank_is_lower_bound = rank_lower != rank_upper
    return {
        "eig_min": eig_min,
        "eig_max": eig_max,
        "condition_number": ridge.condition_number,
        "feature_rank": rank_value,
        "feature_rank_lower_bound": rank_lower,
        "feature_rank_upper_bound": rank_upper,
        "feature_rank_is_lower_bound": rank_is_lower_bound,
        "effective_rank": effective_rank,
        "effective_rank_definition": "participation_ratio_trace_squared_over_frobenius_squared",
        "active_dimensions": active_dimensions,
        "eigenvalue_matrix": "unregularized_A",
        "condition_matrix": "A_plus_gamma_I",
        "condition_norm": "one_norm_lapack_estimate",
        "spectrum_method": "exact_eigvalsh" if dimension <= 1024 else "power_inverse_power_and_512_principal_rank_bound",
    }


def accuracy_rows(
    *,
    task: int,
    method: str,
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    condition: float,
) -> dict:
    logits = np.asarray(features, dtype=np.float64) @ weights
    predictions = logits.argmax(axis=1)
    new_start = (task - 1) * 5
    new_stop = task * 5
    new_mask = (labels >= new_start) & (labels < new_stop)
    old_mask = labels < new_start
    current_columns = np.arange(new_start, new_stop)
    current_predictions = current_columns[logits[new_mask][:, current_columns].argmax(axis=1)]
    return {
        "task": task,
        "method": method,
        "current_acc": float(np.mean(predictions[new_mask] == labels[new_mask])),
        "cumulative_acc": float(np.mean(predictions == labels)),
        "old_acc": float(np.mean(predictions[old_mask] == labels[old_mask])) if np.any(old_mask) else float("nan"),
        "new_acc": float(np.mean(predictions[new_mask] == labels[new_mask])),
        "current_task_5way_acc": float(np.mean(current_predictions == labels[new_mask])),
        "ridge_condition": condition,
        "feature_dim": int(features.shape[1]),
        "feature_nonzero_density": float(np.count_nonzero(features) / features.size),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = BTSPCAMConfig.from_mapping(raw_config)
    provenance = RNGProvenance.from_seed(config.seed)
    class_order = pilot_class_order(config.num_classes, config.seed)

    ridge_h = StreamingRidge(config.feature_dim, 0, config.gamma)
    ridge_s = StreamingRidge(config.code_dim, 0, config.gamma)
    sparse_encoder = RandomSparseEncoder(
        config.feature_dim, config.code_dim, config.k, provenance.projection_seed
    )
    btsp_model = BTSPCAM(
        config,
        class_order=class_order,
        rng_provenance=provenance,
    )
    np.testing.assert_array_equal(
        sparse_encoder.projection, btsp_model.encoder.projection
    )
    np.testing.assert_array_equal(sparse_encoder.bias, btsp_model.encoder.bias)

    ablation_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    geometry_rows: list[dict] = []
    sparse_rows: list[dict] = []
    dynamics_rows: list[dict] = []
    early_rows: list[dict] = []
    drop_rows: list[dict] = []
    test_h_parts: list[np.ndarray] = []
    test_y_parts: list[np.ndarray] = []
    cumulative_train_samples = 0
    started = time.perf_counter()

    for task in range(1, 6):
        task_started = time.perf_counter()
        train_h, train_y = load_cache("train", task)
        test_h, test_y = load_cache("test", task)
        cumulative_train_samples += train_y.size
        test_h_parts.append(test_h)
        test_y_parts.append(test_y)
        cumulative_h = np.concatenate(test_h_parts)
        cumulative_y = np.concatenate(test_y_parts)

        ridge_h.expand_classes(int(train_y.max()) + 1)
        ridge_h.update(train_h, train_y)
        ridge_h.solve()
        sparse_train = sparse_encoder.encode(train_h, config.encoder_batch_size)
        update_sparse_ridge(ridge_s, sparse_train, train_y)
        ridge_s.solve()
        logs = btsp_model.learn_task(train_h, train_y, task_id=task)

        sparse_test = sparse_encoder.encode(cumulative_h, config.encoder_batch_size)
        ca1_test = btsp_model.ca1(cumulative_h)
        ca1_train = btsp_model.ca1(train_h)
        representations = {
            "vit_ridge": (cumulative_h, ridge_h),
            "sparse_ridge": (sparse_test, ridge_s),
            "btsp_cam": (ca1_test, btsp_model.ridge),
        }
        task_rows = {}
        for method, (features, ridge) in representations.items():
            row = accuracy_rows(
                task=task,
                method=method,
                features=features,
                labels=cumulative_y,
                weights=ridge.weights,
                condition=ridge.condition_number,
            )
            ablation_rows.append(row)
            task_rows[method] = row
            spectrum_rows.append(
                {
                    "task": task,
                    "method": method,
                    **spectrum_summary(
                        ridge,
                        sample_count=cumulative_train_samples,
                        seed=config.seed + 100 * task + len(spectrum_rows),
                    ),
                }
            )

        if task == 1 and task_rows["vit_ridge"]["cumulative_acc"] < 0.4:
            raise RuntimeError("Baseline A failed sanity threshold; stop B/C")
        drop_rows.extend(
            [
                {
                    "task": task,
                    "transition": "A_vit_to_B_sparse",
                    "current_acc_drop": task_rows["vit_ridge"]["current_acc"]
                    - task_rows["sparse_ridge"]["current_acc"],
                    "cumulative_acc_drop": task_rows["vit_ridge"]["cumulative_acc"]
                    - task_rows["sparse_ridge"]["cumulative_acc"],
                },
                {
                    "task": task,
                    "transition": "B_sparse_to_C_btsp",
                    "current_acc_drop": task_rows["sparse_ridge"]["current_acc"]
                    - task_rows["btsp_cam"]["current_acc"],
                    "cumulative_acc_drop": task_rows["sparse_ridge"]["cumulative_acc"]
                    - task_rows["btsp_cam"]["cumulative_acc"],
                },
            ]
        )

        for name, representation in (
            ("h", train_h),
            ("s", sparse_train),
            ("z", ca1_train),
        ):
            geometry_rows.append(
                {
                    "task": task,
                    "representation": name,
                    "feature_dim": representation.shape[1],
                    "nonzero_density": float(np.count_nonzero(representation) / representation.size),
                    **representation_geometry(representation, train_y),
                }
            )

        packed = np.packbits(sparse_train, axis=1)
        unique_codes = np.unique(packed, axis=0).shape[0]
        activation_frequency = sparse_train.mean(axis=0)
        sparse_geometry = geometry_rows[-2]
        sparse_rows.append(
            {
                "task": task,
                "within_class_cosine": sparse_geometry["same_class_mean_cosine"],
                "between_class_cosine": sparse_geometry["different_class_mean_cosine"],
                "unique_codes": unique_codes,
                "sample_count": sparse_train.shape[0],
                "collision_rate": 1.0 - unique_codes / sparse_train.shape[0],
                "activation_frequency_mean": float(activation_frequency.mean()),
                "activation_frequency_std": float(activation_frequency.std()),
                "activation_frequency_min": float(activation_frequency.min()),
                "activation_frequency_max": float(activation_frequency.max()),
                "never_active_dimension_fraction": float(np.mean(activation_frequency == 0)),
            }
        )

        def values(key: str) -> np.ndarray:
            return np.asarray([entry[key] for entry in logs], dtype=np.float64)

        dynamics_rows.append(
            {
                "task": task,
                "zero_to_one_flips": int(values("zero_to_one_flips").sum()),
                "one_to_zero_flips": int(values("one_to_zero_flips").sum()),
                "net_flip_balance": int(
                    values("zero_to_one_flips").sum()
                    - values("one_to_zero_flips").sum()
                ),
                "memory_active_fraction": btsp_model.memory.active_fraction,
                "plateau_rate": float(values("plateau_rate").mean()),
                "mean_f_eff": float(values("plateau_probability").mean()),
                "familiarity_mean": float(values("familiarity").mean()),
                "familiarity_std": float(values("familiarity").std()),
                "novelty_mean": float(values("novelty").mean()),
                "novelty_std": float(values("novelty").std()),
                "interference_mean": float(values("interference").mean()),
                "interference_std": float(values("interference").std()),
                "ca1_nonzero_density": float(np.count_nonzero(ca1_train) / ca1_train.size),
            }
        )

        for label in np.unique(train_y):
            positions = np.flatnonzero(train_y == label)
            class_logs = [logs[int(position)] for position in positions]
            positive = [
                index + 1
                for index, entry in enumerate(class_logs)
                if entry["plateau_probability"] > 0
            ]
            first_positive = positive[0] if positive else None
            for window in (1, 5, 20):
                entries = class_logs[:window]
                early_rows.append(
                    {
                        "task": task,
                        "incremental_class": int(label),
                        "original_class": int(class_order[int(label)]),
                        "window_first_n_samples": window,
                        "samples_in_window": len(entries),
                        "familiarity_mean": float(np.mean([x["familiarity"] for x in entries])),
                        "novelty_mean": float(np.mean([x["novelty"] for x in entries])),
                        "interference_mean": float(np.mean([x["interference"] for x in entries])),
                        "f_eff_mean": float(np.mean([x["plateau_probability"] for x in entries])),
                        "plateau_count_sum": int(sum(x["plateau_count"] for x in entries)),
                        "first_positive_f_eff_occurrence": first_positive,
                        "zero_f_eff_prefix_length": (first_positive - 1) if first_positive else len(class_logs),
                    }
                )

        print(
            json.dumps(
                {
                    "task": task,
                    "vit_cumulative": task_rows["vit_ridge"]["cumulative_acc"],
                    "sparse_cumulative": task_rows["sparse_ridge"]["cumulative_acc"],
                    "btsp_cumulative": task_rows["btsp_cam"]["cumulative_acc"],
                    "runtime_seconds": time.perf_counter() - task_started,
                }
            ),
            flush=True,
        )

    write_csv(OUTPUT_DIR / "representation_ablation.csv", ablation_rows)
    write_csv(OUTPUT_DIR / "accuracy_drops.csv", drop_rows)
    write_csv(OUTPUT_DIR / "ridge_spectrum.csv", spectrum_rows)
    write_csv(OUTPUT_DIR / "representation_geometry.csv", geometry_rows)
    write_csv(OUTPUT_DIR / "sparse_diagnostics.csv", sparse_rows)
    write_csv(OUTPUT_DIR / "btsp_learning_dynamics.csv", dynamics_rows)
    write_csv(OUTPUT_DIR / "btsp_early_class_dynamics.csv", early_rows)

    final_rows = {row["method"]: row for row in ablation_rows if row["task"] == 5}
    final_drops = [row for row in drop_rows if row["task"] == 5]
    largest = max(final_drops, key=lambda row: row["cumulative_acc_drop"])
    summary = {
        "tasks": 5,
        "vit_ridge_final_cumulative_accuracy": final_rows["vit_ridge"]["cumulative_acc"],
        "sparse_ridge_final_cumulative_accuracy": final_rows["sparse_ridge"]["cumulative_acc"],
        "btsp_cam_final_cumulative_accuracy": final_rows["btsp_cam"]["cumulative_acc"],
        "vit_to_sparse_drop": final_drops[0]["cumulative_acc_drop"],
        "sparse_to_btsp_drop": final_drops[1]["cumulative_acc_drop"],
        "largest_task5_drop_transition": largest["transition"],
        "runtime_seconds": time.perf_counter() - started,
        "parameter_changes": "none",
    }
    (OUTPUT_DIR / "representation_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        "# Representation diagnostics\n\n"
        "All runs use the exact cached PILOT/ViT features from the formal run. "
        "No hyperparameter was changed. Large-matrix feature rank is reported as "
        "a deterministic 512-column principal-submatrix lower bound plus an exact "
        "sample/active-dimension upper bound. Effective rank is the participation "
        "ratio `(trace(A)^2 / ||A||_F^2)`. Large-matrix eigenvalue extrema use "
        "deterministic power/inverse-power iteration with the fitted Cholesky factor, "
        "except that proven rank deficiency sets `eig_min=0` exactly. Eigenvalues "
        "refer to unregularized `A`; condition numbers are LAPACK one-norm estimates "
        "for `A + gamma I`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
