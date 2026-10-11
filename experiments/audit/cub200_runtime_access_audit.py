"""Instrument the frozen CUB sparse-cache trajectory without changing its code or caches."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from btsp_cam.ridge import StreamingRidge
from run_btsp_cam_cub200 import ca1, evaluate, fit_replay, load_codes, load_config, make_model

DIAG = ROOT / "diagnostics"
SPARSE = DIAG / "cub200_sparse_cache"


def sha_state(model, ridge) -> dict[str, str]:
    arrays = dict(W_B=model.memory.weights, eligibility=model.traces.values,
                  projection=model.encoder.projection, bias=model.encoder.bias,
                  ridge_A=ridge.a, ridge_B=ridge.b, ridge_W_out=ridge.weights,
                  seen_classes=model.seen_classes)
    result = {name: hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
              for name, value in arrays.items()}
    rng_and_counters = dict(projection_rng=model.encoder.rng.bit_generator.state,
                            plateau_rng=model.plateau_rng.bit_generator.state,
                            synapse_rng=model.synapse_rng.bit_generator.state,
                            task_index=model.current_task_index,
                            train_order_state=model.train_order_state,
                            last_summary=model.last_task_summary,
                            metrics_history=model.task_metrics_history,
                            last_ltp=model.memory.last_zero_to_one,
                            last_ltd=model.memory.last_one_to_zero)
    result["rng_and_counters"] = hashlib.sha256(json.dumps(
        rng_and_counters, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
        else x.item() if isinstance(x, np.generic) else str(x)).encode()).hexdigest()
    return result


def load_provenance() -> dict[str, dict[int, list[dict]]]:
    rows = {"train": defaultdict(list), "test": defaultdict(list)}
    for line in (DIAG / "cub200_cache_row_provenance.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        rows[row["split"]][row["task"]].append(row)
    for split in rows:
        assert len(rows[split]) == 20
        for task, task_rows in rows[split].items():
            task_rows.sort(key=lambda x: x["sparse_row_index"])
            assert [r["sparse_row_index"] for r in task_rows] == list(range(len(task_rows)))
            assert len({r["sample_id"] for r in task_rows}) == len(task_rows)
    return rows


def capture_eval(ridge, wt, task, sources, source_ids):
    queue = [list(ids) for ids in source_ids]
    original = ridge.logits
    captured = []
    prediction_ids = []
    def traced_logits(features):
        ids = queue.pop(0)
        assert len(ids) == len(features)
        values = original(features)
        prediction_ids.extend(ids)
        captured.append(values.copy())
        return values
    ridge.logits = traced_logits
    try:
        metrics, _ = evaluate(ridge, wt, task, sources)
    finally:
        ridge.logits = original
    assert not queue
    return metrics, captured, prediction_ids


def main() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from project root: {ROOT}")
    raw, config = load_config()
    assert raw["readout"] == "CURRENT_WB_REPLAY_REFIT"
    provenance = load_provenance()
    all_train = {r["sample_id"] for task in provenance["train"].values() for r in task}
    all_test = {r["sample_id"] for task in provenance["test"].values() for r in task}
    assert len(all_train) == 5994 and len(all_test) == 5794 and not all_train & all_test
    reference = list(csv.DictReader((DIAG / "cub200_full_taskwise.csv").open(newline="", encoding="utf-8")))
    assert len(reference) == 20
    model = make_model(config)
    train, tests, train_ids, test_ids = {}, {}, {}, {}
    task_records = []
    independent = []
    state_result = {}
    canary = {}
    final_prediction_path = DIAG / "cub200_task20_prediction_canary.npz"
    if final_prediction_path.exists():
        raise FileExistsError(final_prediction_path)
    for task in range(1, 21):
        ids = [r["sample_id"] for r in provenance["train"][task]]
        sparse_meta = json.loads(np.load(SPARSE / f"train_task{task:02d}.npz", allow_pickle=False)["metadata"].tobytes())
        train[task] = load_codes(SPARSE / f"train_task{task:02d}.npz", sparse_meta)
        train_ids[task] = ids
        assert len(ids) == len(train[task][1])
        test_meta = json.loads(np.load(SPARSE / f"test_task{task:02d}.npz", allow_pickle=False)["metadata"].tobytes())
        tests[task] = load_codes(SPARSE / f"test_task{task:02d}.npz", test_meta)
        test_ids[task] = [r["sample_id"] for r in provenance["test"][task]]
        assert len(test_ids[task]) == len(tests[task][1])
        context = {"id": None}
        eligibility_ids, writer_ids, ridge_ids = [], [], []
        trace_update = model.traces.update
        memory_update = model.memory.update
        def traced_eligibility(code, label):
            assert context["id"] is not None
            eligibility_ids.append(context["id"])
            return trace_update(code, label)
        def traced_memory(code, plateaus, probability, rng):
            assert context["id"] is not None
            writer_ids.append(context["id"])
            return memory_update(code, plateaus, probability, rng)
        model.traces.update = traced_eligibility
        model.memory.update = traced_memory
        try:
            for row, label, sample_id in zip(train[task][0].toarray(), train[task][1], ids):
                context["id"] = sample_id
                model._learn_code(row, int(label))
        finally:
            context["id"] = None
            model.traces.update = trace_update
            model.memory.update = memory_update
        active_ridge_ids = []
        original_update = StreamingRidge.update
        def traced_ridge_update(self, features, labels):
            assert len(active_ridge_ids) == len(features) == len(labels)
            ridge_ids.extend(active_ridge_ids)
            return original_update(self, features, labels)
        def load_training(source):
            active_ridge_ids[:] = train_ids[source]
            return train[source]
        StreamingRidge.update = traced_ridge_update
        try:
            ridge, wt = fit_replay(model.memory.weights, task, config.gamma, load_training)
        finally:
            StreamingRidge.update = original_update
        source_pairs = [tests[source] for source in range(1, task + 1)]
        source_ids = [test_ids[source] for source in range(1, task + 1)]
        if task == 20:
            before = sha_state(model, ridge)
        _, logits_by_source, prediction_ids = capture_eval(ridge, wt, task, source_pairs, source_ids)
        metric_ids = [sample_id for ids_part in source_ids for sample_id in ids_part]
        assert prediction_ids == metric_ids
        labels_all = np.concatenate([pair[1] for pair in source_pairs])
        logits_all = np.concatenate(logits_by_source)
        predictions = logits_all.argmax(axis=1)
        accuracy = float(np.count_nonzero(predictions == labels_all) / len(labels_all))
        baseline_accuracy = float(reference[task - 1]["cumulative_accuracy"])
        assert abs(accuracy - baseline_accuracy) < 1e-12, (task, accuracy, baseline_accuracy)
        independent.append(dict(task=task, correct=int(np.count_nonzero(predictions == labels_all)),
                                total=len(labels_all), cumulative_accuracy=accuracy,
                                baseline_accuracy=baseline_accuracy))
        seen_train = set().union(*(set(train_ids[s]) for s in range(1, task + 1)))
        future_train = all_train - seen_train
        current_train = set(train_ids[task])
        assert set(writer_ids) == set(eligibility_ids) == current_train
        assert set(ridge_ids) == seen_train
        assert set(prediction_ids) == set(metric_ids) == set().union(*(set(test_ids[s]) for s in range(1, task + 1)))
        record = dict(task=task, writer_count=len(writer_ids), eligibility_count=len(eligibility_ids),
                      ridge_fit_count=len(ridge_ids), test_predict_count=len(prediction_ids),
                      test_metric_count=len(metric_ids),
                      test_in_writer=len(set(writer_ids) & all_test),
                      test_in_eligibility=len(set(eligibility_ids) & all_test),
                      test_in_ridge=len(set(ridge_ids) & all_test),
                      future_train_in_writer=len(set(writer_ids) & future_train),
                      future_train_in_eligibility=len(set(eligibility_ids) & future_train),
                      future_train_in_ridge=len(set(ridge_ids) & future_train),
                      writer_exact_current_train=True, ridge_exact_seen_train=True,
                      writer_update_ids=writer_ids, eligibility_update_ids=eligibility_ids,
                      ridge_fit_ids=ridge_ids, prediction_ids=prediction_ids,
                      metric_ids=metric_ids)
        task_records.append(record)
        if task == 20:
            rng = np.random.default_rng(20261011)
            permuted_labels = labels_all[rng.permutation(len(labels_all))]
            offset = 0
            permuted_sources = []
            for codes, old_labels in source_pairs:
                n = len(old_labels)
                permuted_sources.append((codes, permuted_labels[offset:offset + n].copy()))
                offset += n
            _, permuted_logits_by_source, permuted_ids = capture_eval(
                ridge, wt, task, permuted_sources, source_ids)
            permuted_logits = np.concatenate(permuted_logits_by_source)
            assert permuted_ids == prediction_ids
            canary = dict(task=20, sample_count=len(labels_all),
                          logits_bit_identical=np.array_equal(logits_all, permuted_logits),
                          predictions_identical=np.array_equal(predictions, permuted_logits.argmax(1)),
                          original_accuracy=accuracy,
                          permuted_label_accuracy=float(np.mean(predictions == permuted_labels)),
                          permutation_seed=20261011)
            assert canary["logits_bit_identical"] and canary["predictions_identical"]
            after = sha_state(model, ridge)
            state_result = dict(before=before, after=after,
                                changed_components=[key for key in before if before[key] != after[key]])
            assert not state_result["changed_components"]
            with final_prediction_path.open("xb") as stream:
                np.savez_compressed(stream, labels=labels_all, predictions=predictions,
                                    logits=logits_all, permuted_labels=permuted_labels)
        print(json.dumps(dict(task=task, accuracy=accuracy, writer=len(writer_ids),
                              ridge=len(ridge_ids), test=len(prediction_ids))), flush=True)
    average = float(np.mean([row["cumulative_accuracy"] for row in independent]))
    summary = json.loads((DIAG / "cub200_full_summary.json").read_text(encoding="utf-8"))
    assert abs(average - summary["average_incremental_accuracy"]) < 1e-12
    assert abs(independent[-1]["cumulative_accuracy"] - summary["final_accuracy"]) < 1e-12
    output = dict(status="PASS", source="instrumented_replay_from_existing_CUB_sparse_caches",
                  baseline_taskwise_accuracy_exact_match=True, dataset_train_count=5994,
                  dataset_test_count=5794, tasks=task_records,
                  task20_ridge_unique_sample_count=len(set(task_records[-1]["ridge_fit_ids"])),
                  task20_ridge_per_task_train_counts={str(t): len(train_ids[t]) for t in range(1, 21)},
                  historical_train_replay=True, strict_exemplar_free=False,
                  test_label_canary=canary, evaluation_state_mutation=state_result,
                  independent_metrics=dict(taskwise=independent, average_incremental_accuracy=average,
                                           final_accuracy=independent[-1]["cumulative_accuracy"],
                                           predictions_file=final_prediction_path.name))
    (DIAG / "cub200_runtime_access_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="PASS", average=average, final=independent[-1]["cumulative_accuracy"],
                          test_label_canary=canary,
                          state_changes=state_result["changed_components"])), flush=True)


if __name__ == "__main__":
    main()
