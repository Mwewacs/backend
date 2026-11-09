import hashlib
import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, __version__ as xgboost_version

from backend.prepare_data import FEATURE_NAMES, ROOT, prepare_data


def temporal_split(examples):
    years = sorted({example["episode_group"] for example in examples if example["label"]})
    if len(years) < 3:
        return None
    validation_year, test_year = years[-2:]
    partitions = [
        [example for example in examples if example["issue_date"][:4] < validation_year],
        [example for example in examples if example["issue_date"][:4] == validation_year],
        [example for example in examples if example["issue_date"][:4] >= test_year],
    ]
    if any({example["label"] for example in partition} != {0, 1} for partition in partitions):
        return None
    if any(max(example["window_end"] for example in earlier) >= min(example["issue_date"] for example in later) for earlier, later in zip(partitions, partitions[1:])):
        return None
    for earlier, later in zip(partitions, partitions[1:]):
        cutoff = datetime.fromisoformat(min(example["issue_date"] for example in later)).replace(tzinfo=timezone.utc)
        for example in earlier:
            if not example.get("outcome_available_at"):
                continue
            available = datetime.fromisoformat(example["outcome_available_at"].replace("Z", "+00:00"))
            available = available.replace(tzinfo=timezone.utc) if available.tzinfo is None else available
            if available > cutoff:
                return None
    episode_sets = [{example.get("outbreak_episode_id") for example in partition if example.get("outbreak_episode_id")} for partition in partitions]
    if any(episode_sets[first] & episode_sets[second] for first, second in [(0, 1), (0, 2), (1, 2)]):
        return None
    return partitions


def feature_frame(examples):
    return pd.DataFrame([example["features"] for example in examples], columns=FEATURE_NAMES).astype(float)


def candidate_models(labels):
    positive = int(sum(labels))
    weight = (len(labels) - positive) / positive if positive else 1
    boosting = {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "random_state": 42, "n_jobs": 1, "eval_metric": "logloss", "scale_pos_weight": weight}
    classifiers = {
        "logistic_regression": [StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)],
        "random_forest": [RandomForestClassifier(n_estimators=150, max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=1)],
        "xgboost_constrained": [XGBClassifier(**boosting, monotone_constraints=(1, 0, 0, 0, 0))],
        "xgboost_unconstrained": [XGBClassifier(**boosting)],
    }
    return {name: Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))] + [(f"step_{index}", estimator) for index, estimator in enumerate(steps)]) for name, steps in classifiers.items()}


def choose_threshold(labels, probabilities):
    choices = []
    for threshold in sorted(set([0.0, 1.0, *map(float, probabilities)])):
        predicted = probabilities >= threshold
        precision = precision_score(labels, predicted, zero_division=0)
        recall = recall_score(labels, predicted, zero_division=0)
        score = (5 * precision * recall / (4 * precision + recall)) if precision + recall else 0
        choices.append((score, f1_score(labels, predicted, zero_division=0), precision, threshold))
    return max(choices)[-1]


def evaluate(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    both_classes = len(set(labels)) == 2
    return {"sensitivity": float(recall_score(labels, predicted, zero_division=0)), "precision": float(precision_score(labels, predicted, zero_division=0)), "f1": float(f1_score(labels, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None, "pr_auc": float(average_precision_score(labels, probabilities)) if both_classes else None, "brier_score": float(brier_score_loss(labels, probabilities)), "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}


def evaluate_onsets(examples, probabilities, threshold):
    onsets = {(example["area_id"], example["onset_date"]) for example in examples if example["onset_date"]}
    detected = {}
    false_windows = 0
    for example, probability in zip(examples, probabilities):
        if probability < threshold:
            continue
        if not example["onset_date"]:
            false_windows += 1
            continue
        key = (example["area_id"], example["onset_date"])
        lead_days = (datetime.fromisoformat(example["onset_date"]) - datetime.fromisoformat(example["issue_date"])).days
        detected[key] = max(detected.get(key, 0), lead_days)
    return {"observed_onsets": len(onsets), "detected_onsets": len(detected), "missed_onsets": len(onsets) - len(detected), "onset_sensitivity": len(detected) / len(onsets) if onsets else None, "median_lead_days": float(np.median(list(detected.values()))) if detected else None, "false_alert_windows": false_windows, "observed_windows": len(examples)}


def train_models(dataset=None, preparation=None, output_directory=None):
    output = Path(output_directory or ROOT / "artifacts")
    output.mkdir(parents=True, exist_ok=True)
    if dataset is None or preparation is None:
        dataset, preparation = prepare_data(output_directory=output)
    examples = dataset["examples"]
    split = temporal_split(examples)
    reasons = []
    if split is None:
        reasons.append("Three chronological onset years with positive and negative examples are required for training, validation and independent testing.")
    if not preparation["publication_dates_verified"]:
        reasons.append("Source publication dates are not verified, so historical availability cannot be established.")
    if not preparation.get("episode_ids_verified"):
        reasons.append("Independent outbreak episode identifiers are not verified.")
    labels = [example["label"] for example in examples]
    signature = {"examples": examples, "features": FEATURE_NAMES, "seed": 42, "training_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "preparation_code": hashlib.sha256((ROOT / "backend/prepare_data.py").read_bytes()).hexdigest(), "scikit_learn": sklearn.__version__, "xgboost": xgboost_version}
    report = {
        "version": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12],
        "created_at": datetime.now(timezone.utc).isoformat(), "ready": False,
        "status": "insufficient_data", "reasons": reasons, "selected_model": None,
        "target_definition": preparation["target_definition"], "quiet_period_days": 14, "horizon_days": 7,
        "features": FEATURE_NAMES, "data": {key: preparation[key] for key in ["source_files", "raw_records", "daily_records", "weekly_records", "areas", "examples", "positive_examples", "unique_onsets", "years_with_onsets"]},
        "candidates": [], "evaluation": None,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "xgboost": xgboost_version}, "random_seed": 42,
        "limitations": preparation["limitations"] + ["The episode label is a research proxy, not an official outbreak declaration.", "The monotonic prior-case constraint is an experimental assumption; its effect requires independent evaluation."],
    }
    if set(labels) != {0, 1}:
        report["reasons"].append("The prepared examples must include both onset and non-onset outcomes.")
        (output / "model-report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    training = split[0] if split else examples
    version_directory = output / "models" / report["version"]
    version_directory.mkdir(parents=True, exist_ok=True)
    training_labels = np.array([example["label"] for example in training])
    fitted = {}
    for name, pipeline in candidate_models(training_labels).items():
        pipeline.fit(feature_frame(training), training_labels)
        joblib.dump(pipeline, version_directory / f"{name}.joblib")
        fitted[name] = pipeline
        parameters = json.loads(json.dumps({key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in pipeline.steps[-1][1].get_params().items()}, allow_nan=False))
        candidate = {"name": name, "status": "trained_exploratory" if split is None else "evaluated", "training_rows": len(training), "validation": None, "test": None, "threshold": None, "hyperparameters": parameters, "artifact": f"models/{report['version']}/{name}.joblib"}
        if split:
            validation_labels = np.array([example["label"] for example in split[1]])
            probabilities = pipeline.predict_proba(feature_frame(split[1]))[:, 1]
            candidate["threshold"] = choose_threshold(validation_labels, probabilities)
            candidate["medium_threshold"] = float(np.quantile(probabilities[validation_labels == 0], 0.9))
            candidate["validation"] = evaluate(validation_labels, probabilities, candidate["threshold"])
        report["candidates"].append(candidate)
    report["status"] = "exploratory_models_trained" if split is None else "retrospective_evaluation"
    if split:
        selected = max(report["candidates"], key=lambda candidate: (candidate["validation"]["f1"], candidate["validation"]["sensitivity"]))
        for candidate in report["candidates"]:
            test_labels = np.array([example["label"] for example in split[2]])
            probabilities = fitted[candidate["name"]].predict_proba(feature_frame(split[2]))[:, 1]
            candidate["test"] = evaluate(test_labels, probabilities, candidate["threshold"])
            candidate["onset_evaluation"] = evaluate_onsets(split[2], probabilities, candidate["threshold"])
        report["evaluation"] = {"training_years": sorted({example["issue_date"][:4] for example in split[0]}), "validation_years": sorted({example["issue_date"][:4] for example in split[1]}), "test_years": sorted({example["issue_date"][:4] for example in split[2]}), "selection_basis": "Validation F1, then sensitivity; test outcomes are not used for model selection.", "risk_band_rule": "The high threshold maximizes validation F2. The medium threshold is the 90th percentile of validation non-onset scores. These are research rules requiring review before operational use."}
        if selected["medium_threshold"] >= selected["threshold"]:
            reasons.append("Validation probabilities do not support distinct medium and high risk thresholds.")
        if not reasons:
            report.update({"ready": True, "status": "validated_research_model", "selected_model": selected["name"], "threshold": selected["threshold"], "medium_threshold": selected["medium_threshold"], "training_cutoff": max(example["window_end"] for example in training)})
            joblib.dump(fitted[selected["name"]], output / "active-model.joblib")
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (output / "model-report.json").write_text(serialized)
    (version_directory / "report.json").write_text(serialized)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-only", action="store_true")
    arguments = parser.parse_args()
    prepared, preparation = prepare_data(curated_only=arguments.curated_only)
    print(json.dumps(train_models(prepared, preparation), indent=2, allow_nan=False))
