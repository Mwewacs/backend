import json

import numpy as np

from backend.tests.given import given_examples
from backend.train_models import choose_threshold, evaluate, evaluate_onsets, temporal_split, train_models


def test_splits_whole_years_chronologically():
    partitions = temporal_split(given_examples())
    assert [[example["issue_date"] for example in partition] for partition in partitions] == [["2017-01-15", "2017-02-15"], ["2020-01-15", "2020-02-15"], ["2023-01-15", "2023-02-15"]]


def test_requires_three_onset_years():
    assert temporal_split(given_examples((2017, 2023))) is None


def test_rejects_a_single_class_test_period():
    examples = given_examples()
    assert temporal_split([example for example in examples if example["issue_date"] != "2023-01-15"]) is None


def test_keeps_the_same_episode_out_of_different_partitions():
    examples = given_examples()
    examples[-1]["outbreak_episode_id"] = "outbreak-2017"
    assert temporal_split(examples) is None


def test_excludes_training_outcomes_published_after_validation_started():
    examples = given_examples()
    examples[0]["outcome_available_at"] = "2021-01-01T00:00:00Z"
    assert temporal_split(examples) is None


def test_selects_the_threshold_from_validation_outcomes():
    assert choose_threshold(np.array([0, 1]), np.array([0.1, 0.9])) == 0.9


def test_reports_missed_onsets_and_false_alerts():
    metrics = evaluate([0, 1], np.array([0.9, 0.1]), 0.5)
    assert {name: metrics[name] for name in ["sensitivity", "precision", "f1", "confusion_matrix"]} == {"sensitivity": 0.0, "precision": 0.0, "f1": 0.0, "confusion_matrix": [[0, 1], [1, 0]]}


def test_reports_correct_classification_metrics():
    metrics = evaluate([0, 1], np.array([0.1, 0.9]), 0.5)
    assert {name: metrics[name] for name in ["sensitivity", "precision", "f1", "roc_auc", "pr_auc", "confusion_matrix"]} == {"sensitivity": 1.0, "precision": 1.0, "f1": 1.0, "roc_auc": 1.0, "pr_auc": 1.0, "confusion_matrix": [[1, 0], [0, 1]]}


def test_counts_onsets_once_and_measures_lead_time():
    examples = given_examples((2023,))
    assert evaluate_onsets(examples, [0.9, 0.9], 0.5) == {"observed_onsets": 1, "detected_onsets": 1, "missed_onsets": 0, "onset_sensitivity": 1.0, "median_lead_days": 2.0, "false_alert_windows": 1, "observed_windows": 2}


def test_reports_no_lead_time_for_missed_onsets():
    examples = given_examples((2023,))
    assert evaluate_onsets(examples, [0.1, 0.1], 0.5) == {"observed_onsets": 1, "detected_onsets": 0, "missed_onsets": 1, "onset_sensitivity": 0.0, "median_lead_days": None, "false_alert_windows": 0, "observed_windows": 2}


def given_preparation(examples):
    return {"source_files": 1, "raw_records": 50, "daily_records": 50, "weekly_records": 0, "areas": 1, "examples": len(examples), "positive_examples": sum(example["label"] for example in examples), "unique_onsets": 2, "years_with_onsets": ["2017", "2023"], "target_definition": "Research onset proxy.", "publication_dates_verified": False, "episode_ids_verified": False, "limitations": []}


def test_trains_real_candidates_without_inventing_test_metrics(tmp_path):
    examples = given_examples((2017, 2023))
    report = train_models({"examples": examples}, given_preparation(examples), tmp_path)
    assert report["ready"] is False
    assert [(candidate["name"], candidate["validation"], candidate["test"]) for candidate in report["candidates"]] == [("logistic_regression", None, None), ("random_forest", None, None), ("xgboost_constrained", None, None), ("xgboost_unconstrained", None, None)]
    assert len(list(tmp_path.glob("models/*/*.joblib"))) == 4
    assert json.loads((tmp_path / "model-report.json").read_text()) == report
    assert (tmp_path / "active-model.joblib").exists() is False


def test_does_not_train_on_a_single_class(tmp_path):
    examples = [example for example in given_examples() if example["label"] == 0]
    report = train_models({"examples": examples}, given_preparation(examples), tmp_path)
    assert report["ready"] is False
    assert report["candidates"] == []
    assert list(tmp_path.glob("models/*/*.joblib")) == []


def test_saves_an_active_model_only_after_qualified_evaluation(tmp_path):
    examples = given_examples()
    preparation = {**given_preparation(examples), "publication_dates_verified": True, "episode_ids_verified": True}
    report = train_models({"examples": examples}, preparation, tmp_path)
    assert report["ready"] is True
    assert report["selected_model"] == "logistic_regression"
    assert report["evaluation"]["test_years"] == ["2023"]
    assert report["candidates"][0]["test"]["sensitivity"] == 1.0
    assert report["medium_threshold"] < report["threshold"]
    assert (tmp_path / "active-model.joblib").is_file()


def test_withholds_auc_when_outcomes_have_one_class():
    metrics = evaluate([0, 0], np.array([0.1, 0.2]), 0.5)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
