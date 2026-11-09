import math
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from backend.forecast_features import FEATURES

GROUPS = {"location": ["latitude", "longitude"], "season": ["month_sin", "month_cos"], "rainfall": ["rainfall_lag_week", "rainfall_earlier_week"], "temperature": ["temperature_lag_week", "temperature_earlier_week"], "water": ["water_access"], "sanitation": ["sanitation_access"], "history": ["previous_annual_occurrence"]}


def forest_effects(model, features):
    if not hasattr(model, "steps") or not isinstance(model.steps[-1][1], RandomForestClassifier):
        return None
    classifier = model.steps[-1][1]
    if list(classifier.classes_) != [0, 1]:
        return None
    transformed = np.asarray(model[:-1].transform(pd.DataFrame(features, columns=FEATURES).astype(float)), dtype=np.float32)
    contributions = np.zeros((len(features), len(FEATURES)))
    baseline = 0.0
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        probabilities = tree.value[:, 0, 1] / tree.value[:, 0].sum(axis=1)
        baseline += probabilities[0]
        for index, values in enumerate(transformed):
            node = 0
            while tree.children_left[node] != -1:
                feature = tree.feature[node]
                child = tree.children_left[node] if values[feature] <= tree.threshold[node] else tree.children_right[node]
                contributions[index, feature] += probabilities[child] - probabilities[node]
                node = child
    return {"method": "forest_path", "baseline_score": float(baseline / len(classifier.estimators_)), "contributions": [dict(zip(FEATURES, map(float, values / len(classifier.estimators_)))) for values in contributions]}


def forecast_reasons(inputs, issue_date, seasonal=False, contributions=None):
    rainfall = inputs["rainfall_lag_week"] + inputs["rainfall_earlier_week"]
    temperature = (inputs["temperature_lag_week"] + inputs["temperature_earlier_week"]) / 2
    month = date.fromisoformat(issue_date).strftime("%B")
    descriptions = {
        "location": ("Regional case patterns", "Patterns the model learned from the locations of past reported cases."),
        "season": ("Seasonal pattern", f"The {month} pattern learned from past reported cases."),
        "rainfall": ("Seasonal rainfall" if seasonal else "Rainfall", f"{rainfall:.1f} mm across the 14 weather-input days."),
        "temperature": ("Seasonal temperature" if seasonal else "Temperature", f"{temperature:.1f}°C average across the 14 weather-input days."),
        "water": ("Historical water access", f"{inputs['water_access']:.1f}% improved water access in the historical source." if inputs["water_access"] is not None else "No dated water-access record; the model uses its training median."),
        "sanitation": ("Historical sanitation", f"{inputs['sanitation_access']:.1f}% improved sanitation access in the historical source." if inputs["sanitation_access"] is not None else "No dated sanitation record; the model uses its training median."),
        "history": ("Past cholera reports", f"Cholera recorded in {inputs['previous_annual_occurrence'] * 100:.0f}% of available earlier source years." if inputs["previous_annual_occurrence"] is not None else "No earlier annual case record; the model uses its training median.")
    }
    if contributions is None:
        return [{"key": key, "title": descriptions[key][0], "detail": descriptions[key][1], "direction": "input", "contribution": None} for key in ["rainfall", "temperature", "season"]]
    reasons = []
    for key, names in GROUPS.items():
        effect = sum(contributions[name] for name in names)
        if not math.isfinite(effect) or abs(effect) < 0.001:
            continue
        missing = any(inputs[name] is None for name in names)
        reasons.append({"key": key, "title": descriptions[key][0], "detail": descriptions[key][1], "direction": "input" if missing else "raises" if effect > 0 else "lowers", "contribution": effect})
    if not reasons:
        return forecast_reasons(inputs, issue_date, seasonal)
    return sorted(reasons, key=lambda reason: -abs(reason["contribution"]))[:3]
