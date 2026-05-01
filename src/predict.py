import json
import math
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODELS_DIR = "models"

rf_model = joblib.load(f"{MODELS_DIR}/rf_model.pkl")
lgb_model = joblib.load(f"{MODELS_DIR}/lgb_model.pkl")
xgb_model = joblib.load(f"{MODELS_DIR}/xgb_model.pkl")
ensemble_model = joblib.load(f"{MODELS_DIR}/ensemble_model.pkl")

imputer = joblib.load(f"{MODELS_DIR}/imputer.pkl")
scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
feature_columns = joblib.load(f"{MODELS_DIR}/features.pkl")

with open(f"{MODELS_DIR}/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)



def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_input():
    from input import user_input

    if not isinstance(user_input, dict):
        raise ValueError("input.py must define user_input as a dictionary.")

    return user_input


def normalize_input(user_input):
    row = {}

    for col in feature_columns:
        val = user_input.get(col, np.nan)

        try:
            row[col] = float(val) if val is not None else np.nan
        except Exception:
            row[col] = np.nan

    return pd.DataFrame([row], columns=feature_columns)


def preprocess(df):
    df_imputed = pd.DataFrame(
        imputer.transform(df),
        columns=feature_columns
    )

    df_scaled = pd.DataFrame(
        scaler.transform(df_imputed),
        columns=feature_columns
    )

    return df_scaled, df_imputed


def quality_rating_general(value, ideal, standard):
    if pd.isna(value):
        return np.nan

    if standard == ideal:
        return np.nan

    return max(0.0, ((value - ideal) / (standard - ideal)) * 100.0)


def quality_rating_inverse(value, ideal, standard):
    if pd.isna(value):
        return np.nan

    if ideal == standard:
        return np.nan

    return max(0.0, ((ideal - value) / (ideal - standard)) * 100.0)


def quality_rating_ph(value):
    if pd.isna(value):
        return np.nan

    if 7.0 <= value <= 8.5:
        return max(0.0, ((value - 7.0) / (8.5 - 7.0)) * 100.0)

    return max(0.0, ((7.0 - value) / (7.0 - 6.5)) * 100.0)


def calculate_wqi(raw_row):
    refs = metadata["wqi_reference"]

    q_values = {}
    weights = {}

    for param, cfg in refs.items():
        if param == "ph_rule_type":
            continue

        if param not in raw_row.index:
            continue

        value = raw_row[param]

        if pd.isna(value):
            continue

        ideal = cfg["ideal"]
        standard = cfg["standard"]
        weight = cfg["weight"]

        if param == "dissolved_oxygen":
            qi = quality_rating_inverse(value, ideal, standard)
        elif param == "ph":
            qi = quality_rating_ph(value)
        else:
            qi = quality_rating_general(value, ideal, standard)

        q_values[param] = qi
        weights[param] = weight

    if not q_values:
        return float("nan")

    numerator = sum(q_values[p] * weights[p] for p in q_values)
    denominator = sum(weights.values())

    return numerator / denominator if denominator else float("nan")


def contamination_from_wqi(wqi):
    if pd.isna(wqi):
        return "Unknown", float("nan")

    score = round(float(wqi), 2)

    safe_max = metadata["contamination_rules"]["safe_max_wqi"]
    moderate_max = metadata["contamination_rules"]["moderate_max_wqi"]

    if wqi <= safe_max:
        return "Safe", score
    elif wqi <= moderate_max:
        return "Moderate", score
    else:
        return "Unsafe", score



def blue_baby_risk(raw_row):
    nitrate = raw_row.get("nitrate", np.nan)

    if pd.isna(nitrate):
        return 0.0

    threshold = metadata["disease_rules"]["blue_baby_nitrate_threshold"]
    x = (nitrate - threshold) / 2.5

    return round(clamp(sigmoid(x)), 4)


def fluorosis_risk(user_input):
    fluoride = user_input.get("fluoride", np.nan)

    try:
        fluoride = float(fluoride)
    except Exception:
        fluoride = np.nan

    if pd.isna(fluoride):
        return 0.0

    threshold = metadata["disease_rules"]["fluorosis_fluoride_threshold"]
    x = (fluoride - threshold) / 0.35

    return round(clamp(sigmoid(x)), 4)


def bacterial_contamination_risk(raw_row):
    dissolved_oxygen = raw_row.get("dissolved_oxygen", np.nan)
    fecal_coliform = raw_row.get("fecal_coliform", np.nan)
    total_coliform = raw_row.get("total_coliform", np.nan)

    risks = []

    if not pd.isna(dissolved_oxygen):
        threshold = metadata["disease_rules"]["bacterial_contamination_do_threshold"]
        risks.append(sigmoid((threshold - dissolved_oxygen) / 0.8))

    if not pd.isna(fecal_coliform):
        risks.append(sigmoid((fecal_coliform - 2500.0) / 700.0))

    if not pd.isna(total_coliform):
        risks.append(sigmoid((total_coliform - 5000.0) / 1200.0))

    if not risks:
        return 0.0

    return round(clamp(sum(risks) / len(risks)), 4)


def overall_disease_risk(blue_baby, fluorosis, bacterial):
    score = max(blue_baby, fluorosis, bacterial)

    if score < 0.33:
        label = "Low"
    elif score < 0.66:
        label = "Moderate"
    else:
        label = "High"

    return label, round(score, 4)


def potability_from_probability(prob):
    return "Potable - 1" if prob >= 0.5 else "Not Potable - 0"


def get_accuracy(model_key):
    metrics = metadata.get("model_metrics", {})
    acc = metrics.get(model_key, {}).get("accuracy", None)

    if acc is None:
        return "N/A"

    return f"{float(acc):.4f}"


def print_model_output(
    model_name,
    model_key,
    probability,
    contamination_label,
    contamination_score,
    disease_label,
    disease_score
):
    print(f"Model               : {model_name}")
    print(f"Accuracy            : {get_accuracy(model_key)}")
    print(f"Potability          : {potability_from_probability(probability)}")
    print(f"Contamination       : {contamination_label}")

    if pd.isna(contamination_score):
        print("Contamination Score : Unknown")
    else:
        print(f"Contamination Score : {contamination_score:.2f}")

    print(f"Disease Risk        : {disease_label}")
    print(f"Disease Risk Score  : {disease_score:.4f}")
    print("-" * 45)



def main():
    user_input = load_input()

    raw_df = normalize_input(user_input)
    raw_row = raw_df.iloc[0]

    X_scaled, X_imputed = preprocess(raw_df)

    rf_prob = float(rf_model.predict_proba(X_scaled)[0][1])
    lgb_prob = float(lgb_model.predict_proba(X_scaled)[0][1])
    xgb_prob = float(xgb_model.predict_proba(X_scaled)[0][1])
    ensemble_prob = float(ensemble_model.predict_proba(X_scaled)[0][1])

    wqi = calculate_wqi(raw_row)
    contamination_label, contamination_score = contamination_from_wqi(wqi)

    blue_baby = blue_baby_risk(raw_row)
    fluorosis = fluorosis_risk(user_input)
    bacterial = bacterial_contamination_risk(raw_row)

    disease_label, disease_score = overall_disease_risk(
        blue_baby,
        fluorosis,
        bacterial
    )

    print("\n===== WATER QUALITY PREDICTION =====\n")

    print_model_output(
        "Random Forest",
        "rf",
        rf_prob,
        contamination_label,
        contamination_score,
        disease_label,
        disease_score
    )

    print_model_output(
        "LightGBM",
        "lgb",
        lgb_prob,
        contamination_label,
        contamination_score,
        disease_label,
        disease_score
    )

    print_model_output(
        "XGBoost",
        "xgb",
        xgb_prob,
        contamination_label,
        contamination_score,
        disease_label,
        disease_score
    )

    print_model_output(
        "Ensemble",
        "ensemble",
        ensemble_prob,
        contamination_label,
        contamination_score,
        disease_label,
        disease_score
    )

    print("\nDetailed Disease Breakdown")
    print(f"Blue baby syndrome      : {blue_baby:.4f}")
    print(f"Fluorosis               : {fluorosis:.4f}")
    print(f"Bacterial contamination : {bacterial:.4f}")

    print("\nWater Quality Index")
    if pd.isna(wqi):
        print("WQI                  : Not enough input parameters")
    else:
        print(f"WQI                  : {wqi:.2f}")

    print("\n===================================\n")


if __name__ == "__main__":
    main()
