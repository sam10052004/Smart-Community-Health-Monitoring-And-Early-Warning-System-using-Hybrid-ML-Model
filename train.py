import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


TRAIN_FILE = os.path.join("water_data", "clean_water_quality (2).csv")
TEST_FILE = os.path.join("water_data", "test_water_quality.csv")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "ph",
    "temperature",
    "dissolved_oxygen",
    "conductivity",
    "bod",
    "nitrate",
    "fecal_coliform",
    "total_coliform",
    "hardness",
    "solids",
    "chloramines",
    "sulfate",
    "organic_carbon",
    "trihalomethanes",
    "turbidity",
]

TARGET_COLUMN = "target"

PREDICTION_RULE_COLUMNS = [
    "ph",
    "dissolved_oxygen",
    "conductivity",
    "bod",
    "nitrate",
    "fecal_coliform",
    "total_coliform",
    "hardness",
    "solids",
    "chloramines",
    "sulfate",
    "organic_carbon",
    "trihalomethanes",
    "turbidity",
    "temperature",
]



def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def advanced_clean_preprocess_split_and_save(raw_df, output_dir="water_data"):
    import os
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)

    df = raw_df.copy()

    
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates()

    
    df.replace(
        ["", " ", "na", "n/a", "nan", "null", "-", "--", "bdl"],
        np.nan,
        inplace=True
    )

    
    def extract_numeric(value):
        if pd.isna(value):
            return np.nan

        value = str(value).strip().lower().replace("–", "-")

        
        if "-" in value:
            nums = pd.Series(value.split("-")).str.extract(r"(\d+\.?\d*)")[0]
            nums = nums.dropna().astype(float)
            if len(nums) == 2:
                return nums.mean()

        
        num = pd.Series([value]).str.extract(r"(\d+\.?\d*)")[0]
        return float(num.iloc[0]) if pd.notna(num.iloc[0]) else np.nan

    df = df.applymap(extract_numeric)

    
    useful_cols = [
        "ph",
        "temperature",
        "dissolved_oxygen",
        "conductivity",
        "bod",
        "nitrate",
        "fecal_coliform",
        "total_coliform",
        "hardness",
        "solids",
        "chloramines",
        "sulfate",
        "organic_carbon",
        "trihalomethanes",
        "turbidity",
        "target"
    ]

    df = df[[col for col in useful_cols if col in df.columns]]

    
    df = df.dropna(thresh=int(0.6 * len(df.columns)))

    
    X = df.drop("target", axis=1, errors="ignore")
    y = df["target"] if "target" in df.columns else None

    
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X.columns
    )

    
    cleaned_df = X_scaled.copy()

    if y is not None:
        cleaned_df["target"] = y.reset_index(drop=True)

    
    split_index = int(0.9 * len(cleaned_df))

    train_df = cleaned_df.iloc[:split_index].reset_index(drop=True)
    test_df = cleaned_df.iloc[split_index:].reset_index(drop=True)

   
    train_path = os.path.join(output_dir, "clean_water_quality.csv")
    test_path = os.path.join(output_dir, "test_water_quality.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(" Cleaned data saved successfully!")
    print(f"Training file : {train_path} | Shape: {train_df.shape}")
    print(f"Testing file  : {test_path} | Shape: {test_df.shape}")

    return train_df, test_df, imputer, scaler

def load_dataset(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} dataset not found: {path}")

    df = pd.read_csv(path)
    df = clean_column_names(df)

    print(f" Loaded {name} dataset: {df.shape}")
    return df


def validate_dataset(df, name):
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{name} dataset missing columns: {missing}")

    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError(f"{name} target must contain at least 2 classes.")

    print(f"\n {name} Class Distribution:")
    print(df[TARGET_COLUMN].value_counts())


def split_features_target(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy().astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y


def preprocess_train_test(X_train_raw, X_test_raw):
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=FEATURE_COLUMNS
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=FEATURE_COLUMNS
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=FEATURE_COLUMNS
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=FEATURE_COLUMNS
    )

    return X_train_scaled, X_test_scaled, imputer, scaler



def build_models():
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
        force_col_wise=True,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    return rf, lgb, xgb


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name:<12} Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return acc, f1


def save_artifact(obj, filename):
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, path)
    print(f" Saved: {path}")


def train():
    train_df = load_dataset(TRAIN_FILE, "training")
    test_df = load_dataset(TEST_FILE, "testing")

    validate_dataset(train_df, "Training")
    validate_dataset(test_df, "Testing")

    X_train_raw, y_train = split_features_target(train_df)
    X_test_raw, y_test = split_features_target(test_df)

    X_train, X_test, imputer, scaler = preprocess_train_test(
        X_train_raw,
        X_test_raw
    )

    print("\n Training Separate Models...")

    rf_model, lgb_model, xgb_model = build_models()

    rf_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    print("\n Test Performance on test_water_quality.csv:")

    rf_acc, rf_f1 = evaluate_model("RF", rf_model, X_test, y_test)
    lgb_acc, lgb_f1 = evaluate_model("LightGBM", lgb_model, X_test, y_test)
    xgb_acc, xgb_f1 = evaluate_model("XGBoost", xgb_model, X_test, y_test)

    total_f1 = rf_f1 + lgb_f1 + xgb_f1

    if total_f1 == 0:
        weights = [1 / 3, 1 / 3, 1 / 3]
    else:
        weights = [
            rf_f1 / total_f1,
            lgb_f1 / total_f1,
            xgb_f1 / total_f1,
        ]

    print("\n Ensemble Weights:")
    print(f"RF  : {weights[0]:.4f}")
    print(f"LGB : {weights[1]:.4f}")
    print(f"XGB : {weights[2]:.4f}")

    print("\n Training Ensemble Model...")

    ensemble_model = VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("lgb", lgb_model),
            ("xgb", xgb_model),
        ],
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )

    ensemble_model.fit(X_train, y_train)

    ens_acc, ens_f1 = evaluate_model(
        "Ensemble",
        ensemble_model,
        X_test,
        y_test
    )

    print("\n Ensemble Classification Report:")
    print(classification_report(y_test, ensemble_model.predict(X_test), digits=4))

    print("\n Saving Artifacts...")

    save_artifact(rf_model, "rf_model.pkl")
    save_artifact(lgb_model, "lgb_model.pkl")
    save_artifact(xgb_model, "xgb_model.pkl")
    save_artifact(ensemble_model, "ensemble_model.pkl")
    save_artifact(imputer, "imputer.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(FEATURE_COLUMNS, "features.pkl")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,

        "prediction_rule_columns": PREDICTION_RULE_COLUMNS,

        "wqi_reference": {
            "ph": {
                "ideal": 7.0,
                "standard": 8.5,
                "weight": 0.11
            },
            "dissolved_oxygen": {
                "ideal": 14.6,
                "standard": 5.0,
                "weight": 0.17
            },
            "conductivity": {
                "ideal": 0.0,
                "standard": 300.0,
                "weight": 0.07
            },
            "bod": {
                "ideal": 0.0,
                "standard": 3.0,
                "weight": 0.11
            },
            "nitrate": {
                "ideal": 0.0,
                "standard": 10.0,
                "weight": 0.10
            },
            "turbidity": {
                "ideal": 0.0,
                "standard": 5.0,
                "weight": 0.08
            },
            "total_coliform": {
                "ideal": 0.0,
                "standard": 5000.0,
                "weight": 0.12
            },
            "fecal_coliform": {
                "ideal": 0.0,
                "standard": 2500.0,
                "weight": 0.14
            },
            "ph_rule_type": "range"
        },

        "contamination_rules": {
            "safe_max_wqi": 50,
            "moderate_max_wqi": 100
        },

        "disease_rules": {
            "blue_baby_nitrate_threshold": 10.0,
            "fluorosis_fluoride_threshold": 1.5,
            "bacterial_contamination_do_threshold": 5.0
        },

        "ensemble_weights": {
            "rf": float(weights[0]),
            "lgb": float(weights[1]),
            "xgb": float(weights[2])
        },

        "test_file": TEST_FILE,

        "model_metrics": {
            "rf": {
                "accuracy": float(rf_acc),
                "f1_score": float(rf_f1)
            },
            "lgb": {
                "accuracy": float(lgb_acc),
                "f1_score": float(lgb_f1)
            },
            "xgb": {
                "accuracy": float(xgb_acc),
                "f1_score": float(xgb_f1)
            },
            "ensemble": {
                "accuracy": float(ens_acc),
                "f1_score": float(ens_f1)
            }
        }
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f" Saved: {metadata_path}")
    print("\n Training and testing complete.")


if __name__ == "__main__":
    train()