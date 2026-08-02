"""
TalentPulse — ML Layer
=========================================================
1. Feature engineering from raw longitudinal tables
2. XGBoost classifier -> attrition probability + SHAP top-3 drivers
3. KMeans clustering -> flight-risk personas
4. Cox Proportional Hazards (lifelines) -> predicted remaining tenure
5. VADER sentiment on survey / exit-interview text
6. Writes results back into attrition_predictions & sentiment_scores tables

Run: python train_models.py
"""
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "talentpulse.db"

CLUSTER_NAMES = {
    0: "Stable Performer",
    1: "Burnt-out High Performer",
    2: "Disengaged Veteran",
    3: "New & Uncertain",
}


def load_tables(conn):
    tables = ["employees", "departments", "job_roles", "managers", "salary_history",
              "performance_reviews", "engagement_surveys", "attendance", "attrition_events"]
    return {t: pd.read_sql(f"SELECT * FROM {t}", conn) for t in tables}


def build_features(t):
    emp = t["employees"].copy()

    # latest salary + hike
    sal = t["salary_history"].sort_values("effective_date")
    latest_sal = sal.groupby("employee_id").last().reset_index()[["employee_id", "monthly_salary", "hike_percent"]]
    sal_growth = sal.groupby("employee_id")["monthly_salary"].agg(["first", "last"])
    sal_growth["salary_growth_pct"] = (sal_growth["last"] - sal_growth["first"]) / sal_growth["first"].replace(0, np.nan) * 100
    sal_growth = sal_growth.reset_index()[["employee_id", "salary_growth_pct"]]

    # performance
    perf = t["performance_reviews"]
    perf_agg = perf.groupby("employee_id").agg(
        avg_rating=("rating", "mean"),
        last_rating=("rating", "last"),
        n_promotions=("promotion_flag", "sum"),
    ).reset_index()

    # engagement
    surv = t["engagement_surveys"]
    surv_agg = surv.groupby("employee_id").agg(
        avg_satisfaction=("satisfaction_score", "mean"),
        last_satisfaction=("satisfaction_score", "last"),
        avg_worklife=("worklife_balance", "mean"),
    ).reset_index()

    # attendance
    att = t["attendance"]
    att_agg = att.groupby("employee_id").agg(
        avg_absent_days=("absent_days", "mean"),
        avg_overtime=("overtime_hours", "mean"),
        total_months_recorded=("month", "count"),
    ).reset_index()

    df = emp.merge(latest_sal, on="employee_id", how="left") \
            .merge(sal_growth, on="employee_id", how="left") \
            .merge(perf_agg, on="employee_id", how="left") \
            .merge(surv_agg, on="employee_id", how="left") \
            .merge(att_agg, on="employee_id", how="left")

    df["hire_date"] = pd.to_datetime(df["hire_date"])
    today = pd.Timestamp.today()
    df["tenure_months"] = ((today - df["hire_date"]).dt.days / 30.44).round(1)

    attr = t["attrition_events"][["employee_id", "exit_date"]].copy()
    attr["attrited"] = 1
    df = df.merge(attr, on="employee_id", how="left")
    df["attrited"] = df["attrited"].fillna(0).astype(int)

    # duration for survival model: months to event (or months observed if censored)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["duration_months"] = np.where(
        df["attrited"] == 1,
        ((df["exit_date"] - df["hire_date"]).dt.days / 30.44),
        df["tenure_months"]
    )
    df["duration_months"] = df["duration_months"].clip(lower=1).round(1)

    df = df.fillna({
        "salary_growth_pct": 0, "avg_rating": 3, "last_rating": 3, "n_promotions": 0,
        "avg_satisfaction": 3, "last_satisfaction": 3, "avg_worklife": 3,
        "avg_absent_days": 0, "avg_overtime": 0,
    })
    return df


def train_classifier(df):
    feature_cols = [
        "age", "education_level", "monthly_salary", "hike_percent", "salary_growth_pct",
        "avg_rating", "last_rating", "n_promotions", "avg_satisfaction", "last_satisfaction",
        "avg_worklife", "avg_absent_days", "avg_overtime", "tenure_months",
    ]
    X = df[feature_cols].fillna(0)
    y = df["attrited"]

    # Held-out split purely for an honest, reportable evaluation metric
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    eval_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
        random_state=42,
    )
    eval_model.fit(X_train, y_train)
    test_proba = eval_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)
    print(f"  Held-out test AUC: {auc:.3f}")
    print(classification_report(y_test, (test_proba > 0.5).astype(int), target_names=["Stayed", "Left"]))

    # Final production model refit on all data (for scoring the whole current workforce)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    proba = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    top_drivers = []
    for i in range(len(X)):
        row_shap = shap_values[i]
        order = np.argsort(-np.abs(row_shap))[:3]
        drivers = [feature_cols[j] for j in order]
        top_drivers.append(drivers)

    return proba, top_drivers, feature_cols, model


def risk_band(p):
    if p < 0.25:
        return "Low"
    elif p < 0.5:
        return "Medium"
    elif p < 0.75:
        return "High"
    return "Critical"


def train_clusters(df):
    feats = df[["avg_satisfaction", "avg_worklife", "avg_overtime", "tenure_months", "avg_rating"]].fillna(0)
    scaled = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)

    # map cluster index -> human name based on centroid characteristics (heuristic, order-independent)
    centroids = pd.DataFrame(km.cluster_centers_, columns=feats.columns)
    name_map = {}
    remaining = list(CLUSTER_NAMES.values())
    for idx in centroids.sort_values("avg_satisfaction").index:
        name_map[idx] = remaining.pop(0) if remaining else "Other"
    return [name_map.get(l, "Other") for l in labels]


def train_survival(df):
    cph = CoxPHFitter()
    cols = ["duration_months", "attrited", "age", "avg_satisfaction", "avg_overtime",
            "avg_rating", "salary_growth_pct", "tenure_months"]
    surv_df = df[cols].dropna()
    cph.fit(surv_df, duration_col="duration_months", event_col="attrited", show_progress=False)

    pred_median = cph.predict_median(df[cols].fillna(0))
    pred_median = pred_median.replace([np.inf, -np.inf], np.nan)
    pred_median = pred_median.fillna(df["tenure_months"] + 24)
    return pred_median.clip(lower=1).round(0).astype(int)


def run_sentiment(t):
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for _, r in t["engagement_surveys"].iterrows():
        score = analyzer.polarity_scores(str(r["comment_text"]))["compound"]
        label = "Positive" if score > 0.2 else "Negative" if score < -0.2 else "Neutral"
        rows.append({"employee_id": r["employee_id"], "source_table": "engagement_surveys",
                      "source_id": r["survey_id"], "sentiment_label": label, "sentiment_score": score})
    for _, r in t["attrition_events"].iterrows():
        score = analyzer.polarity_scores(str(r["exit_interview_text"]))["compound"]
        label = "Positive" if score > 0.2 else "Negative" if score < -0.2 else "Neutral"
        rows.append({"employee_id": r["employee_id"], "source_table": "attrition_events",
                      "source_id": r["event_id"], "sentiment_label": label, "sentiment_score": score})
    return pd.DataFrame(rows)


def main():
    conn = sqlite3.connect(DB_PATH)
    t = load_tables(conn)
    df = build_features(t)

    print("Training XGBoost classifier + SHAP explainability...")
    proba, top_drivers, feature_cols, model = train_classifier(df)

    print("Training KMeans flight-risk segmentation...")
    cluster_labels = train_clusters(df)

    print("Fitting Cox Proportional Hazards survival model...")
    predicted_tenure = train_survival(df)

    print("Running VADER sentiment on surveys + exit interviews...")
    sentiment_df = run_sentiment(t)

    preds = pd.DataFrame({
        "employee_id": df["employee_id"],
        "risk_score": np.round(proba, 4),
        "top_driver_1": [d[0] for d in top_drivers],
        "top_driver_2": [d[1] for d in top_drivers],
        "top_driver_3": [d[2] for d in top_drivers],
        "predicted_tenure_months": predicted_tenure.values,
        "cluster_label": cluster_labels,
    })
    preds["risk_band"] = preds["risk_score"].apply(risk_band)
    preds["model_run_date"] = datetime.now().isoformat()

    preds.to_sql("attrition_predictions", conn, if_exists="replace", index=False)
    sentiment_df.to_sql("sentiment_scores", conn, if_exists="replace", index=False)

    # persist model + feature list + feature dataframe for the Streamlit what-if simulator
    import joblib
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, models_dir / "xgb_attrition_model.pkl")
    df.to_pickle(models_dir / "feature_frame.pkl")

    conn.commit()
    conn.close()

    print("\nDone. Sample of predictions:")
    print(preds.sort_values("risk_score", ascending=False).head(10).to_string(index=False))
    print(f"\nAttrition base rate: {df['attrited'].mean():.1%}")
    print(f"Risk band distribution:\n{preds['risk_band'].value_counts()}")


if __name__ == "__main__":
    main()
