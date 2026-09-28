"""
Student Dropout Early-Warning System (EWS)
==========================================

Trains and evaluates scikit-learn classifiers that flag students at risk of
dropping out, using the UCI "Predict Students' Dropout and Academic Success"
dataset (Realinho et al., 2022; CC BY 4.0).

The model is evaluated at three points in a student's timeline, because an
early-warning system is only useful if it fires early:

  * enrollment  - only information known on day one (demographics, admission,
                  socio-economic, financial status)
  * semester_1  - enrollment + 1st-semester academic record
  * semester_2  - enrollment + 1st + 2nd-semester academic record

For each stage, five classifiers are compared with stratified 5-fold CV on the
training split, the best (by CV ROC-AUC) is refit, a decision threshold is tuned
on out-of-fold predictions to reach a target recall, and everything is reported
on an untouched 20 % test split. A 3-class model (Dropout / Enrolled /
Graduate) is also reported for completeness.

Usage:
    python src/dropout_ews.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET_RECALL = 0.80  # an EWS should catch at least 80 % of eventual dropouts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "students_dropout_academic_success.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

# Integer-coded nominal variables -> one-hot encoded.
CATEGORICAL = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

SEM1 = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]
SEM2 = [c.replace("1st", "2nd") for c in SEM1]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]  # one header has a stray tab
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple, interpretable ratios an advisor would also look at."""
    df = df.copy()
    for sem in ("1st", "2nd"):
        enrolled = df[f"Curricular units {sem} sem (enrolled)"]
        approved = df[f"Curricular units {sem} sem (approved)"]
        evals = df[f"Curricular units {sem} sem (evaluations)"]
        df[f"{sem} sem approval rate"] = np.where(enrolled > 0, approved / enrolled.replace(0, 1), 0.0)
        df[f"{sem} sem evaluation rate"] = np.where(enrolled > 0, evals / enrolled.replace(0, 1), 0.0)
    return df


def stage_features(df: pd.DataFrame) -> dict[str, list[str]]:
    base = [c for c in df.columns if c not in SEM1 + SEM2 + ["Target"] and "sem " not in c]
    s1 = SEM1 + ["1st sem approval rate", "1st sem evaluation rate"]
    s2 = SEM2 + ["2nd sem approval rate", "2nd sem evaluation rate"]
    return {"enrollment": base, "semester_1": base + s1, "semester_2": base + s1 + s2}


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False), cat),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=SEED,
        ),
        "SVM (RBF)": CalibratedClassifierCV(SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=SEED), ensemble=False),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def threshold_for_recall(y_true, scores, target_recall: float) -> float:
    """Highest threshold whose recall still meets the target (=> best precision)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else 0.5


def binary_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 4),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "flagged_pct": pred.mean(),
    }


def risk_band(p: float) -> str:
    return "High" if p >= 0.60 else "Medium" if p >= 0.30 else "Low"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    for d in (RESULTS, FIGURES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = add_features(load_data())
    y_multi = df["Target"]
    y = (y_multi == "Dropout").astype(int)
    print(f"Loaded {len(df)} students, {df.shape[1] - 1} columns. Dropout rate = {y.mean():.1%}")

    idx_train, idx_test = train_test_split(df.index, test_size=0.20, stratify=y_multi, random_state=SEED)
    train, test = df.loc[idx_train], df.loc[idx_test]
    y_tr, y_te = y.loc[idx_train], y.loc[idx_test]

    # ---- EDA figures ------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 4))
    y_multi.value_counts().reindex(["Dropout", "Enrolled", "Graduate"]).plot.bar(ax=ax, color=["#d1495b", "#edae49", "#00798c"])
    ax.set_title("Outcome distribution (n = 4,424)"); ax.set_ylabel("Students"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIGURES / "target_distribution.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Admission grade", "1st sem approval rate", "Age at enrollment"]):
        sns.kdeplot(data=df, x=col, hue="Target", common_norm=False, fill=True, alpha=0.3, ax=ax,
                    hue_order=["Dropout", "Enrolled", "Graduate"], palette=["#d1495b", "#edae49", "#00798c"])
        ax.set_title(col)
    fig.tight_layout(); fig.savefig(FIGURES / "key_feature_distributions.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    rates = df.groupby("Tuition fees up to date")["Target"].value_counts(normalize=True).unstack()[["Dropout", "Enrolled", "Graduate"]]
    rates.index = ["Fees NOT up to date", "Fees up to date"]
    rates.plot.barh(stacked=True, ax=ax, color=["#d1495b", "#edae49", "#00798c"])
    ax.set_xlabel("Share of students"); ax.set_title("Outcome by tuition-fee status"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "outcome_by_tuition_status.png", dpi=150); plt.close(fig)

    # ---- Binary EWS at three stages -------------------------------------- #
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    stages = stage_features(df)
    cv_rows, test_rows, final = [], [], {}

    for stage, feats in stages.items():
        print(f"\n=== Stage: {stage} ({len(feats)} features) ===")
        best_name, best_auc = None, -1.0
        for name, est in candidate_models().items():
            pipe = Pipeline([("prep", make_preprocessor(feats)), ("clf", est)])
            s = cross_validate(pipe, train[feats], y_tr, cv=cv, n_jobs=-1,
                               scoring=["roc_auc", "average_precision", "f1", "balanced_accuracy"])
            row = {"stage": stage, "model": name,
                   **{k.replace("test_", "cv_"): v.mean() for k, v in s.items() if k.startswith("test_")},
                   "cv_roc_auc_std": s["test_roc_auc"].std()}
            cv_rows.append(row)
            print(f"  {name:<24} ROC-AUC {row['cv_roc_auc']:.4f} ± {row['cv_roc_auc_std']:.4f}   PR-AUC {row['cv_average_precision']:.4f}")
            if row["cv_roc_auc"] > best_auc:
                best_name, best_auc = name, row["cv_roc_auc"]

        # Evaluate every model on test (for the comparison table) and keep the best.
        for name, est in candidate_models().items():
            pipe = Pipeline([("prep", make_preprocessor(feats)), ("clf", est)])
            oof = cross_val_predict(pipe, train[feats], y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            thr = threshold_for_recall(y_tr, oof, TARGET_RECALL)
            pipe.fit(train[feats], y_tr)
            proba = pipe.predict_proba(test[feats])[:, 1]
            m = binary_metrics(y_te, proba, thr)
            test_rows.append({"stage": stage, "model": name, "selected": name == best_name, **m})
            if name == best_name:
                final[stage] = {"model": name, "pipeline": pipe, "features": feats, "threshold": thr, "proba": proba, "metrics": m}
        print(f"  -> selected: {best_name}  | test ROC-AUC {final[stage]['metrics']['roc_auc']:.4f}, "
              f"recall {final[stage]['metrics']['recall']:.3f}, precision {final[stage]['metrics']['precision']:.3f}")

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    cv_df.round(4).to_csv(RESULTS / "cv_model_comparison.csv", index=False)
    test_df.round(4).to_csv(RESULTS / "test_model_comparison.csv", index=False)

    # ---- Figures: model comparison, ROC / PR per stage --------------------- #
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=cv_df, x="model", y="cv_roc_auc", hue="stage", ax=ax, palette="viridis")
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("CV ROC-AUC (5-fold)"); ax.set_xlabel("")
    ax.legend(title="stage", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Dropout prediction quality improves as the academic record accumulates")
    fig.tight_layout(); fig.savefig(FIGURES / "cv_roc_auc_by_stage.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for stage, f in final.items():
        label = f"{stage} – {f['model']}"
        RocCurveDisplay.from_predictions(y_te, f["proba"], name=label, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_te, f["proba"], name=label, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set_title("ROC – held-out test set")
    axes[1].axhline(y_te.mean(), color="k", ls="--", lw=0.8); axes[1].set_title("Precision–Recall – held-out test set")
    fig.tight_layout(); fig.savefig(FIGURES / "roc_pr_curves_by_stage.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (stage, f) in zip(axes, final.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_te, (f["proba"] >= f["threshold"]).astype(int), display_labels=["Not dropout", "Dropout"],
            cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{stage}\n{f['model']} (thr = {f['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrices_by_stage.png", dpi=150); plt.close(fig)

    # ---- Explainability: permutation importance on the semester-1 model --- #
    importances = {}
    for stage in ("enrollment", "semester_1"):
        f = final[stage]
        pi = permutation_importance(f["pipeline"], test[f["features"]], y_te, scoring="roc_auc",
                                    n_repeats=15, random_state=SEED, n_jobs=-1)
        imp = pd.DataFrame({"feature": f["features"], "importance_mean": pi.importances_mean,
                            "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
        imp.round(5).to_csv(RESULTS / f"permutation_importance_{stage}.csv", index=False)
        importances[stage] = imp
        top = imp.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#00798c")
        ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title(f"Top risk drivers – {stage} model ({f['model']})")
        fig.tight_layout(); fig.savefig(FIGURES / f"feature_importance_{stage}.png", dpi=150); plt.close(fig)

    # ---- Risk list for the test cohort (semester-1 model = the practical EWS) #
    ews = final["semester_1"]
    risk = test[["Course", "Age at enrollment", "Tuition fees up to date", "Scholarship holder",
                 "1st sem approval rate", "Curricular units 1st sem (grade)"]].copy()
    risk.insert(0, "student_index", test.index)
    risk["dropout_probability"] = ews["proba"].round(4)
    risk["risk_band"] = [risk_band(p) for p in ews["proba"]]
    risk["flagged"] = (ews["proba"] >= ews["threshold"]).astype(int)
    risk["actual_outcome"] = y_multi.loc[idx_test].values
    risk.sort_values("dropout_probability", ascending=False).to_csv(RESULTS / "test_risk_scores_semester_1.csv", index=False)
    band_summary = (risk.groupby("risk_band")["actual_outcome"].value_counts().unstack(fill_value=0)
                    .reindex(["High", "Medium", "Low"]))
    band_summary["n"] = band_summary.sum(axis=1)
    band_summary["actual_dropout_rate"] = (band_summary["Dropout"] / band_summary["n"]).round(3)
    band_summary.to_csv(RESULTS / "risk_band_summary.csv")

    # ---- 3-class model (Dropout / Enrolled / Graduate) ------------------- #
    feats = stages["semester_2"]
    mc = Pipeline([("prep", make_preprocessor(feats)), ("clf", clone(candidate_models()["Hist Gradient Boosting"]))])
    mc.fit(train[feats], y_multi.loc[idx_train])
    mc_pred = mc.predict(test[feats])
    mc_report = classification_report(y_multi.loc[idx_test], mc_pred, digits=4)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_multi.loc[idx_test], mc_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("3-class model (semester_2 features)")
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrix_3class.png", dpi=150); plt.close(fig)

    # ---- Persist models + summary --------------------------------------- #
    for stage, f in final.items():
        joblib.dump({"pipeline": f["pipeline"], "features": f["features"], "threshold": f["threshold"], "model": f["model"]},
                    MODELS / f"dropout_ews_{stage}.joblib", compress=3)

    summary = {
        "dataset": {"source": "UCI ML Repository id 697", "n_students": int(len(df)),
                    "dropout_rate": round(float(y.mean()), 4), "n_train": int(len(train)), "n_test": int(len(test))},
        "target_recall": TARGET_RECALL,
        "stages": {s: {"model": f["model"], "n_features": len(f["features"]),
                       **{k: round(float(v), 4) for k, v in f["metrics"].items()}} for s, f in final.items()},
        "three_class_accuracy": round(float(accuracy_score(y_multi.loc[idx_test], mc_pred)), 4),
        "three_class_macro_f1": round(float(f1_score(y_multi.loc[idx_test], mc_pred, average="macro")), 4),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    with open(RESULTS / "results.md", "w", encoding="utf-8") as fh:
        fh.write("# Results\n\n")
        fh.write(f"Held-out test set: {len(test)} students ({y_te.mean():.1%} dropouts). "
                 f"Decision thresholds tuned on out-of-fold training predictions for recall ≥ {TARGET_RECALL:.0%}.\n\n")
        fh.write("## Selected model per stage (test set)\n\n")
        sel = test_df[test_df.selected].drop(columns="selected").round(3)
        fh.write(sel.to_markdown(index=False) + "\n\n")
        fh.write("## 5-fold CV comparison (training set)\n\n")
        fh.write(cv_df.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## All models on test set\n\n")
        fh.write(test_df.drop(columns="selected").round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Risk bands (semester-1 model, test set)\n\n")
        fh.write(band_summary.to_markdown() + "\n\n")
        fh.write("## Top 10 risk drivers (semester-1 model, permutation importance)\n\n")
        fh.write(importances["semester_1"].head(10).round(4).to_markdown(index=False) + "\n\n")
        fh.write("## 3-class model (Dropout / Enrolled / Graduate, semester-2 features)\n\n```\n" + mc_report + "```\n")

    print("\n3-class report:\n" + mc_report)
    print(json.dumps(summary, indent=2))
    print("\nArtifacts written to results/ and models/")


if __name__ == "__main__":
    main()
