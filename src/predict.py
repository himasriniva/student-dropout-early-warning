"""
Score new students with a trained early-warning model.

Usage:
    python src/predict.py --stage semester_1 --input data/students_dropout_academic_success.csv --output scored.csv

The input CSV must use the same columns (and ';' separator) as the UCI dataset.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from dropout_ews import add_features, risk_band

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["enrollment", "semester_1", "semester_2"], default="semester_1")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="scored_students.csv")
    args = ap.parse_args()

    bundle = joblib.load(ROOT / "models" / f"dropout_ews_{args.stage}.joblib")
    df = pd.read_csv(args.input, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = add_features(df)

    proba = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]
    out = pd.DataFrame({
        "student_index": df.index,
        "dropout_probability": proba.round(4),
        "risk_band": [risk_band(p) for p in proba],
        "flagged": (proba >= bundle["threshold"]).astype(int),
    }).sort_values("dropout_probability", ascending=False)
    out.to_csv(args.output, index=False)
    print(f"{bundle['model']} ({args.stage}), threshold {bundle['threshold']:.3f}: "
          f"flagged {out.flagged.sum()} of {len(out)} students -> {args.output}")


if __name__ == "__main__":
    main()
