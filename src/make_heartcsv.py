from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

OUT = Path("data/heart.csv")

def main():
    heart = fetch_ucirepo(id=45)  # UCI Heart Disease
    X = heart.data.features.copy()
    y = heart.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    # Normalize column names: lower case, underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Some sources ship 'num' as target (0..4). Create binary target = (num>0)
    if "target" not in df.columns and "num" in df.columns:
        df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype("Int64")

    # Coerce '?' to NaN and cast common categoricals to numeric
    df = df.replace("?", pd.NA)
    for c in ["sex","cp","fbs","restecg","exang","slope","ca","thal","num","target"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If ucimlrepo returned multiple hospitals (cleveland/hungary/long-beach/switzerland)
    # keep Cleveland to match your proposal (303 rows)
    for candidate in ["dataset", "source", "hospital"]:
        if candidate in df.columns:
            cle = df[df[candidate].astype(str).str.contains("cleveland", case=False, na=False)]
            if len(cle) >= 300:
                df = cle
            break

    # Keep the columns
    keep = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target","num"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT.resolve()} with shape {df.shape}")

if __name__ == "__main__":
    main()
