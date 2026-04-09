import argparse, os, pickle
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--name", required=True)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--drop_cols", default="date")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--train_only_normal", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    for c in [x.strip() for x in args.drop_cols.split(",") if x.strip()]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if args.label_col not in df.columns:
        raise ValueError(f"label col '{args.label_col}' not found")

    y = df[args.label_col].astype(int).to_numpy()
    X = (
        df.drop(columns=[args.label_col])
        .select_dtypes(include=[np.number])
        .to_numpy(dtype=np.float32)
    )

    T, D = X.shape
    split = int(T * args.train_ratio)

    X_train = X[:split]
    X_test = X[split:]
    y_test = y[split:]

    if args.train_only_normal:
        X_train = X_train[y[:split] == 0]

    with open(os.path.join(args.out_dir, f"{args.name}_train.pkl"), "wb") as f:
        pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.out_dir, f"{args.name}_test.pkl"), "wb") as f:
        pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.out_dir, f"{args.name}_test_label.pkl"), "wb") as f:
        pickle.dump(y_test, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] {args.name}: X_train={X_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}, D={D}")

if __name__ == "__main__":
    main()
