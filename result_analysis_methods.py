from pathlib import Path
import math
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(r"/Users/jinlinxiang/Desktop/未命名文件夹 2")
OUTPUT_DIR = DATA_DIR / "final_analysis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_LABELS = [
    "Environment Disturbance / Unable to Detect Change",
    "Instruction Error",
    "Catastrophic Forgetting",
    "False Assumptions",
    "Planning Errors (Sub-plan & Action)",
    "History Error Accumulation",
    "Memory Limitations",
]

def infer_model_name(file_path: Path) -> str:
    name = file_path.stem
    name = re.sub(r"^failure_classifications[-_]*", "", name, flags=re.IGNORECASE)
    return name.strip("_- ")

def load_all_csvs(data_dir: Path):
    csv_files = sorted(data_dir.glob("failure_classifications*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching 'failure_classifications*.csv' were found in: {data_dir}"
        )

    dfs = []

    for fp in csv_files:
        df = pd.read_csv(fp)
        expected_cols = {"task_index", "classification_type"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"{fp.name} is missing required columns. Found: {list(df.columns)}")

        df = df[["task_index", "classification_type"]].copy()
        df["task_index"] = pd.to_numeric(df["task_index"], errors="raise").astype(int)
        df["classification_type"] = (
            df["classification_type"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        df["model"] = fp.stem
        df["model_display"] = infer_model_name(fp)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined, csv_files

def make_tables(df_long: pd.DataFrame):
    model_display_map = (
        df_long[["model", "model_display"]]
        .drop_duplicates()
        .set_index("model")["model_display"]
        .to_dict()
    )

    counts = (
        df_long.groupby(["model", "classification_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=VALID_LABELS, fill_value=0)
    )

    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    # 图表和打印时都显示简洁模型名
    counts.index = [model_display_map[m] for m in counts.index]
    pct.index = [model_display_map[m] for m in pct.index]

    pivot = (
        df_long.pivot_table(
            index="task_index",
            columns="model",
            values="classification_type",
            aggfunc="first"
        )
        .sort_index()
    )

    return counts, pct, pivot

def compute_task_agreement(pivot: pd.DataFrame):
    rows = []

    for task_idx, row in pivot.iterrows():
        votes = row.dropna()
        total_votes = len(votes)

        if total_votes == 0:
            continue

        vc = votes.value_counts()
        majority_label = vc.index[0]
        majority_count = int(vc.iloc[0])
        agreement_ratio = majority_count / total_votes

        probs = vc / total_votes
        entropy = -sum(float(p) * math.log2(float(p)) for p in probs if p > 0)

        rows.append({
            "task_index": int(task_idx),
            "num_models": total_votes,
            "num_unique_labels": int(vc.size),
            "majority_label": majority_label,
            "majority_count": majority_count,
            "agreement_ratio": agreement_ratio,
            "vote_entropy": entropy,
            "vote_detail": "; ".join([f"{k}: {v}" for k, v in vc.items()]),
        })

    return pd.DataFrame(rows).sort_values(
        by=["agreement_ratio", "vote_entropy", "task_index"],
        ascending=[True, False, True]
    )

def pairwise_agreement_matrix(pivot: pd.DataFrame):
    models = list(pivot.columns)
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                matrix.loc[m1, m2] = 1.0
                continue

            s1 = pivot[m1]
            s2 = pivot[m2]
            valid = s1.notna() & s2.notna()

            if valid.sum() == 0:
                matrix.loc[m1, m2] = np.nan
            else:
                agreement = (s1[valid] == s2[valid]).mean()
                matrix.loc[m1, m2] = float(agreement)

    return matrix

def cohens_kappa(series_a: pd.Series, series_b: pd.Series, labels):
    sub = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    if len(sub) == 0:
        return np.nan

    conf = pd.crosstab(sub["a"], sub["b"]).reindex(index=labels, columns=labels, fill_value=0)
    n = conf.to_numpy().sum()
    if n == 0:
        return np.nan

    po = np.trace(conf.to_numpy()) / n
    row_marg = conf.sum(axis=1).to_numpy() / n
    col_marg = conf.sum(axis=0).to_numpy() / n
    pe = np.sum(row_marg * col_marg)

    if abs(1 - pe) < 1e-12:
        return np.nan

    return (po - pe) / (1 - pe)

def pairwise_kappa_matrix(pivot: pd.DataFrame):
    models = list(pivot.columns)
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                matrix.loc[m1, m2] = 1.0
                continue

            matrix.loc[m1, m2] = cohens_kappa(pivot[m1], pivot[m2], VALID_LABELS)

    return matrix

def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def save_stacked_count_plot(counts: pd.DataFrame, path: Path):
    ax = counts.plot(kind="bar", stacked=True, figsize=(14, 7))
    ax.set_title("Failure Type Distribution by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")
    ax.legend(title="Classification Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_stacked_pct_plot(pct: pd.DataFrame, path: Path):
    ax = pct.plot(kind="bar", stacked=True, figsize=(14, 7))
    ax.set_title("Failure Type Percentage by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Classification Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_heatmap(matrix: pd.DataFrame, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix.values.astype(float), aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            txt = "" if pd.isna(val) else f"{float(val):.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_agreement_hist(task_summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(task_summary["agreement_ratio"].dropna(), bins=10)
    ax.set_title("Distribution of Task-Level Agreement Ratios")
    ax.set_xlabel("Agreement Ratio")
    ax.set_ylabel("Number of Tasks")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_top_disagreement_plot(task_summary: pd.DataFrame, path: Path, top_n: int = 15):
    top = task_summary.nsmallest(top_n, "agreement_ratio").sort_values(
        ["agreement_ratio", "vote_entropy", "task_index"]
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["task_index"].astype(str), top["agreement_ratio"])
    ax.set_title(f"Top {len(top)} Most Disagreed Tasks")
    ax.set_xlabel("Agreement Ratio")
    ax.set_ylabel("Task Index")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    try:
        df_long, csv_files = load_all_csvs(DATA_DIR)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    counts, pct, pivot = make_tables(df_long)
    task_summary = compute_task_agreement(pivot)
    pairwise_agree = pairwise_agreement_matrix(pivot)
    pairwise_kappa = pairwise_kappa_matrix(pivot)

    unknown_labels = sorted(set(df_long["classification_type"]) - set(VALID_LABELS))

    fig_paths = [
        OUTPUT_DIR / "01_failure_type_distribution_counts.png",
        OUTPUT_DIR / "02_failure_type_distribution_percentages.png",
        OUTPUT_DIR / "03_pairwise_agreement_heatmap.png",
        OUTPUT_DIR / "04_pairwise_kappa_heatmap.png",
        OUTPUT_DIR / "05_task_agreement_histogram.png",
        OUTPUT_DIR / "06_top_disagreement_tasks.png",
    ]

    save_stacked_count_plot(counts, fig_paths[0])
    save_stacked_pct_plot(pct, fig_paths[1])
    save_heatmap(pairwise_agree, "Pairwise Exact Agreement Between Models", fig_paths[2])
    save_heatmap(pairwise_kappa, "Pairwise Cohen's Kappa Between Models", fig_paths[3])
    save_agreement_hist(task_summary, fig_paths[4])
    save_top_disagreement_plot(task_summary, fig_paths[5], top_n=min(15, len(task_summary)))

    total_tasks = pivot.shape[0]
    total_models = pivot.shape[1]
    mean_agreement = task_summary["agreement_ratio"].mean()
    median_agreement = task_summary["agreement_ratio"].median()
    unanimous_count = int((task_summary["agreement_ratio"] == 1.0).sum())
    disagreement_count = int((task_summary["agreement_ratio"] < 0.5).sum())

    majority_dist = task_summary["majority_label"].value_counts().reindex(VALID_LABELS, fill_value=0)
    model_top_label = counts.idxmax(axis=1)
    model_top_label_ratio = pct.max(axis=1)

    pairwise_no_diag = pairwise_agree.where(~np.eye(pairwise_agree.shape[0], dtype=bool))
    best_pair = pairwise_no_diag.stack().sort_values(ascending=False).head(1)
    worst_pair = pairwise_no_diag.stack().sort_values(ascending=True).head(1)

    kappa_no_diag = pairwise_kappa.where(~np.eye(pairwise_kappa.shape[0], dtype=bool))
    best_kappa_pair = kappa_no_diag.stack().sort_values(ascending=False).head(1)
    worst_kappa_pair = kappa_no_diag.stack().sort_values(ascending=True).head(1)

    print_section("INPUT OVERVIEW")
    print(f"Data folder: {DATA_DIR}")
    print(f"Figure output folder: {OUTPUT_DIR}")
    print(f"CSV files found: {len(csv_files)}")
    print("Files:")
    for fp in csv_files:
        print(f" - {fp.name}")
    print("Models:", ", ".join(counts.index.tolist()))
    print(f"Unique tasks: {total_tasks}")
    print(f"Total model-task judgments: {len(df_long)}")

    if unknown_labels:
        print("Warning: found labels outside the closed set:")
        for x in unknown_labels:
            print(" -", x)

    print_section("PER-MODEL LABEL COUNTS")
    print(counts.to_string())

    print_section("PER-MODEL LABEL PERCENTAGES (%)")
    print(pct.round(2).to_string())

    print_section("GLOBAL SUMMARY")
    print(f"Mean task-level agreement ratio:   {mean_agreement:.3f}")
    print(f"Median task-level agreement ratio: {median_agreement:.3f}")
    print(f"Unanimous tasks:                   {unanimous_count}")
    print(f"Tasks with agreement < 0.5:        {disagreement_count}")

    print("\nMajority label distribution across tasks:")
    for label, val in majority_dist.items():
        print(f" - {label}: {int(val)}")

    print("\nEach model's most frequent label:")
    for model in counts.index:
        print(f" - {model}: {model_top_label[model]} ({model_top_label_ratio[model]:.2f}%)")

    print_section("PAIRWISE EXACT AGREEMENT MATRIX")
    print(pairwise_agree.round(3).to_string())

    print_section("PAIRWISE COHEN'S KAPPA MATRIX")
    print(pairwise_kappa.round(3).to_string())

    if len(best_pair) > 0:
        (m1, m2), val = best_pair.index[0], best_pair.iloc[0]
        print("\nHighest exact agreement pair:")
        print(f" - {m1} vs {m2}: {val:.3f}")

    if len(worst_pair) > 0:
        (m1, m2), val = worst_pair.index[0], worst_pair.iloc[0]
        print("Lowest exact agreement pair:")
        print(f" - {m1} vs {m2}: {val:.3f}")

    if len(best_kappa_pair) > 0:
        (m1, m2), val = best_kappa_pair.index[0], best_kappa_pair.iloc[0]
        print("Highest kappa pair:")
        print(f" - {m1} vs {m2}: {val:.3f}")

    if len(worst_kappa_pair) > 0:
        (m1, m2), val = worst_kappa_pair.index[0], worst_kappa_pair.iloc[0]
        print("Lowest kappa pair:")
        print(f" - {m1} vs {m2}: {val:.3f}")

    print_section("TOP 15 MOST DISAGREED TASKS")
    cols = ["task_index", "agreement_ratio", "num_unique_labels", "majority_label", "vote_detail"]
    print(task_summary[cols].head(15).to_string(index=False))

    print_section("FIGURES GENERATED")
    for p in fig_paths:
        print(p)

if __name__ == "__main__":
    main()