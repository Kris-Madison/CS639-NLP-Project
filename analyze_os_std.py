"""
analyze_os_std.py
-----------------
Loads all results/os-std-*.jsonl files and produces six publication-ready
pyplot figures saved to final_analysis_figures/.

Plots
-----
1. Overall accuracy per model (bar chart, with ±1σ error bars from run variance)
2. Per-task success-rate heatmap  (model × task)
3. Turns-to-solve distribution    (box + swarm per model, split by outcome)
4. Run-to-run consistency         (stacked bar: both pass / mixed / both fail / error)
5. Task difficulty histogram      (distribution of per-task average accuracy)
6. Turns vs. outcome              (side-by-side mean-turns bar for success vs. fail)
"""

from pathlib import Path
import json
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR     = Path(__file__).parent / "final_analysis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Friendly display names (keep alphabetical by key)
MODEL_LABELS = {
    "claude-3":          "Claude 3",
    "claude-sonnet-3.7": "Claude Sonnet 3.7",
    "GPT-4":             "GPT-4",
    "gpt-3.5-turbo":     "GPT-3.5 Turbo",
    "gpt-4o":            "GPT-4o",
    "gpt-5-mini":        "GPT-5 Mini",
    "o3-mini":           "o3-mini",
    "o4-mini":           "o4-mini",
}

# Fixed colour palette (one colour per model)
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    records = []
    for fp in sorted(RESULTS_DIR.glob("os-std-*.jsonl")):
        with open(fp) as fh:
            for line in fh:
                r = json.loads(line)
                msgs = r["messages"]

                # Number of tool-call turns by the assistant
                num_turns = sum(
                    1 for m in msgs
                    if m["role"] == "assistant"
                    and m.get("tool_calls")
                )

                # Normalise result to numeric (error → NaN)
                raw_result = r["result"]
                if raw_result == "error":
                    result_num = np.nan
                else:
                    result_num = int(raw_result)

                records.append(
                    {
                        "model":      r["model"],
                        "task_index": int(r["task_index"]),
                        "run_number": int(r["run_number"]),
                        "result":     result_num,
                        "num_turns":  num_turns,
                    }
                )

    df = pd.DataFrame(records)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ---------------------------------------------------------------------------
# Plot 1 – Overall accuracy per model
# ---------------------------------------------------------------------------

def plot_overall_accuracy(df: pd.DataFrame):
    """Mean accuracy ± std across tasks (each task averaged over its runs)."""
    # Per-task accuracy for each model
    task_acc = (
        df.groupby(["model_label", "task_index"])["result"]
        .mean()          # mean over runs (NaN-safe)
        .reset_index()
    )

    model_stats = (
        task_acc.groupby("model_label")["result"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "std": "std"})
        .sort_values("accuracy", ascending=False)
    )

    models   = model_stats["model_label"].tolist()
    accs     = model_stats["accuracy"].tolist()
    stds     = model_stats["std"].fillna(0).tolist()
    colours  = [PALETTE[i % len(PALETTE)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, accs, color=colours, edgecolor="white", linewidth=0.8,
                  yerr=stds, capsize=5, error_kw={"elinewidth": 1.4, "ecolor": "#333"})

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{acc:.1%}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Accuracy (success rate)", fontsize=11)
    ax.set_title("Overall Accuracy per Model\n(mean ± std across tasks)", fontsize=13)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "A1_overall_accuracy.png")


# ---------------------------------------------------------------------------
# Plot 2 – Per-task success-rate heatmap
# ---------------------------------------------------------------------------

def plot_task_heatmap(df: pd.DataFrame):
    """Heatmap: rows = models, columns = tasks, value = mean result."""
    pivot = (
        df.groupby(["model_label", "task_index"])["result"]
        .mean()
        .unstack("task_index")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(20, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=7, rotation=90)
    ax.set_xlabel("Task Index", fontsize=11)
    ax.set_title("Per-Task Success Rate by Model  (green = success, red = fail)", fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Success Rate", fontsize=9)

    save(fig, "A2_task_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 3 – Turns-to-solve distribution (box plot)
# ---------------------------------------------------------------------------

def plot_turns_distribution(df: pd.DataFrame):
    """Box plots of #turns, split by outcome, one panel per model."""
    models = sorted(df["model_label"].unique())
    n = len(models)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=False)
    axes_flat = axes.flatten()

    outcome_colors = {1.0: "#55A868", 0.0: "#C44E52"}
    outcome_labels = {1.0: "Success", 0.0: "Failure"}

    for i, model in enumerate(models):
        ax = axes_flat[i]
        sub = df[df["model_label"] == model].dropna(subset=["result"])

        data   = [sub[sub["result"] == v]["num_turns"].tolist() for v in [1.0, 0.0]]
        labels = [outcome_labels[v] for v in [1.0, 0.0]]
        colors = [outcome_colors[v] for v in [1.0, 0.0]]

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.45,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_ylabel("#Turns", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Number of Turns Distribution by Model and Outcome", fontsize=14, y=1.01)
    plt.tight_layout()
    save(fig, "A3_turns_distribution.png")


# ---------------------------------------------------------------------------
# Plot 4 – Run-to-run consistency (stacked bar)
# ---------------------------------------------------------------------------

def plot_run_consistency(df: pd.DataFrame):
    """
    For each model, categorise each task into:
      Both Pass | Mixed | Both Fail | Has Error
    """
    models = sorted(df["model_label"].unique())
    categories = ["Both Pass", "Mixed", "Both Fail", "Has Error"]
    cat_colors  = ["#55A868", "#FFCC00", "#C44E52", "#8C8C8C"]

    counts = {m: {c: 0 for c in categories} for m in models}

    for (model, task), grp in df.groupby(["model_label", "task_index"]):
        results = grp["result"].tolist()
        if any(pd.isna(r) for r in results):
            counts[model]["Has Error"] += 1
        elif all(r == 1 for r in results):
            counts[model]["Both Pass"] += 1
        elif all(r == 0 for r in results):
            counts[model]["Both Fail"] += 1
        else:
            counts[model]["Mixed"] += 1

    count_df = pd.DataFrame(counts).T.reindex(columns=categories)
    total = count_df.sum(axis=1)
    pct_df = count_df.div(total, axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(models))
    model_list = sorted(models)

    for cat, color in zip(categories, cat_colors):
        vals = [pct_df.loc[m, cat] for m in model_list]
        bars = ax.bar(model_list, vals, bottom=bottom, color=color,
                      label=cat, edgecolor="white", linewidth=0.6)
        # Label segments > 8 %
        for bar, val, bot in zip(bars, vals, bottom):
            if val > 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                )
        bottom = bottom + np.array(vals)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Percentage of Tasks (%)", fontsize=11)
    ax.set_title("Run-to-Run Consistency per Model", fontsize=13)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "A4_run_consistency.png")


# ---------------------------------------------------------------------------
# Plot 5 – Task difficulty histogram
# ---------------------------------------------------------------------------

def plot_task_difficulty(df: pd.DataFrame):
    """Distribution of per-task mean accuracy (averaged over models and runs)."""
    task_acc = (
        df.groupby("task_index")["result"]
        .mean()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(task_acc, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.7, alpha=0.85)

    ax.axvline(task_acc.mean(), color="#C44E52", linewidth=2, linestyle="--",
               label=f"Mean = {task_acc.mean():.2f}")
    ax.axvline(task_acc.median(), color="#55A868", linewidth=2, linestyle="-.",
               label=f"Median = {task_acc.median():.2f}")

    ax.set_xlabel("Average Success Rate across All Models", fontsize=11)
    ax.set_ylabel("Number of Tasks", fontsize=11)
    ax.set_title("Task Difficulty Distribution\n(per-task accuracy averaged across all models & runs)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "A5_task_difficulty.png")


# ---------------------------------------------------------------------------
# Plot 6 – Mean turns for success vs. failure per model
# ---------------------------------------------------------------------------

def plot_turns_vs_outcome(df: pd.DataFrame):
    """Side-by-side bar chart: mean #turns for successes vs. failures."""
    sub = df.dropna(subset=["result"])
    stats = (
        sub.groupby(["model_label", "result"])["num_turns"]
        .mean()
        .unstack("result")
        .rename(columns={0.0: "Failure", 1.0: "Success"})
        .sort_index()
    )

    models = stats.index.tolist()
    x      = np.arange(len(models))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - width / 2, stats.get("Success", 0), width, label="Success",
                color="#55A868", edgecolor="white", linewidth=0.7)
    b2 = ax.bar(x + width / 2, stats.get("Failure", 0), width, label="Failure",
                color="#C44E52", edgecolor="white", linewidth=0.7)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Mean Number of Turns", fontsize=11)
    ax.set_title("Mean Turns to Solve: Success vs. Failure per Model", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "A6_turns_vs_outcome.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    df = load_data()
    print(f"  {len(df)} records | {df['model'].nunique()} models | "
          f"{df['task_index'].nunique()} tasks")

    print("Generating plots …")
    plot_overall_accuracy(df)
    plot_task_heatmap(df)
    plot_turns_distribution(df)
    plot_run_consistency(df)
    plot_task_difficulty(df)
    plot_turns_vs_outcome(df)

    print("Done. Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
