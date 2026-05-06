"""
analyze_os_std_extended.py
15 publication-ready plots from os-std-*.jsonl evaluation logs.

Group 1 — Basic Performance       (G1, G2)
Group 2 — Run Consistency          (G3, G4)
Group 3 — Conversation Behavior    (G5, G6, G7)
Group 4 — Tool Usage Patterns      (G8, G9, G10)
Group 5 — Task Difficulty          (G11, G12, G13)
Group 6 — Temporal & Meta          (G14, G15)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR     = Path(__file__).parent / "final_analysis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

ALL_TOOLS = ["bash_action", "finish_action", "answer_action"]
TOOL_COLORS = {
    "bash_action":   "#4C72B0",
    "finish_action": "#C44E52",
    "answer_action": "#55A868",
    "none":          "#BBBBBB",
}

PCT_FMT = mticker.FuncFormatter(lambda y, _: f"{y:.0%}")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    records = []
    for fp in sorted(RESULTS_DIR.glob("os-std-*.jsonl")):
        with open(fp) as fh:
            for line in fh:
                r = json.loads(line)
                msgs = r["messages"]

                tool_seq = []
                for m in msgs:
                    if m["role"] == "assistant" and m.get("tool_calls"):
                        for tc in m["tool_calls"]:
                            tool_seq.append(tc["function"]["name"])

                raw = r["result"]
                result_num = np.nan if raw == "error" else int(raw)

                records.append({
                    "model":           r["model"],
                    "task_index":      int(r["task_index"]),
                    "run_number":      int(r["run_number"]),
                    "result":          result_num,
                    "num_messages":    len(msgs),
                    "num_turns":       len(tool_seq),
                    "tool_seq":        tool_seq,
                    "terminal_action": tool_seq[-1] if tool_seq else "none",
                    "timestamp":       r.get("timestamp", np.nan),
                    **{f"n_{t}": tool_seq.count(t) for t in ALL_TOOLS},
                })

    df = pd.DataFrame(records)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df


def build_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_label, task_idx), grp in df.groupby(["model_label", "task_index"]):
        results = grp["result"].tolist()
        if any(pd.isna(r) for r in results):
            cat = "has_error"
        elif all(r == 1 for r in results):
            cat = "both_pass"
        elif all(r == 0 for r in results):
            cat = "both_fail"
        else:
            cat = "inconsistent"
        rows.append({"model_label": model_label, "task_index": task_idx, "consistency": cat})
    return pd.DataFrame(rows)


def model_palette(models) -> dict:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(set(models)))}


def save(fig: plt.Figure, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ===========================================================================
# GROUP 1 — Basic Performance
# ===========================================================================

def plot_G1_overall_accuracy(df: pd.DataFrame):
    task_acc = df.groupby(["model_label", "task_index"])["result"].mean().reset_index()
    stats = (
        task_acc.groupby("model_label")["result"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "acc", "std": "sd"})
        .sort_values("acc", ascending=False)
    )

    mc = model_palette(stats["model_label"])
    models = stats["model_label"].tolist()
    accs   = stats["acc"].tolist()
    stds   = stats["sd"].fillna(0).tolist()
    colors = [mc[m] for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(models, accs, color=colors, edgecolor="white", linewidth=0.8,
                  yerr=stds, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.013,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(PCT_FMT)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("G1 — Overall Success Rate per Model\n(mean ± std across tasks)", fontsize=13)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G1_overall_accuracy.png")


def plot_G2_task_heatmap(df: pd.DataFrame):
    pivot = (
        df.groupby(["model_label", "task_index"])["result"]
        .mean().unstack("task_index").sort_index()
    )

    fig, ax = plt.subplots(figsize=(22, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    tick_step = 5
    ax.set_xticks(range(0, len(pivot.columns), tick_step))
    ax.set_xticklabels(pivot.columns.tolist()[::tick_step], fontsize=8, rotation=90)
    ax.set_xlabel("Task Index", fontsize=11)
    ax.set_title("G2 — Success Rate Heatmap (Model × Task)   [green=pass, red=fail]", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Success Rate")
    save(fig, "G2_task_heatmap.png")


# ===========================================================================
# GROUP 2 — Run Consistency
# ===========================================================================

def plot_G3_run_consistency(df: pd.DataFrame):
    cons_df = build_consistency(df)
    cats        = ["both_pass", "inconsistent", "both_fail", "has_error"]
    cat_labels  = ["Both Pass", "Inconsistent", "Both Fail", "Has Error"]
    cat_colors  = ["#55A868",  "#FFCC00",      "#C44E52",   "#8C8C8C"]

    counts = (
        cons_df.groupby(["model_label", "consistency"])
        .size().unstack(fill_value=0)
        .reindex(columns=cats, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    models = sorted(pct.index)

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(models))
    for cat, label, color in zip(cats, cat_labels, cat_colors):
        vals = [pct.loc[m, cat] if m in pct.index else 0 for m in models]
        bars = ax.bar(models, vals, bottom=bottom, color=color, label=label,
                      edgecolor="white", linewidth=0.6)
        for bar, val, bot in zip(bars, vals, bottom):
            if val > 8:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_ylim(0, 108)
    ax.set_ylabel("% of Tasks", fontsize=11)
    ax.set_title("G3 — Run Consistency per Model  (2 Runs per Task)", fontsize=13)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G3_run_consistency.png")


def plot_G4_inconsistency_scatter(df: pd.DataFrame):
    cons_df = build_consistency(df)
    rows = []
    for model, grp in cons_df.groupby("model_label"):
        total  = len(grp)
        incons = (grp["consistency"] == "inconsistent").sum() / total
        sr     = df[df["model_label"] == model]["result"].mean()
        rows.append({"model_label": model, "success_rate": sr, "inconsistency_rate": incons})
    ms = pd.DataFrame(rows)
    mc = model_palette(ms["model_label"])

    mean_sr = ms["success_rate"].mean()
    mean_ir = ms["inconsistency_rate"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(mean_sr, color="gray", linestyle="--", alpha=0.45, linewidth=1)
    ax.axhline(mean_ir, color="gray", linestyle="--", alpha=0.45, linewidth=1)

    # Quadrant annotations
    xmax = ms["success_rate"].max()
    ymax = ms["inconsistency_rate"].max()
    ymin = ms["inconsistency_rate"].min()
    xmin = ms["success_rate"].min()
    ax.text(mean_sr + 0.005, ymax * 0.97, "High success\nHigh inconsistency\n(\"Lucky\")",
            fontsize=8, color="#AA7700", ha="left", va="top")
    ax.text(mean_sr + 0.005, ymin + 0.002, "High success\nLow inconsistency\n(Reliable)",
            fontsize=8, color="#227733", ha="left", va="bottom")
    ax.text(mean_sr - 0.005, ymax * 0.97, "Low success\nHigh inconsistency",
            fontsize=8, color="#C44E52", ha="right", va="top")

    for _, row in ms.iterrows():
        ax.scatter(row["success_rate"], row["inconsistency_rate"],
                   color=mc[row["model_label"]], s=160, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(row["model_label"], (row["success_rate"], row["inconsistency_rate"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.xaxis.set_major_formatter(PCT_FMT)
    ax.yaxis.set_major_formatter(PCT_FMT)
    ax.set_xlabel("Overall Success Rate", fontsize=12)
    ax.set_ylabel("Inconsistency Rate (mixed-outcome tasks)", fontsize=12)
    ax.set_title("G4 — Inconsistency Rate vs. Success Rate per Model", fontsize=13)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G4_inconsistency_scatter.png")


# ===========================================================================
# GROUP 3 — Conversation Behavior
# ===========================================================================

def plot_G5_message_distribution(df: pd.DataFrame):
    models = sorted(df["model_label"].unique())
    mc = model_palette(models)

    data = [df[df["model_label"] == m]["num_messages"].dropna().tolist() for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(data, positions=range(len(models)),
                          showmedians=True, showextrema=True)
    for pc, m in zip(parts["bodies"], models):
        pc.set_facecolor(mc[m])
        pc.set_alpha(0.72)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Total Messages in Conversation", fontsize=11)
    ax.set_title("G5 — Distribution of Conversation Length per Model", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G5_message_distribution.png")


def plot_G6_length_by_result(df: pd.DataFrame):
    models = sorted(df["model_label"].unique())
    sub    = df.dropna(subset=["result"])
    ncols  = 4
    nrows  = (len(models) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharey=False)
    axes_flat  = axes.flatten()

    outcome_specs = [(1, "Success", "#55A868"), (0, "Failure", "#C44E52")]

    for i, model in enumerate(models):
        ax    = axes_flat[i]
        m_sub = sub[sub["model_label"] == model]

        valid_data, valid_pos, valid_colors, valid_labels = [], [], [], []
        for pos, (val, label, color) in enumerate(outcome_specs, start=1):
            data = m_sub[m_sub["result"] == val]["num_messages"].tolist()
            if len(data) >= 2:
                valid_data.append(data)
                valid_pos.append(pos)
                valid_colors.append(color)
                valid_labels.append(label)

        if valid_data:
            parts = ax.violinplot(valid_data, positions=valid_pos, showmedians=True)
            for pc, color in zip(parts["bodies"], valid_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.72)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Success", "Failure"], fontsize=9)
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_ylabel("#Messages", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("G6 — Conversation Length by Outcome per Model  (Success vs. Failure)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save(fig, "G6_length_by_result.png")


def plot_G7_message_heatmap(df: pd.DataFrame):
    pivot = (
        df.groupby(["model_label", "task_index"])["num_messages"]
        .mean().unstack("task_index").sort_index()
    )

    fig, ax = plt.subplots(figsize=(22, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=pivot.values.max())
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    tick_step = 5
    ax.set_xticks(range(0, len(pivot.columns), tick_step))
    ax.set_xticklabels(pivot.columns.tolist()[::tick_step], fontsize=8, rotation=90)
    ax.set_xlabel("Task Index", fontsize=11)
    ax.set_title("G7 — Average Conversation Length Heatmap (Model × Task)", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Avg # Messages")
    save(fig, "G7_message_heatmap.png")


# ===========================================================================
# GROUP 4 — Tool Usage Patterns
# ===========================================================================

def plot_G8_tool_frequency(df: pd.DataFrame):
    models = sorted(df["model_label"].unique())
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, tool in enumerate(ALL_TOOLS):
        vals = []
        for m in models:
            sub   = df[df["model_label"] == m]
            total = sub[[f"n_{t}" for t in ALL_TOOLS]].sum().sum()
            vals.append(sub[f"n_{tool}"].sum() / total * 100 if total > 0 else 0)
        ax.bar(x + i * width, vals, width, label=tool,
               color=TOOL_COLORS[tool], edgecolor="white", linewidth=0.6, alpha=0.88)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("% of Total Tool Calls", fontsize=11)
    ax.set_title("G8 — Tool Call Frequency Distribution per Model", fontsize=13)
    ax.legend(title="Tool", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G8_tool_frequency.png")


def plot_G9_tool_transitions(df: pd.DataFrame):
    models = sorted(df["model_label"].unique())
    t2i    = {t: j for j, t in enumerate(ALL_TOOLS)}
    ncols  = 4
    nrows  = (len(models) + ncols - 1) // ncols
    short  = [t.replace("_action", "") for t in ALL_TOOLS]

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes_flat  = axes.flatten()

    for i, model in enumerate(models):
        ax  = axes_flat[i]
        sub = df[df["model_label"] == model]

        matrix = np.zeros((len(ALL_TOOLS), len(ALL_TOOLS)))
        for _, row in sub.iterrows():
            seq = row["tool_seq"]
            for a, b in zip(seq[:-1], seq[1:]):
                if a in t2i and b in t2i:
                    matrix[t2i[a], t2i[b]] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix /= row_sums

        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(ALL_TOOLS)))
        ax.set_yticks(range(len(ALL_TOOLS)))
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title(model, fontsize=9, fontweight="bold")
        ax.set_xlabel("→ next", fontsize=7)
        ax.set_ylabel("current ↓", fontsize=7)

        for r in range(len(ALL_TOOLS)):
            for c in range(len(ALL_TOOLS)):
                v = matrix[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.55 else "black")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("G9 — Tool Transition Matrix per Model\n"
                 "(row = current tool → column = next tool, row-normalised)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save(fig, "G9_tool_transitions.png")


def plot_G10_terminal_action(df: pd.DataFrame):
    models     = sorted(df["model_label"].unique())
    sub        = df.dropna(subset=["result"])
    term_tools = ALL_TOOLS + ["none"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (outcome, olabel) in zip(axes, [(1, "Success"), (0, "Failure")]):
        out_sub = sub[sub["result"] == outcome]
        bottom  = np.zeros(len(models))

        for tool in term_tools:
            color = TOOL_COLORS.get(tool, "#BBBBBB")
            vals  = []
            for m in models:
                m_sub = out_sub[out_sub["model_label"] == m]
                total = len(m_sub)
                vals.append(m_sub["terminal_action"].eq(tool).sum() / total * 100
                            if total > 0 else 0)
            bars = ax.bar(models, vals, bottom=bottom, color=color, label=tool,
                          edgecolor="white", linewidth=0.5)
            for bar, val, bot in zip(bars, vals, bottom):
                if val > 10:
                    ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                            f"{val:.0f}%", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += np.array(vals)

        ax.set_title(f"Outcome: {olabel}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("% of Episodes", fontsize=10)
        ax.legend(title="Terminal Tool", fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("G10 — Terminal Tool Call by Outcome per Model\n"
                 "(finish_action on success = overcautious; answer_action on failure = wrong answer)",
                 fontsize=12)
    plt.tight_layout()
    save(fig, "G10_terminal_action.png")


# ===========================================================================
# GROUP 5 — Task Difficulty & Clustering
# ===========================================================================

def plot_G11_task_difficulty(df: pd.DataFrame):
    task_acc = (
        df.groupby("task_index")["result"]
        .mean().dropna().sort_values(ascending=False)
        .reset_index()
    )
    task_acc.columns = ["task_index", "success_rate"]

    colors = [
        "#55A868" if sr > 0.7 else ("#FFCC00" if sr > 0.3 else "#C44E52")
        for sr in task_acc["success_rate"]
    ]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(len(task_acc)), task_acc["success_rate"],
           color=colors, edgecolor="none", width=1.0)
    ax.axhline(0.7, color="#55A868", linestyle="--", linewidth=1.5, label="Easy  > 70%")
    ax.axhline(0.3, color="#C44E52", linestyle="--", linewidth=1.5, label="Hard  < 30%")
    ax.set_xticks([])
    ax.set_xlabel("Tasks sorted by average success rate →", fontsize=11)
    ax.set_ylabel("Avg Success Rate (all models & runs)", fontsize=11)
    ax.set_title("G11 — Task Difficulty Ranking", fontsize=13)
    ax.legend(fontsize=10)

    easy = (task_acc["success_rate"] > 0.7).sum()
    med  = ((task_acc["success_rate"] >= 0.3) & (task_acc["success_rate"] <= 0.7)).sum()
    hard = (task_acc["success_rate"] < 0.3).sum()
    ax.text(0.02, 0.94, f"Easy: {easy}   Medium: {med}   Hard: {hard}",
            transform=ax.transAxes, fontsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G11_task_difficulty.png")


def plot_G12_task_correlation(df: pd.DataFrame):
    pivot = (
        df.groupby(["task_index", "model_label"])["result"]
        .mean().unstack("model_label")
    )
    pivot = pivot.dropna(how="all").fillna(pivot.mean())
    corr  = pivot.T.corr()   # tasks × tasks

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    step = 10
    ax.set_xticks(range(0, len(corr.columns), step))
    ax.set_xticklabels(corr.columns.tolist()[::step], rotation=90, fontsize=8)
    ax.set_yticks(range(0, len(corr.index), step))
    ax.set_yticklabels(corr.index.tolist()[::step], fontsize=8)
    ax.set_xlabel("Task Index", fontsize=11)
    ax.set_ylabel("Task Index", fontsize=11)
    ax.set_title("G12 — Task Correlated Failure Heatmap\n"
                 "(red = fail together across models, blue = opposite pattern)", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Pearson r (success rates)")
    save(fig, "G12_task_correlation.png")


def plot_G13_difficulty_tier_performance(df: pd.DataFrame):
    task_acc = df.groupby("task_index")["result"].mean().dropna()

    def tier(sr):
        if sr > 0.7:   return "Easy (>70%)"
        elif sr > 0.3: return "Medium (30–70%)"
        else:          return "Hard (<30%)"

    task_tier = task_acc.apply(tier).rename("tier").reset_index()
    df2 = df.merge(task_tier, on="task_index", how="left")

    tiers       = ["Easy (>70%)", "Medium (30–70%)", "Hard (<30%)"]
    tier_colors = ["#55A868",     "#FFCC00",          "#C44E52"]
    models      = sorted(df["model_label"].unique())
    x           = np.arange(len(models))
    width       = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (tlabel, tcolor) in enumerate(zip(tiers, tier_colors)):
        vals = [
            df2[(df2["model_label"] == m) & (df2["tier"] == tlabel)]["result"].mean()
            for m in models
        ]
        vals = [0 if np.isnan(v) else v for v in vals]
        ax.bar(x + i * width, vals, width, label=tlabel, color=tcolor,
               edgecolor="white", linewidth=0.6, alpha=0.88)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(PCT_FMT)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_title("G13 — Model Performance by Task Difficulty Tier", fontsize=13)
    ax.legend(title="Task Tier", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G13_difficulty_tier.png")


# ===========================================================================
# GROUP 6 — Temporal & Meta Signals
# ===========================================================================

def plot_G14_latency_proxy(df: pd.DataFrame):
    """Median inter-episode timestamp gap per model as a latency proxy."""
    models = sorted(df["model_label"].unique())
    mc     = model_palette(models)

    stats = []
    for m in models:
        ts = (
            df[df["model_label"] == m]["timestamp"]
            .dropna().sort_values()
        )
        diffs = ts.diff().dropna()
        diffs = diffs[(diffs > 1_000) & (diffs < 600_000)]  # 1 s – 10 min sane range
        stats.append({
            "model_label": m,
            "median_s":    diffs.median() / 1000 if len(diffs) > 0 else np.nan,
            "mean_s":      diffs.mean()   / 1000 if len(diffs) > 0 else np.nan,
        })

    ls = pd.DataFrame(stats).sort_values("median_s", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [mc[m] for m in ls["model_label"]]
    bars   = ax.bar(ls["model_label"], ls["median_s"], color=colors,
                    edgecolor="white", linewidth=0.7, alpha=0.88)
    for bar, val in zip(bars, ls["median_s"]):
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.0f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Median Inter-Episode Interval (seconds)", fontsize=11)
    ax.set_title("G14 — Latency Proxy: Median Time Between Episodes per Model\n"
                 "(derived from episode completion timestamps)", fontsize=13)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G14_latency_proxy.png")


def plot_G15_rolling_success(df: pd.DataFrame):
    models    = sorted(df["model_label"].unique())
    mc        = model_palette(models)
    all_tasks = sorted(df["task_index"].unique())
    window    = 15

    fig, ax = plt.subplots(figsize=(14, 6))
    for model in models:
        series = (
            df[df["model_label"] == model]
            .groupby("task_index")["result"]
            .mean()
            .reindex(all_tasks)
        )
        rolling = series.rolling(window=window, center=True, min_periods=3).mean()
        ax.plot(all_tasks, rolling.values, label=model,
                color=mc[model], linewidth=2, alpha=0.88)

    ax.yaxis.set_major_formatter(PCT_FMT)
    ax.set_xlabel("Task Index (0 → 143)", fontsize=11)
    ax.set_ylabel("Rolling Success Rate", fontsize=11)
    ax.set_title(f"G15 — Success Rate Drift by Task Index Order  "
                 f"(rolling window = {window} tasks)", fontsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "G15_rolling_success.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    df = load_data()
    print(f"  {len(df):,} records | {df['model'].nunique()} models "
          f"| {df['task_index'].nunique()} tasks")

    print("\n── Group 1: Basic Performance ──")
    plot_G1_overall_accuracy(df)
    plot_G2_task_heatmap(df)

    print("\n── Group 2: Run Consistency ──")
    plot_G3_run_consistency(df)
    plot_G4_inconsistency_scatter(df)

    print("\n── Group 3: Conversation Behavior ──")
    plot_G5_message_distribution(df)
    plot_G6_length_by_result(df)
    plot_G7_message_heatmap(df)

    print("\n── Group 4: Tool Usage Patterns ──")
    plot_G8_tool_frequency(df)
    plot_G9_tool_transitions(df)
    plot_G10_terminal_action(df)

    print("\n── Group 5: Task Difficulty & Clustering ──")
    plot_G11_task_difficulty(df)
    plot_G12_task_correlation(df)
    plot_G13_difficulty_tier_performance(df)

    print("\n── Group 6: Temporal & Meta ──")
    plot_G14_latency_proxy(df)
    plot_G15_rolling_success(df)

    print(f"\nAll 15 plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
