#!/usr/bin/env python3
"""
Extended Horizon + Error Taxonomy Analysis — 21 graphs across 6 groups
Models : claude-sonnet-4, gpt-5-mini
JSONL  : results/os-aug-cf-{model}.jsonl       (horizons 1–10, 3 runs, 46 base tasks)
CSVs   : Extension_Failure_Classification_Results/{model}/runs_{H}_failure_results.csv
           error classifications at H = 1, 4, 7, 10

Data notes
----------
- "horizon" here is a task-complexity tier (cf1=H1, …, cf10=H10).
  Each of the 46 base tasks appears at ALL 10 horizons (i.e., 460 task_indices).
  base_task = "aug-" + parts after cfN  (e.g. aug-001-stock-00000)
- Error CSVs cover FAILED tasks only; absence ≡ success at that (model, H).
- Error types observed (6): History Error Accumulation, Instruction Error,
  Planning Errors (Sub-plan & Action), False Assumptions,
  Catastrophic Forgetting, Environment Disturbance / Unable to Detect Change
"""

import json, warnings, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
ERR_DIR     = Path("Extension_Failure_Classification_Results")
OUT_DIR     = Path("final_analysis_figures")
OUT_DIR.mkdir(exist_ok=True)

MODELS = ["claude-sonnet-4", "gpt-5-mini"]
MODEL_COLORS = {"claude-sonnet-4": "#4C72B0", "gpt-5-mini": "#DD8452"}
MODEL_SHORT  = {"claude-sonnet-4": "Claude",  "gpt-5-mini": "GPT-5-mini"}

HORIZONS      = list(range(1, 11))
ERR_HORIZONS  = [1, 4, 7, 10]           # horizons with error classifications
N_TASKS       = 46
DPI           = 180

ALL_ERROR_TYPES = [
    "Planning Errors (Sub-plan & Action)",
    "History Error Accumulation",
    "Instruction Error",
    "False Assumptions",
    "Catastrophic Forgetting",
    "Environment Disturbance / Unable to Detect Change",
]
ERR_SHORT = {
    "Planning Errors (Sub-plan & Action)":              "Planning Errors",
    "History Error Accumulation":                       "History Accum.",
    "Instruction Error":                                "Instruction Error",
    "False Assumptions":                                "False Assumptions",
    "Catastrophic Forgetting":                          "Catast. Forgetting",
    "Environment Disturbance / Unable to Detect Change":"Env. Disturbance",
}
ERR_COLORS = {
    "Planning Errors (Sub-plan & Action)":              "#4C72B0",
    "History Error Accumulation":                       "#DD8452",
    "Instruction Error":                                "#55A868",
    "False Assumptions":                                "#C44E52",
    "Catastrophic Forgetting":                          "#8172B3",
    "Environment Disturbance / Unable to Detect Change":"#937860",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def base_task(task_index: str) -> str:
    """aug-cf4-001-stock-00000 → aug-001-stock-00000"""
    parts = task_index.split("-")
    return "-".join(parts[0:1] + parts[2:])


def count_actions(messages):
    bash = finish = answer = 0
    for m in messages:
        if m["role"] == "agent":
            c = m["content"]
            if "Act: bash"   in c: bash   += 1
            if "Act: finish" in c: finish += 1
            if "Act: answer" in c: answer += 1
    return bash, finish, answer


def save(fig, tag, tight=True):
    path = OUT_DIR / f"{tag}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Data Loading ──────────────────────────────────────────────────────────────
print("Loading JSONL data…")
jsonl_rows = []
for model in MODELS:
    fp = RESULTS_DIR / f"os-aug-cf-{model}.jsonl"
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            bash, finish, answer = count_actions(r["messages"])
            jsonl_rows.append({
                "model":     model,
                "task_index":r["task_index"],
                "base":      base_task(r["task_index"]),
                "horizon":   r["horizon"],
                "run":       r["run_number"],
                "result":    r["result"],
                "n_bash":    bash,
                "n_finish":  finish,
                "n_answer":  answer,
                "n_total":   bash + finish + answer,
            })

df = pd.DataFrame(jsonl_rows)
BASE_TASKS = sorted(df["base"].unique())          # 46 base tasks

# Per (model, base, horizon) mean success rate
task_h_rate = (df.groupby(["model", "base", "horizon"])["result"]
               .mean().reset_index()
               .rename(columns={"result": "rate"}))

print(f"  {len(df)} records, {len(BASE_TASKS)} base tasks, "
      f"{sorted(df.horizon.unique())} horizons")

print("Loading error classification CSVs…")
err_rows = []
for model in MODELS:
    for h in ERR_HORIZONS:
        fp = ERR_DIR / model / f"runs_{h}_failure_results.csv"
        d  = pd.read_csv(fp)
        d["model"]   = model
        d["horizon"] = h
        d["base"]    = d["task_index"].apply(base_task)
        err_rows.append(d)

err_df = pd.concat(err_rows, ignore_index=True)
print(f"  {len(err_df)} error records across models × horizons")

# ─────────────────────────────────────────────────────────────────────────────
# GROUP A — Horizon Effect on Success
# ─────────────────────────────────────────────────────────────────────────────

def plot_A1():
    """Success rate curve vs. horizon (per model)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        sub = df[df.model == model].groupby("horizon")["result"].mean()
        ax.plot(sub.index, sub.values, marker="o", lw=2.5, ms=7,
                color=MODEL_COLORS[model], label=MODEL_SHORT[model])
    ax.set_xlabel("Horizon (task-complexity tier)")
    ax.set_ylabel("Mean Success Rate")
    ax.set_title("A1 — Success Rate vs. Horizon per Model", fontsize=13, fontweight="bold")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    save(fig, "HA1_success_curve")


def plot_A2():
    """Task × Horizon success heatmap — one subplot per model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
    for ax, model in zip(axes, MODELS):
        mat = np.full((N_TASKS, 10), np.nan)
        for i, bt in enumerate(BASE_TASKS):
            for j, h in enumerate(HORIZONS):
                row = task_h_rate[(task_h_rate.model == model) &
                                   (task_h_rate.base == bt) &
                                   (task_h_rate.horizon == h)]
                if len(row):
                    mat[i, j] = row.iloc[0]["rate"]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"H{h}" for h in HORIZONS], fontsize=8)
        ax.set_yticks(range(N_TASKS))
        ax.set_yticklabels(BASE_TASKS, fontsize=5)
        ax.set_title(f"{MODEL_SHORT[model]}", fontsize=11)
        ax.set_xlabel("Horizon")
    axes[0].set_ylabel("Base Task (sorted)")
    plt.colorbar(im, ax=axes, fraction=0.02, label="Mean Success Rate")
    fig.suptitle("A2 — Task × Horizon Success Heatmap\n"
                 "(green=always solved, red=never solved — horizontal gradient = horizon-sensitive tasks)",
                 fontsize=12, fontweight="bold")
    save(fig, "HA2_task_horizon_heatmap")


def plot_A3():
    """Marginal gain per horizon increment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x, w = np.arange(len(HORIZONS) - 1), 0.35
    for i, model in enumerate(MODELS):
        sub    = df[df.model == model].groupby("horizon")["result"].mean()
        deltas = [sub[h] - sub[h - 1] for h in HORIZONS[1:]]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, deltas, width=w, color=MODEL_COLORS[model],
                      alpha=0.85, label=MODEL_SHORT[model])
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h-1}→H{h}" for h in HORIZONS[1:]], rotation=30)
    ax.set_xlabel("Horizon Transition")
    ax.set_ylabel("Δ Success Rate")
    ax.set_title("A3 — Marginal Gain per Horizon Increment\n"
                 "(negative = harder task tier, positive = easier tasks unlock)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    save(fig, "HA3_marginal_gain")


def plot_A4():
    """Histogram: minimum horizon at which each base task was first solved, per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, model in zip(axes, MODELS):
        min_h = []
        never = 0
        for bt in BASE_TASKS:
            sub = task_h_rate[(task_h_rate.model == model) & (task_h_rate.base == bt)]
            solved = sub[sub.rate > 0]["horizon"]
            if len(solved):
                min_h.append(solved.min())
            else:
                never += 1
        ax.hist(min_h, bins=HORIZONS, align="left", color=MODEL_COLORS[model],
                edgecolor="white", alpha=0.85, rwidth=0.8)
        ax.bar([11], [never], color="#C44E52", edgecolor="white", alpha=0.85,
               width=0.8, label=f"Never (n={never})")
        ax.set_xticks(list(range(1, 11)) + [11])
        ax.set_xticklabels([str(h) for h in HORIZONS] + ["Never"])
        ax.set_xlabel("Minimum Horizon at First Success")
        ax.set_ylabel("Number of Tasks")
        ax.set_title(f"A4 — Min Horizon to First Success\n{MODEL_SHORT[model]}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)
    fig.suptitle("A4 — Minimum Horizon Needed per Task (Both Models)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "HA4_min_horizon_hist")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP B — Cross-Model Comparison by Horizon
# ─────────────────────────────────────────────────────────────────────────────

def plot_B1():
    """Head-to-head delta: claude rate − gpt rate per horizon."""
    fig, ax = plt.subplots(figsize=(9, 5))
    c_rate = df[df.model == "claude-sonnet-4"].groupby("horizon")["result"].mean()
    g_rate = df[df.model == "gpt-5-mini"].groupby("horizon")["result"].mean()
    delta  = c_rate - g_rate
    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in delta.values]
    ax.bar(HORIZONS, delta.values, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=1.1)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Horizon (task-complexity tier)")
    ax.set_ylabel("Δ Success Rate  (Claude − GPT)")
    ax.set_title("B1 — Head-to-Head Success Delta per Horizon\n"
                 "(blue = Claude leads, orange = GPT leads)",
                 fontsize=12, fontweight="bold")
    patches = [Patch(color="#4C72B0", label="Claude leads"),
               Patch(color="#DD8452", label="GPT leads")]
    ax.legend(handles=patches)
    ax.grid(axis="y", alpha=0.4)
    save(fig, "HB1_delta_plot")


def plot_B2():
    """Per-task × Horizon winner heatmap."""
    mat = np.full((N_TASKS, 10), np.nan)
    for i, bt in enumerate(BASE_TASKS):
        for j, h in enumerate(HORIZONS):
            cr = task_h_rate[(task_h_rate.model == "claude-sonnet-4") &
                              (task_h_rate.base == bt) &
                              (task_h_rate.horizon == h)]
            gr = task_h_rate[(task_h_rate.model == "gpt-5-mini") &
                              (task_h_rate.base == bt) &
                              (task_h_rate.horizon == h)]
            if len(cr) and len(gr):
                diff = cr.iloc[0]["rate"] - gr.iloc[0]["rate"]
                mat[i, j] = np.clip(diff, -1, 1)

    fig, ax = plt.subplots(figsize=(12, 14))
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"H{h}" for h in HORIZONS])
    ax.set_yticks(range(N_TASKS))
    ax.set_yticklabels(BASE_TASKS, fontsize=5)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Base Task")
    ax.set_title("B2 — Per-Task Winner Heatmap (Claude − GPT)\n"
                 "(green = Claude wins, red = GPT wins, yellow = tie)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02,
                 label="Success Rate Difference (Claude − GPT)")
    save(fig, "HB2_winner_heatmap")


def plot_B3():
    """Crossover horizon: tasks where model rankings swap across horizons."""
    # For each base task: find all horizons where claude>gpt and where gpt>claude
    crossover_counts = defaultdict(int)
    for bt in BASE_TASKS:
        c_rates = {}
        g_rates = {}
        for h in HORIZONS:
            cr = task_h_rate[(task_h_rate.model == "claude-sonnet-4") &
                              (task_h_rate.base == bt) &
                              (task_h_rate.horizon == h)]
            gr = task_h_rate[(task_h_rate.model == "gpt-5-mini") &
                              (task_h_rate.base == bt) &
                              (task_h_rate.horizon == h)]
            if len(cr):
                c_rates[h] = cr.iloc[0]["rate"]
            if len(gr):
                g_rates[h] = gr.iloc[0]["rate"]
        prev_sign = None
        for h in HORIZONS:
            if h in c_rates and h in g_rates:
                curr_sign = np.sign(c_rates[h] - g_rates[h])
                if prev_sign is not None and prev_sign != 0 and curr_sign != 0 and curr_sign != prev_sign:
                    crossover_counts[h] += 1
                prev_sign = curr_sign if curr_sign != 0 else prev_sign

    fig, ax = plt.subplots(figsize=(9, 5))
    hs = HORIZONS[1:]  # crossovers detected at H >= 2
    counts = [crossover_counts.get(h, 0) for h in hs]
    ax.bar(hs, counts, color="#8172B3", alpha=0.85, edgecolor="white")
    ax.set_xticks(hs)
    ax.set_xlabel("Horizon at which Crossover Detected")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("B3 — Crossover Horizon per Task\n"
                 "(tasks where model rankings swap: high bar = many context-dependent tasks)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    save(fig, "HB3_crossover_horizon")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP C — Error Taxonomy Distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_C1():
    """Error type distribution bar chart per model (stacked, normalized)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, model in zip(axes, MODELS):
        sub   = err_df[err_df.model == model]
        total = len(sub)
        counts = {e: (sub.classification_type == e).sum() for e in ALL_ERROR_TYPES}
        labels = [ERR_SHORT[e] for e in ALL_ERROR_TYPES]
        vals   = [counts[e] / total * 100 for e in ALL_ERROR_TYPES]
        colors = [ERR_COLORS[e] for e in ALL_ERROR_TYPES]
        bars   = ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("% of All Failures")
        ax.set_title(f"C1 — Error Distribution\n{MODEL_SHORT[model]}  (n={total})")
        ax.set_xlim(0, max(vals) * 1.25)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("C1 — Error Type Distribution per Model (Failure Fingerprint)\n"
                 "Aggregated across measured horizons H=1,4,7,10",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "HC1_error_distribution")


def plot_C2():
    """Error type × Horizon heatmap (one per model), normalized by N tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub  = err_df[err_df.model == model]
        mat  = np.zeros((len(ALL_ERROR_TYPES), len(ERR_HORIZONS)))
        for j, h in enumerate(ERR_HORIZONS):
            hdf = sub[sub.horizon == h]
            for i, e in enumerate(ALL_ERROR_TYPES):
                mat[i, j] = (hdf.classification_type == e).sum() / N_TASKS
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(ERR_HORIZONS)))
        ax.set_xticklabels([f"H={h}" for h in ERR_HORIZONS])
        ax.set_yticks(range(len(ALL_ERROR_TYPES)))
        ax.set_yticklabels([ERR_SHORT[e] for e in ALL_ERROR_TYPES], fontsize=9)
        ax.set_title(f"{MODEL_SHORT[model]}")
        ax.set_xlabel("Horizon")
        for i in range(len(ALL_ERROR_TYPES)):
            for j in range(len(ERR_HORIZONS)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if mat[i,j] < 0.4 else "white")
        plt.colorbar(im, ax=ax, fraction=0.04,
                     label="Fraction of Tasks with Error")
    fig.suptitle("C2 — Error Type × Horizon Heatmap\n"
                 "(fraction of 46 tasks showing each error at each measured horizon)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "HC2_error_horizon_heatmap")


def plot_C3():
    """Error type × Task heatmap (aggregated across horizons and models)."""
    mat = np.zeros((N_TASKS, len(ALL_ERROR_TYPES)))
    for i, bt in enumerate(BASE_TASKS):
        sub = err_df[err_df.base == bt]
        for j, e in enumerate(ALL_ERROR_TYPES):
            mat[i, j] = (sub.classification_type == e).sum()

    # Sort tasks by dominant error type
    dominant = np.argmax(mat, axis=1)
    order    = np.argsort(dominant)
    mat_sorted = mat[order]
    labels_sorted = [BASE_TASKS[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 16))
    im = ax.imshow(mat_sorted, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(ALL_ERROR_TYPES)))
    ax.set_xticklabels([ERR_SHORT[e] for e in ALL_ERROR_TYPES],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(N_TASKS))
    ax.set_yticklabels(labels_sorted, fontsize=5)
    ax.set_ylabel("Base Task (sorted by dominant error)")
    ax.set_title("C3 — Error Type × Task Heatmap\n"
                 "(implicit task taxonomy from failure signatures — both models × all measured horizons)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02, label="Error Count")
    save(fig, "HC3_error_task_heatmap")


def plot_C4():
    """Radar chart of error fingerprint per model."""
    N    = len(ALL_ERROR_TYPES)
    cats = [ERR_SHORT[e] for e in ALL_ERROR_TYPES]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model in MODELS:
        sub   = err_df[err_df.model == model]
        total = max(len(sub), 1)
        vals  = [(sub.classification_type == e).sum() / total for e in ALL_ERROR_TYPES]
        vals  += vals[:1]
        ax.plot(angles, vals, color=MODEL_COLORS[model], lw=2.5,
                label=MODEL_SHORT[model])
        ax.fill(angles, vals, color=MODEL_COLORS[model], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_rlabel_position(30)
    ax.set_title("C4 — Error Fingerprint Radar Chart\n"
                 "(normalized fraction of failures per error type)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save(fig, "HC4_radar")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP D — Error-Horizon Interaction
# ─────────────────────────────────────────────────────────────────────────────

def plot_D1():
    """Error survival curve — fraction of ALL tasks with each error at measured horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model in zip(axes, MODELS):
        sub = err_df[err_df.model == model]
        for e in ALL_ERROR_TYPES:
            ys = []
            for h in ERR_HORIZONS:
                hdf = sub[sub.horizon == h]
                ys.append((hdf.classification_type == e).sum() / N_TASKS)
            ax.plot(ERR_HORIZONS, ys, marker="o", lw=2,
                    color=ERR_COLORS[e], label=ERR_SHORT[e])
        ax.set_xticks(ERR_HORIZONS)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Fraction of Tasks with Error")
        ax.set_title(f"D1 — Error Survival Curve\n{MODEL_SHORT[model]}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, 10.5)
    fig.suptitle("D1 — Error Survival Curves\n"
                 "(rising = error type increases with task complexity; "
                 "flat near zero = recoverable; flat high = structural gap)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "HD1_error_survival")


def plot_D2():
    """
    Error Transition Matrix per model.
    For consecutive measured horizons (1→4, 4→7, 7→10):
      source = error type at H_n
      target = error type at H_{n+1} OR 'Success' if task not in failure CSV
    Aggregated across all 3 transitions.
    """
    transitions_list = list(zip(ERR_HORIZONS[:-1], ERR_HORIZONS[1:]))
    target_labels = [ERR_SHORT[e] for e in ALL_ERROR_TYPES] + ["Success"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, model in zip(axes, MODELS):
        # Build failed set at each horizon: {base_task → error_type}
        fail_at = {}
        for h in ERR_HORIZONS:
            hdf = err_df[(err_df.model == model) & (err_df.horizon == h)]
            fail_at[h] = dict(zip(hdf.base, hdf.classification_type))

        # Aggregate transition counts
        mat = np.zeros((len(ALL_ERROR_TYPES), len(target_labels)))
        for h1, h2 in transitions_list:
            for bt, src_err in fail_at[h1].items():
                src_idx = ALL_ERROR_TYPES.index(src_err)
                if bt in fail_at[h2]:
                    tgt_err = fail_at[h2][bt]
                    tgt_idx = ALL_ERROR_TYPES.index(tgt_err)
                else:
                    tgt_idx = len(ALL_ERROR_TYPES)   # "Success"
                mat[src_idx, tgt_idx] += 1

        # Row-normalize
        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = np.where(row_sums > 0, mat / row_sums, 0)

        im = ax.imshow(mat_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(target_labels)))
        ax.set_xticklabels(target_labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(ALL_ERROR_TYPES)))
        ax.set_yticklabels([ERR_SHORT[e] for e in ALL_ERROR_TYPES], fontsize=8)
        ax.set_xlabel("Outcome at Next Measured Horizon")
        ax.set_ylabel("Source Error Type at Current Horizon")
        ax.set_title(f"D2 — Error Transition Matrix\n{MODEL_SHORT[model]}")
        for i in range(len(ALL_ERROR_TYPES)):
            for j in range(len(target_labels)):
                if mat[i, j] > 0:
                    ax.text(j, i, f"{mat_norm[i,j]:.2f}", ha="center", va="center",
                            fontsize=7.5, color="white" if mat_norm[i,j] > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("D2 — Error Transition Matrix (Aggregated across H=1→4, 4→7, 7→10)\n"
                 "(last column 'Success' = task resolved at next horizon)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "HD2_error_transition")


def plot_D3():
    """First error at H=1 → final outcome at H=10 matrix."""
    # For each model: base tasks in H=1 failure CSV → their H=10 result
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, model in zip(axes, MODELS):
        h1_fail = err_df[(err_df.model == model) & (err_df.horizon == 1)]
        h1_dict = dict(zip(h1_fail.base, h1_fail.classification_type))

        # H=10 success rate per base task
        h10_rate = task_h_rate[(task_h_rate.model == model) &
                                (task_h_rate.horizon == 10)].set_index("base")["rate"]

        # Build matrix: error_type × [Failure@10, Success@10]
        mat = np.zeros((len(ALL_ERROR_TYPES), 2))
        for bt, err in h1_dict.items():
            if bt in h10_rate.index:
                outcome = 1 if h10_rate[bt] >= 0.5 else 0
                mat[ALL_ERROR_TYPES.index(err), outcome] += 1

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fails at H=10", "Succeeds at H=10"])
        ax.set_yticks(range(len(ALL_ERROR_TYPES)))
        ax.set_yticklabels([ERR_SHORT[e] for e in ALL_ERROR_TYPES], fontsize=9)
        ax.set_title(f"D3 — First Error (H=1) → Final Outcome (H=10)\n{MODEL_SHORT[model]}")
        for i in range(len(ALL_ERROR_TYPES)):
            for j in range(2):
                ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if mat[i, j] > mat.max() * 0.6 else "black")
    fig.suptitle("D3 — First Error at H=1 vs. Final Outcome at H=10\n"
                 "(right column green = recoverable; left column red = persistent failure)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "HD3_first_error_outcome")


def plot_D4():
    """Error distribution by task difficulty tier (min horizon to first solve)."""
    # Compute min horizon to solve per base task (avg across models)
    min_h_per_task = {}
    for bt in BASE_TASKS:
        hs = []
        for model in MODELS:
            sub = task_h_rate[(task_h_rate.model == model) & (task_h_rate.base == bt)]
            solved = sub[sub.rate > 0]["horizon"]
            hs.append(int(solved.min()) if len(solved) else 11)   # 11 = never
        min_h_per_task[bt] = min(hs)

    def tier(h):
        if h <= 3:   return "Easy (H≤3)"
        if h <= 7:   return "Medium (H≤7)"
        if h <= 10:  return "Hard (H≤10)"
        return "Never"

    tiers  = ["Easy (H≤3)", "Medium (H≤7)", "Hard (H≤10)", "Never"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, model in zip(axes, MODELS):
        sub = err_df[err_df.model == model].copy()
        sub["tier"] = sub["base"].map(lambda b: tier(min_h_per_task.get(b, 11)))
        x = np.arange(len(tiers))
        w = 1.0 / (len(ALL_ERROR_TYPES) + 1)
        for k, e in enumerate(ALL_ERROR_TYPES):
            vals = []
            for t in tiers:
                td = sub[sub.tier == t]
                vals.append((td.classification_type == e).sum() /
                             max(len(td), 1))
            ax.bar(x + k * w, vals, width=w, color=ERR_COLORS[e],
                   label=ERR_SHORT[e], alpha=0.85)
        ax.set_xticks(x + (len(ALL_ERROR_TYPES) - 1) * w / 2)
        ax.set_xticklabels(tiers, fontsize=9)
        ax.set_ylabel("Fraction of Failures in Tier")
        ax.set_title(f"D4 — Error by Difficulty Tier\n{MODEL_SHORT[model]}")
        if model == MODELS[0]:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("D4 — Error Concentration by Task Difficulty\n"
                 "(do harder tasks trigger categorically different errors?)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "HD4_error_difficulty")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP E — Horizon Utilization Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def plot_E1():
    """Scatter: actual bash turns used vs. horizon, colored by result."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rng = np.random.RandomState(0)
    for ax, model in zip(axes, MODELS):
        sub    = df[df.model == model].copy()
        jitter = rng.uniform(-0.25, 0.25, len(sub))
        cols   = ["#55A868" if r == 1 else "#C44E52" for r in sub.result]
        ax.scatter(sub.horizon + jitter, sub.n_bash, c=cols,
                   alpha=0.18, s=10, rasterized=True)
        # Mean lines
        for res, col, lbl in [(1, "#1a6e3a", "Success mean"), (0, "#8b1010", "Failure mean")]:
            grp = sub[sub.result == res].groupby("horizon")["n_bash"].mean()
            if len(grp):
                ax.plot(grp.index, grp.values, color=col, lw=2.5,
                        marker="s", ms=6, label=lbl, zorder=5)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Horizon (complexity tier)")
        ax.set_ylabel("Bash Turns Used")
        ax.set_title(f"E1 — Bash Turns vs. Horizon\n{MODEL_SHORT[model]}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    patches = [Patch(color="#55A868", label="Success"), Patch(color="#C44E52", label="Failure")]
    fig.legend(handles=patches, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("E1 — Actual Bash Turns Used vs. Horizon Budget\n"
                 "(mean lines reveal whether models work harder on harder tasks)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "HE1_turns_scatter")


def plot_E2():
    """Violin: bash turns distribution per model × result."""
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [1, 2, 4, 5]
    labels    = [f"{MODEL_SHORT[m]}\n{'Success' if r==1 else 'Failure'}"
                 for m in MODELS for r in [1, 0]]
    colors    = [MODEL_COLORS[m] for m in MODELS for _ in [1, 0]]
    alphas    = [0.85, 0.45, 0.85, 0.45]

    data = [df[(df.model == m) & (df.result == r)]["n_bash"].values
            for m in MODELS for r in [1, 0]]

    vp = ax.violinplot(data, positions=positions, showmedians=True,
                       showextrema=True, widths=0.8)
    for i, (body, col, alpha) in enumerate(zip(vp["bodies"], colors, alphas)):
        body.set_facecolor(col)
        body.set_alpha(alpha)
        body.set_edgecolor("gray")
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[key].set_color("black")
        vp[key].set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Bash Turns Used")
    ax.set_title("E2 — Bash Turn Distribution by Model × Result\n"
                 "(wider/taller = more variable; "
                 "high failure violin = stuck-looping behavior)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "HE2_utilization_violin")


def plot_E3():
    """Efficiency curve: success_rate / horizon vs. horizon."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        sub  = df[df.model == model].groupby("horizon")["result"].mean()
        eff  = sub / np.array(HORIZONS)
        ax.plot(HORIZONS, eff.values, marker="o", lw=2.5, ms=7,
                color=MODEL_COLORS[model], label=MODEL_SHORT[model])
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Efficiency  (success_rate / horizon)")
    ax.set_title("E3 — Horizon Efficiency Curve\n"
                 "(peak = optimal cost-performance operating point)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.4)
    save(fig, "HE3_efficiency_curve")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP F — Task-Level Deep Structure
# ─────────────────────────────────────────────────────────────────────────────

def plot_F1():
    """Task difficulty ranked by minimum horizon needed."""
    min_h = {}
    for bt in BASE_TASKS:
        hs = []
        for model in MODELS:
            sub = task_h_rate[(task_h_rate.model == model) &
                               (task_h_rate.base == bt)]
            solved = sub[sub.rate > 0]["horizon"]
            hs.append(int(solved.min()) if len(solved) else 11)
        min_h[bt] = min(hs)

    sorted_tasks = sorted(BASE_TASKS, key=lambda bt: min_h[bt])
    sorted_vals  = [min_h[bt] for bt in sorted_tasks]

    def bar_color(v):
        if v <= 3:  return "#55A868"
        if v <= 7:  return "#DD8452"
        if v <= 10: return "#C44E52"
        return "#8C8C8C"

    colors = [bar_color(v) for v in sorted_vals]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(N_TASKS), sorted_vals, color=colors, edgecolor="none")
    ax.set_xlim(-1, N_TASKS)
    ax.set_xticks([])
    ax.set_ylabel("Minimum Horizon to First Success  (11 = never solved)")
    ax.set_title("F1 — Task Difficulty Ranking by Minimum Horizon Needed\n"
                 "(green=knowledge tasks, orange=planning-hard, red=very hard, gray=never solved)",
                 fontsize=12, fontweight="bold")
    patches = [Patch(color="#55A868", label="Easy (H≤3)"),
               Patch(color="#DD8452", label="Medium (H≤7)"),
               Patch(color="#C44E52", label="Hard (H≤10)"),
               Patch(color="#8C8C8C", label="Never Solved")]
    ax.legend(handles=patches, loc="upper left")
    save(fig, "HF1_difficulty_ranking")


def plot_F2():
    """Task clustering by error signature — dendrogram + heatmap."""
    try:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
    except ImportError:
        print("  scipy not available — skipping F2 dendrogram, using PCA scatter only")
        hierarchy = None

    # Feature vector per base task: error frequency across all models × measured horizons
    feat = np.zeros((N_TASKS, len(ALL_ERROR_TYPES)))
    for i, bt in enumerate(BASE_TASKS):
        sub = err_df[err_df.base == bt]
        total = max(len(sub), 1)
        for j, e in enumerate(ALL_ERROR_TYPES):
            feat[i, j] = (sub.classification_type == e).sum() / total

    # Task type from base task name (4th part)
    def task_type(bt):
        return bt.split("-")[2]   # aug-001-stock-... → "001"; use index part
    # better: position 2 is the 3-digit number, position 3 is the type
    def task_type2(bt):
        parts = bt.split("-")
        return parts[3] if len(parts) > 3 else parts[-1]

    task_types = [task_type2(bt) for bt in BASE_TASKS]
    type_set   = sorted(set(task_types))
    type_colors = dict(zip(type_set, plt.cm.tab10(np.linspace(0, 1, len(type_set)))))

    if hierarchy is not None:
        fig = plt.figure(figsize=(14, 10))
        ax_dend  = fig.add_axes([0.05, 0.05, 0.20, 0.85])
        ax_heat  = fig.add_axes([0.26, 0.05, 0.60, 0.85])
        ax_cbar  = fig.add_axes([0.87, 0.05, 0.02, 0.85])

        linkage = hierarchy.linkage(pdist(feat, metric="euclidean"), method="ward")
        dend    = hierarchy.dendrogram(linkage, ax=ax_dend,
                                       orientation="left", no_labels=True,
                                       color_threshold=0)
        order   = dend["leaves"]
        sorted_feat   = feat[order]
        sorted_labels = [BASE_TASKS[i] for i in order]
        sorted_types  = [task_types[i] for i in order]

        im = ax_heat.imshow(sorted_feat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax_heat.set_xticks(range(len(ALL_ERROR_TYPES)))
        ax_heat.set_xticklabels([ERR_SHORT[e] for e in ALL_ERROR_TYPES],
                                 rotation=35, ha="right", fontsize=9)
        ax_heat.set_yticks(range(N_TASKS))
        ax_heat.set_yticklabels(sorted_labels, fontsize=5)
        ax_heat.set_title("F2 — Task Clustering by Error Signature\n(ward linkage)",
                           fontsize=12, fontweight="bold")
        plt.colorbar(im, cax=ax_cbar, label="Error Fraction")
        ax_dend.axis("off")
        ax_dend.set_title("Dendrogram", fontsize=9)
    else:
        # PCA fallback
        cov  = np.cov(feat.T)
        vals, vecs = np.linalg.eigh(cov)
        idx  = np.argsort(vals)[::-1]
        pcs  = feat @ vecs[:, idx[:2]]
        fig, ax = plt.subplots(figsize=(9, 7))
        for tt in type_set:
            mask = [t == tt for t in task_types]
            ax.scatter(pcs[mask, 0], pcs[mask, 1], label=tt,
                       color=type_colors[tt], s=60, alpha=0.85)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("F2 — Task Clustering by Error Signature (PCA)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
    save(fig, "HF2_task_clustering")


def plot_F3():
    """Cross-model error agreement heatmap per task × measured horizon."""
    # Build lookup: (model, base, horizon) → error_type or None (success)
    fail_lookup = {}
    for _, row in err_df.iterrows():
        fail_lookup[(row.model, row.base, row.horizon)] = row.classification_type

    # Also confirm success from JSONL (majority vote)
    success_lookup = {}
    for model in MODELS:
        for bt in BASE_TASKS:
            for h in ERR_HORIZONS:
                # Approximate task_index from base + horizon
                rate_row = task_h_rate[(task_h_rate.model == model) &
                                       (task_h_rate.base  == bt) &
                                       (task_h_rate.horizon == h)]
                success_lookup[(model, bt, h)] = (
                    rate_row.iloc[0]["rate"] >= 0.5 if len(rate_row) else None
                )

    # Agreement states: 0=both succeed, 1=claude wins, 2=gpt wins,
    #                   3=both fail same error, 4=both fail diff error
    STATE_COLORS = ["#55A868", "#4C72B0", "#DD8452", "#C44E52", "#D3A0D3"]
    STATE_LABELS = ["Both Succeed", "Claude Wins", "GPT Wins",
                    "Both Fail (same err)", "Both Fail (diff err)"]

    mat = np.full((N_TASKS, len(ERR_HORIZONS)), np.nan)
    for j, h in enumerate(ERR_HORIZONS):
        for i, bt in enumerate(BASE_TASKS):
            c_err = fail_lookup.get(("claude-sonnet-4", bt, h), None)
            g_err = fail_lookup.get(("gpt-5-mini",      bt, h), None)
            c_ok  = success_lookup.get(("claude-sonnet-4", bt, h), None)
            g_ok  = success_lookup.get(("gpt-5-mini",      bt, h), None)
            if c_err is None and g_err is None:   mat[i, j] = 0   # both succeed
            elif c_err is None and g_err is not None: mat[i, j] = 1  # Claude wins
            elif c_err is not None and g_err is None: mat[i, j] = 2  # GPT wins
            elif c_err == g_err:                   mat[i, j] = 3   # same error
            else:                                  mat[i, j] = 4   # diff error

    cmap = mcolors.ListedColormap(STATE_COLORS)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    # Sort rows by dominant state
    order  = np.argsort(np.nanmean(mat, axis=1))
    mat_s  = mat[order]
    labels = [BASE_TASKS[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 16))
    im = ax.imshow(mat_s, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(ERR_HORIZONS)))
    ax.set_xticklabels([f"H={h}" for h in ERR_HORIZONS])
    ax.set_yticks(range(N_TASKS))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("Measured Horizon")
    ax.set_ylabel("Base Task")
    ax.set_title("F3 — Cross-Model Error Agreement\n"
                 "(task-design flaws = both models fail with same error)\n"
                 "(model-specific weakness = models fail with diff errors)",
                 fontsize=11, fontweight="bold")
    patches = [Patch(color=c, label=l)
               for c, l in zip(STATE_COLORS, STATE_LABELS)]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.55, 1.0), fontsize=8)
    save(fig, "HF3_cross_model_agreement")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
PLOTS = [
    ("A1  Success Rate Curve vs. Horizon",             plot_A1),
    ("A2  Task × Horizon Success Heatmap",             plot_A2),
    ("A3  Marginal Gain per Horizon Increment",        plot_A3),
    ("A4  Min Horizon to First Success",               plot_A4),
    ("B1  Head-to-Head Success Delta",                 plot_B1),
    ("B2  Per-Task Winner Heatmap",                    plot_B2),
    ("B3  Crossover Horizon per Task",                 plot_B3),
    ("C1  Error Type Distribution",                    plot_C1),
    ("C2  Error Type × Horizon Heatmap",               plot_C2),
    ("C3  Error Type × Task Heatmap",                  plot_C3),
    ("C4  Error Fingerprint Radar Chart",              plot_C4),
    ("D1  Error Survival Curve",                       plot_D1),
    ("D2  Error Transition Matrix",                    plot_D2),
    ("D3  First Error vs. Final Outcome",              plot_D3),
    ("D4  Error Concentration by Difficulty",          plot_D4),
    ("E1  Bash Turns vs. Horizon (Scatter)",           plot_E1),
    ("E2  Utilization Rate Distribution (Violin)",     plot_E2),
    ("E3  Horizon Efficiency Curve",                   plot_E3),
    ("F1  Task Difficulty Ranking",                    plot_F1),
    ("F2  Task Clustering by Error Signature",         plot_F2),
    ("F3  Cross-Model Error Agreement Heatmap",        plot_F3),
]

for name, fn in PLOTS:
    print(f"Plotting {name}…")
    try:
        fn()
    except Exception as exc:
        import traceback
        print(f"  ERROR: {exc}")
        traceback.print_exc()

print(f"\nAll 21 plots done → {OUT_DIR}/")
