#!/usr/bin/env python3
"""
Horizon Analysis — 15 graphs for claude-sonnet-4 vs gpt-5-mini
Files: results/os-aug-cf-claude-sonnet-4.jsonl, results/os-aug-cf-gpt-5-mini.jsonl

Data structure note:
  Each cf set (cf1–cf10) maps to one horizon level (1–10).
  46 tasks per horizon × 10 horizons = 460 unique tasks.
  Each (task, horizon) evaluated across 3 runs.
  Graphs requiring same-task cross-horizon tracking are adapted accordingly.
"""

import json, re, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── Constants ────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
OUT_DIR = Path("final_analysis_figures")
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "claude-sonnet-4": RESULTS_DIR / "os-aug-cf-claude-sonnet-4.jsonl",
    "gpt-5-mini":      RESULTS_DIR / "os-aug-cf-gpt-5-mini.jsonl",
}
MODEL_COLORS = {"claude-sonnet-4": "#4C72B0", "gpt-5-mini": "#DD8452"}
MODELS   = ["claude-sonnet-4", "gpt-5-mini"]
HORIZONS = list(range(1, 11))
DPI      = 180

# ── Helpers ──────────────────────────────────────────────────────────────────
def count_actions(messages):
    bash = finish = answer = 0
    for m in messages:
        if m["role"] == "agent":
            c = m["content"]
            if "Act: bash"   in c: bash   += 1
            if "Act: finish" in c: finish += 1
            if "Act: answer" in c: answer += 1
    return bash, finish, answer

def get_think_lengths(messages):
    """Return list of (turn_idx, think_char_len) for each agent turn."""
    result, turn = [], 0
    for m in messages:
        if m["role"] == "agent":
            match = re.search(r"Think:(.*?)(?:Act:|$)", m["content"], re.DOTALL)
            if match:
                result.append((turn, len(match.group(1).strip())))
            turn += 1
    return result

def parse_task_type(task_index: str) -> str:
    """aug-cf1-001-stock-00000 → 'stock'"""
    parts = task_index.split("-")
    # format: aug - cf{N} - {num} - {type...} - {instance}
    # parts[3] is the type descriptor
    return parts[3] if len(parts) > 3 else "unknown"

# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    rows = []
    for model, fpath in FILES.items():
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                bash, finish, answer = count_actions(r["messages"])
                rows.append({
                    "model":      model,
                    "task_index": r["task_index"],
                    "category":   r["task_index"].split("-")[1],   # cf1…cf10
                    "task_type":  parse_task_type(r["task_index"]),
                    "horizon":    r["horizon"],
                    "run_number": r["run_number"],
                    "result":     r["result"],
                    "n_bash":     bash,
                    "n_finish":   finish,
                    "n_answer":   answer,
                    "think_lens": get_think_lengths(r["messages"]),
                    "timestamp":  r.get("timestamp", 0),
                })
    return pd.DataFrame(rows)

print("Loading data…")
df = load_data()
print(f"  {len(df)} records | models: {df.model.unique()} | horizons: {sorted(df.horizon.unique())}")

# ── Save helper ──────────────────────────────────────────────────────────────
def save(fig, tag):
    path = OUT_DIR / f"H{tag}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Horizon Effect
# ════════════════════════════════════════════════════════════════════════════

def plot_G1():
    """Success Rate vs. Horizon — one line per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        sub = df[df.model == model].groupby("horizon")["result"].mean()
        ax.plot(sub.index, sub.values, marker="o", color=MODEL_COLORS[model],
                label=model, lw=2.5, ms=7)
    ax.set_xlabel("Horizon (task complexity tier)")
    ax.set_ylabel("Mean Success Rate")
    ax.set_title("G1 — Success Rate vs. Horizon", fontsize=13, fontweight="bold")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    save(fig, "01_success_vs_horizon")


def plot_G2():
    """Marginal change in success rate at each horizon step."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(2, 11)
    width = 0.35
    for i, model in enumerate(MODELS):
        sub    = df[df.model == model].groupby("horizon")["result"].mean()
        deltas = [sub[h] - sub[h - 1] for h in range(2, 11)]
        offset = (i - 0.5) * width
        ax.bar(x + offset, deltas, width=width, color=MODEL_COLORS[model],
               label=model, alpha=0.85)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("Δ Success Rate")
    ax.set_title("G2 — Marginal Change per Horizon Step\n"
                 "(negative = harder tasks at this tier)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h-1}→H{h}" for h in range(2, 11)], rotation=30)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    save(fig, "02_marginal_gain")


def plot_G3():
    """
    First-Solve Run Distribution.
    Data note: tasks appear at only one horizon, so 'first solve' is across runs 1–3.
    Shows how decisive models are — do they solve tasks on run 1 or need retries?
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, model in zip(axes, MODELS):
        sub = df[df.model == model]
        first_run = {}
        for task, grp in sub.groupby("task_index"):
            solved = grp[grp.result == 1]["run_number"]
            first_run[task] = int(solved.min()) if len(solved) > 0 else 0  # 0 = never
        vals_solved  = [v for v in first_run.values() if v > 0]
        n_never      = sum(1 for v in first_run.values() if v == 0)
        counts = [sum(1 for v in vals_solved if v == r) for r in [1, 2, 3]]
        bars = ax.bar([1, 2, 3], counts, color=MODEL_COLORS[model],
                      alpha=0.85, edgecolor="white")
        ax.bar([4], [n_never], color="#C44E52", alpha=0.85, edgecolor="white",
               label="Never solved")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom", fontsize=9)
        ax.text(3.5 + 0.5, n_never + 0.5, str(n_never), ha="center",
                va="bottom", fontsize=9, color="#C44E52")
        ax.set_xlabel("Run Number of First Solve  (4 = Never)")
        ax.set_ylabel("Number of Tasks")
        ax.set_title(f"G3 — First-Run Solve Distribution\n{model}")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Run 1", "Run 2", "Run 3", "Never"])
        ax.grid(axis="y", alpha=0.4)
    fig.suptitle("G3 — First-Run Solve Distribution (Decisiveness)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_first_solve_dist")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Model × Horizon Comparison
# ════════════════════════════════════════════════════════════════════════════

def plot_G4():
    """Performance gap by horizon: claude_rate − gpt_rate, with zero line."""
    fig, ax = plt.subplots(figsize=(9, 5))
    claude_rate = df[df.model == "claude-sonnet-4"].groupby("horizon")["result"].mean()
    gpt_rate    = df[df.model == "gpt-5-mini"].groupby("horizon")["result"].mean()
    gap = claude_rate - gpt_rate
    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in gap.values]
    ax.bar(HORIZONS, gap.values, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xlabel("Horizon (complexity tier)")
    ax.set_ylabel("Δ Success Rate  (Claude − GPT)")
    ax.set_title("G4 — Performance Gap by Horizon\n"
                 "(blue = Claude ahead, orange = GPT ahead)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(HORIZONS)
    ax.grid(axis="y", alpha=0.4)
    patches = [Patch(color="#4C72B0", label="Claude leads"),
               Patch(color="#DD8452", label="GPT leads")]
    ax.legend(handles=patches)
    save(fig, "04_perf_gap")


def plot_G5():
    """Head-to-head outcome matrix per horizon (stacked bar)."""
    def majority(series):
        return int(series.mean() >= 0.5)

    claude_agg = (df[df.model == "claude-sonnet-4"]
                  .groupby(["task_index", "horizon"])["result"]
                  .apply(majority).reset_index())
    gpt_agg    = (df[df.model == "gpt-5-mini"]
                  .groupby(["task_index", "horizon"])["result"]
                  .apply(majority).reset_index())
    merged = claude_agg.merge(gpt_agg, on=["task_index", "horizon"],
                              suffixes=("_claude", "_gpt"))

    both_pass, claude_win, gpt_win, both_fail = [], [], [], []
    for h in HORIZONS:
        sub = merged[merged.horizon == h]
        both_pass.append(((sub.result_claude == 1) & (sub.result_gpt == 1)).sum())
        claude_win.append(((sub.result_claude == 1) & (sub.result_gpt == 0)).sum())
        gpt_win.append(((sub.result_claude == 0) & (sub.result_gpt == 1)).sum())
        both_fail.append(((sub.result_claude == 0) & (sub.result_gpt == 0)).sum())

    fig, ax = plt.subplots(figsize=(11, 5))
    x  = np.arange(len(HORIZONS))
    bp = np.array(both_pass)
    cw = np.array(claude_win)
    gw = np.array(gpt_win)
    bf = np.array(both_fail)
    ax.bar(x, bp,          label="Both Succeed",   color="#55A868")
    ax.bar(x, cw, bottom=bp,          label="Claude Wins",    color="#4C72B0")
    ax.bar(x, gw, bottom=bp + cw,      label="GPT Wins",       color="#DD8452")
    ax.bar(x, bf, bottom=bp + cw + gw, label="Both Fail",      color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in HORIZONS])
    ax.set_xlabel("Horizon (complexity tier)")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("G5 — Head-to-Head Outcome Matrix per Horizon\n"
                 "(majority vote across 3 runs)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    save(fig, "05_head2head_matrix")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Task Analysis
# ════════════════════════════════════════════════════════════════════════════

def plot_G6():
    """Task difficulty ranking — all models, horizons, runs aggregated."""
    task_difficulty = df.groupby("task_index")["result"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(14, 4))
    colors  = ["#C44E52" if v < 0.3 else "#DD8452" if v < 0.7 else "#55A868"
               for v in task_difficulty.values]
    ax.bar(range(len(task_difficulty)), task_difficulty.values,
           color=colors, width=1.0, edgecolor="none")
    ax.axhline(0.3, color="#C44E52", lw=1, ls="--")
    ax.axhline(0.7, color="#55A868", lw=1, ls="--")
    ax.set_xlabel("Tasks (sorted by mean success rate across all models & runs)")
    ax.set_ylabel("Mean Success Rate")
    ax.set_title("G6 — Task Difficulty Ranking (Aggregated)", fontsize=13, fontweight="bold")
    ax.set_xlim(-1, len(task_difficulty))
    patches = [Patch(color="#C44E52", label="Hard (<30%)"),
               Patch(color="#DD8452", label="Medium (30–70%)"),
               Patch(color="#55A868", label="Easy (>70%)")]
    ax.legend(handles=patches, loc="upper left")
    save(fig, "06_task_difficulty")


def plot_G7():
    """
    Horizon × Task-Type Success Rate Heatmap.
    Data note: tasks span only one horizon, so per-task cross-horizon sensitivity
    is unavailable. Instead shows success rate by task_type × horizon (complexity tier),
    revealing which skill domains remain solvable as task complexity grows.
    """
    task_types = sorted(df.task_type.unique())
    matrix     = np.zeros((len(task_types), len(HORIZONS)))
    for i, tt in enumerate(task_types):
        for j, h in enumerate(HORIZONS):
            sub = df[(df.task_type == tt) & (df.horizon == h)]
            matrix[i, j] = sub["result"].mean() if len(sub) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"H={h}" for h in HORIZONS])
    ax.set_yticks(range(len(task_types)))
    ax.set_yticklabels(task_types)
    ax.set_xlabel("Horizon (complexity tier)")
    ax.set_ylabel("Task Type (domain)")
    ax.set_title("G7 — Task Domain × Horizon Success Rate Heatmap\n"
                 "(shows which domains degrade fastest with complexity)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Mean Success Rate")
    for i in range(len(task_types)):
        for j in range(len(HORIZONS)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < matrix[i, j] < 0.85 else "white")
    save(fig, "07_task_domain_heatmap")


def plot_G8():
    """
    First Run to Achieve Success — heatmap (task × model).
    Data note: 'horizon' encodes complexity tier, not allowed turns for the same task.
    Color = first run_number (1/2/3) at which result=1; white/grey = never solved.
    Shows reliability and consistency patterns.
    """
    all_tasks = sorted(df.task_index.unique())
    matrix    = np.full((len(all_tasks), len(MODELS)), fill_value=np.nan)
    for i, task in enumerate(all_tasks):
        for j, model in enumerate(MODELS):
            sub    = df[(df.task_index == task) & (df.model == model)]
            solved = sub[sub.result == 1]["run_number"]
            if len(solved) > 0:
                matrix[i, j] = solved.min()

    # Sort tasks by mean first-run (NaN → 4)
    sort_key = np.nanmean(np.where(np.isnan(matrix), 4, matrix), axis=1)
    order    = np.argsort(sort_key)
    matrix   = matrix[order]

    fig, ax = plt.subplots(figsize=(5, 14))
    cmap    = plt.cm.get_cmap("RdYlGn_r", 4)   # 1, 2, 3, NaN=never
    im      = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=1, vmax=4)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=15)
    ax.set_yticks([])
    ax.set_ylabel(f"Tasks (sorted by first-run solve, n={len(all_tasks)})")
    ax.set_title("G8 — First Run to Achieve Success\n(4 / grey = never solved)",
                 fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, ticks=[1.375, 2.125, 2.875, 3.625])
    cbar.ax.set_yticklabels(["Run 1", "Run 2", "Run 3", "Never"])
    save(fig, "08_first_run_success")


def plot_G9():
    """Task domain analysis — success rate by task type × model."""
    task_types = sorted(df.task_type.unique())
    x     = np.arange(len(task_types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(MODELS):
        sub  = df[df.model == model].groupby("task_type")["result"].mean().reindex(task_types)
        ax.bar(x + (i - 0.5) * width, sub.values, width=width,
               color=MODEL_COLORS[model], label=model, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(task_types, rotation=30, ha="right")
    ax.set_xlabel("Task Domain / Type")
    ax.set_ylabel("Mean Success Rate (all horizons & runs)")
    ax.set_title("G9 — Task Domain Performance by Model\n"
                 "(per-domain strengths embedded in task naming)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    save(fig, "09_task_domain")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 4 — Conversation Behavior
# ════════════════════════════════════════════════════════════════════════════

def plot_G10():
    """Actual bash turns used vs. horizon complexity tier, colored by result."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rng = np.random.RandomState(42)
    for ax, model in zip(axes, MODELS):
        sub    = df[df.model == model].copy()
        jitter = rng.uniform(-0.3, 0.3, len(sub))
        colors = ["#55A868" if r == 1 else "#C44E52" for r in sub.result]
        ax.scatter(sub.horizon + jitter, sub.n_bash, c=colors,
                   alpha=0.2, s=8, rasterized=True)
        # Overlay means per horizon × result
        for res, col, lbl in [(1, "#2d7a4f", "Success mean"), (0, "#8b1010", "Failure mean")]:
            grp = sub[sub.result == res].groupby("horizon")["n_bash"].mean()
            if len(grp):
                ax.plot(grp.index, grp.values, color=col, lw=2.5, marker="s",
                        ms=6, label=lbl, zorder=5)
        ax.set_xlabel("Horizon (complexity tier)")
        ax.set_ylabel("Actual Bash Turns Used")
        ax.set_title(f"G10 — Turns Used vs. Horizon\n{model}")
        ax.set_xticks(HORIZONS)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    patches = [Patch(color="#55A868", label="Success"), Patch(color="#C44E52", label="Failure")]
    fig.legend(handles=patches, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("G10 — Actual Bash Turns Used vs. Horizon Allowed", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "10_turns_vs_horizon")


def plot_G11():
    """Action type distribution (bash / finish / answer) per model × result."""
    combos      = [(m, r) for m in MODELS for r in [1, 0]]
    labels      = [f"{m}\n{'Success' if r==1 else 'Failure'}" for m, r in combos]
    bash_vals, finish_vals, answer_vals = [], [], []
    for model, res in combos:
        sub = df[(df.model == model) & (df.result == res)]
        n   = len(sub)
        bash_vals.append(sub.n_bash.sum()   / n if n else 0)
        finish_vals.append(sub.n_finish.sum() / n if n else 0)
        answer_vals.append(sub.n_answer.sum() / n if n else 0)

    x, w = np.arange(len(combos)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,  bash_vals,   width=w, label="bash",   color="#4C72B0", alpha=0.85)
    ax.bar(x,      finish_vals, width=w, label="finish",  color="#55A868", alpha=0.85)
    ax.bar(x + w,  answer_vals, width=w, label="answer",  color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Avg Actions per Episode")
    ax.set_title("G11 — Action Type Distribution by Model × Result\n"
                 "(high finish on failures = premature quit; high bash = stuck-looping)",
                 fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    save(fig, "11_action_distribution")


def plot_G12():
    """Think content length across turns, per model, split by success/failure."""
    # Aggregate: (model, result, turn) → [think_lengths]
    agg = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        key = (row["model"], row["result"])
        for turn, length in row["think_lens"]:
            agg[key][turn].append(length)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    MAX_TURN = 12
    for ax, model in zip(axes, MODELS):
        for res, style, lbl in [(1, "-", "Success"), (0, "--", "Failure")]:
            turn_data = agg[(model, res)]
            if not turn_data:
                continue
            turns = list(range(MAX_TURN + 1))
            means = [np.mean(turn_data[t]) if t in turn_data else np.nan for t in turns]
            ax.plot(turns, means, linestyle=style, color=MODEL_COLORS[model],
                    label=lbl, lw=2, marker="o", ms=4)
        ax.set_xlabel("Turn Number (within episode)")
        ax.set_ylabel("Avg Think Length (characters)")
        ax.set_title(f"G12 — Think Length Across Turns\n{model}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(range(MAX_TURN + 1))
    fig.suptitle("G12 — Think Content Length Across Turns per Model\n"
                 "(declining curve on failures = cognitive exhaustion signal)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "12_think_length")


# ════════════════════════════════════════════════════════════════════════════
# GROUP 5 — Efficiency & Reliability
# ════════════════════════════════════════════════════════════════════════════

def plot_G13():
    """
    Horizon Efficiency Curve (normalized).
    Data note: horizons encode task complexity tiers (higher = harder), not
    allowed turns for the same task. Efficiency = success_rate(H) / success_rate(H=1),
    showing what fraction of the easiest-tier performance is retained as complexity grows.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        sub      = df[df.model == model].groupby("horizon")["result"].mean()
        baseline = sub[1]
        eff      = sub / baseline if baseline > 0 else sub
        ax.plot(eff.index, eff.values, marker="o", color=MODEL_COLORS[model],
                label=model, lw=2.5, ms=7)
    ax.axhline(0.9, color="gray", ls="--", lw=1, label="90% of H=1 performance")
    ax.axhline(0.5, color="gray", ls=":",  lw=1, label="50% of H=1 performance")
    ax.set_xlabel("Horizon (complexity tier)")
    ax.set_ylabel("Efficiency  (success@H / success@H=1)")
    ax.set_title("G13 — Horizon Efficiency Curve (Normalized to H=1)\n"
                 "(1.0 = same performance as easiest tier)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(alpha=0.4)
    save(fig, "13_efficiency_curve")


def plot_G14():
    """Run consistency at each horizon — stacked bar: always-pass / inconsistent / always-fail."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub  = df[df.model == model]
        ap_list, af_list, inc_list = [], [], []
        for h in HORIZONS:
            grp = sub[sub.horizon == h].groupby("task_index")["result"].agg(list)
            ap  = sum(all(v == 1 for v in vs) for vs in grp)
            af  = sum(all(v == 0 for v in vs) for vs in grp)
            inc = len(grp) - ap - af
            ap_list.append(ap); af_list.append(af); inc_list.append(inc)

        x  = np.arange(len(HORIZONS))
        ap = np.array(ap_list); af = np.array(af_list); inc = np.array(inc_list)
        ax.bar(x, ap,          label="Always Pass",   color="#55A868")
        ax.bar(x, inc, bottom=ap,          label="Inconsistent", color="#DD8452")
        ax.bar(x, af,  bottom=ap + inc,    label="Always Fail",  color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H={h}" for h in HORIZONS], rotation=30)
        ax.set_title(f"G14 — Run Consistency per Horizon\n{model}")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Number of Tasks")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)
    fig.suptitle("G14 — Run Consistency at Each Horizon\n"
                 "(high inconsistency = stochastic decision boundary)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "14_run_consistency")


def plot_G15():
    """
    Cross-Run Stability Matrix.
    Data note: 'solved at H=1 still solved at H=10' is unavailable (different tasks).
    Instead: for tasks solved in run 1, what fraction are also solved in run 2 and run 3?
    Reveals whether first-run successes are robust or lucky.
    Per model × horizon tier.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    check_runs = [2, 3]

    for ax, model in zip(axes, MODELS):
        sub    = df[df.model == model]
        r1_stable_r2, r1_stable_r3, n_solved_r1 = [], [], []
        for h in HORIZONS:
            htier = sub[sub.horizon == h]
            # tasks solved in run 1
            solved_r1 = set(htier[(htier.run_number == 1) & (htier.result == 1)].task_index)
            n_solved_r1.append(len(solved_r1))
            for check_run, out_list in [(2, r1_stable_r2), (3, r1_stable_r3)]:
                if len(solved_r1) == 0:
                    out_list.append(np.nan)
                    continue
                sub_run = htier[(htier.run_number == check_run) & (htier.task_index.isin(solved_r1))]
                out_list.append(sub_run["result"].mean() if len(sub_run) else np.nan)

        x = np.arange(len(HORIZONS))
        ax.plot(x, r1_stable_r2, marker="o", color="#4C72B0", lw=2, label="Still pass in Run 2")
        ax.plot(x, r1_stable_r3, marker="s", color="#DD8452", lw=2, label="Still pass in Run 3")
        ax.axhline(1.0, color="gray", ls="--", lw=1, label="Perfect stability")
        # Annotate N solved in run 1
        for i, n in enumerate(n_solved_r1):
            ax.text(i, 0.02, f"n={n}", ha="center", va="bottom", fontsize=7, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H={h}" for h in HORIZONS], rotation=30)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel("Horizon (complexity tier)")
        ax.set_ylabel("Fraction of Run-1 Successes Still Solved")
        ax.set_title(f"G15 — First-Run Success Stability\n{model}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)
    fig.suptitle("G15 — 'Solved Early, Still Solved Later' Stability Matrix\n"
                 "(< 1.0 = non-monotonic: model sometimes fails on tasks it solved before)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "15_stability_matrix")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
PLOTS = [
    ("G1  Success Rate vs Horizon",          plot_G1),
    ("G2  Marginal Change per Horizon Step",  plot_G2),
    ("G3  First-Run Solve Distribution",      plot_G3),
    ("G4  Performance Gap by Horizon",        plot_G4),
    ("G5  Head-to-Head Outcome Matrix",       plot_G5),
    ("G6  Task Difficulty Ranking",           plot_G6),
    ("G7  Task Domain × Horizon Heatmap",     plot_G7),
    ("G8  First Run to Achieve Success",      plot_G8),
    ("G9  Task Domain Analysis",              plot_G9),
    ("G10 Turns Used vs Horizon",             plot_G10),
    ("G11 Action Type Distribution",          plot_G11),
    ("G12 Think Length Across Turns",         plot_G12),
    ("G13 Horizon Efficiency Curve",          plot_G13),
    ("G14 Run Consistency per Horizon",       plot_G14),
    ("G15 First-Run Stability Matrix",        plot_G15),
]

for name, fn in PLOTS:
    print(f"Plotting {name}…")
    try:
        fn()
    except Exception as exc:
        import traceback
        print(f"  ERROR in {name}: {exc}")
        traceback.print_exc()

print("\nAll 15 plots done.")
