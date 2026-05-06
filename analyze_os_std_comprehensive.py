#!/usr/bin/env python3
"""
Standard OS-Benchmark Comprehensive Analysis — 23 graphs (Groups A–G)
Dataset : 8 API models × 144 tasks × 2 runs × error taxonomy
JSONL   : results/os-std-*.jsonl
CSVs    : results/failure_classifications-*.csv
Output  : final_analysis_figures/STD_*.png

Graph inventory
  A1  Ranked Success Rate Bar Chart (run-shading decomposition)
  A2  Model × Task Success Heatmap
  A3  Pairwise Model Overlap Matrix
  B1  Per-Model Consistency vs. Success Rate
  B2  Flip Matrix Heatmap  (Model × Task)
  B3  Run 1 vs. Run 2 Success Rate Scatter
  B4  Task Instability Ranking
  C1  Stacked Error Distribution per Model
  C2  Error Type × Model Heatmap
  C3  Global Error Type Frequency
  D1  Multi-Model Error Fingerprint Radar
  D2  Per-Model Multi-Metric Profile Radar
  D3  Small Multiple Radars (8 panels)
  D4  Task-Category Success Rate Radar
  E1  Cross-Model Error Agreement Heatmap (8×8)
  E2  Error Co-occurrence Heatmap
  E3  Per-Error-Type Model Ranking
  F1  Task Difficulty Spectrum
  F2  Task Clustering by Error Signature
  F3  Universal-Hard vs. Differential-Hard Scatter
  F4  Error-to-Success Transition Map
  G1  Model Similarity Dendrogram
  G2  Ensemble Potential Radar
"""

import json, glob, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
OUT_DIR     = Path("final_analysis_figures")
OUT_DIR.mkdir(exist_ok=True)

N_TASKS = 144
DPI     = 180

CSV_TO_MODEL = {
    "failure_classifications_gpt5-mini.csv": "gpt-5-mini",
    "failure_classifications-claude-3.csv":  "claude-3",
    "failure_classifications-claude-soon.csv": "claude-sonnet-3.7",
    "failure_classifications-gpt3.5.csv":    "gpt-3.5-turbo",
    "failure_classifications-gpt4.csv":      "GPT-4",
    "failure_classifications-gpt4o.csv":     "gpt-4o",
    "failure_classifications-o3-mini.csv":   "o3-mini",
    "failure_classifications-o4-mini.csv":   "o4-mini",
}

# Preferred display order (weakest → strongest roughly)
MODEL_PREF_ORDER = [
    "gpt-3.5-turbo", "claude-3", "GPT-4", "gpt-4o",
    "gpt-5-mini", "claude-sonnet-3.7", "o3-mini", "o4-mini",
]
MODEL_SHORT = {
    "claude-3":          "Claude-3",
    "claude-sonnet-3.7": "Claude-3.7",
    "gpt-3.5-turbo":     "GPT-3.5",
    "GPT-4":             "GPT-4",
    "gpt-4o":            "GPT-4o",
    "gpt-5-mini":        "GPT-5-mini",
    "o3-mini":           "o3-mini",
    "o4-mini":           "o4-mini",
}

ERR_SHORT = {
    "Environment Disturbance / Unable to Detect Change": "Env. Disturbance",
    "False Assumptions":                                 "False Assumptions",
    "Instruction Error":                                 "Instruction Error",
    "Planning Errors (Sub-plan & Action)":               "Planning Errors",
}

TASK_CATEGORIES_DEF = {
    "Log Analysis":       ["log", ".log", "stock", "how many times", "count"],
    "File Management":    ["file", "directory", "mkdir", " mv ", " cp ", " rm ", "create a file"],
    "Package/System":     ["install", "apt", "package", "upgrade", "uninstall", "cpu", "memory",
                           "disk", "space", "uname", "uptime"],
    "User/Auth":          ["user", "password", "passwd", "adduser", "group", "sudoer"],
    "Process Management": ["process", "pid", "kill", " ps ", "running", "daemon", "service"],
    "Networking":         ["network", "port", "ip address", "ssh", "curl", "wget", "http", "dns"],
    "Permissions":        ["permission", "chmod", "chown", "owner"],
    "Text Processing":    ["grep", "sed", "awk", "sort", "wc ", "output", "line", "pattern"],
}

# ── Data Loading ──────────────────────────────────────────────────────────────
print("Loading JSONL data…")
raw_rows  = []
task_text = {}        # task_index → first user message

for fp in sorted(glob.glob(str(RESULTS_DIR / "os-std-*.jsonl"))):
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ti = int(r["task_index"])
            res_raw = r["result"]
            if isinstance(res_raw, str):
                if res_raw.lower() in ("1", "true", "success"): res_val = 1
                elif res_raw.lower() in ("0", "false", "fail", "error"): res_val = 0
                else: res_val = 0
            else:
                res_val = int(res_raw)
            raw_rows.append(dict(
                model=r["model"],
                task=ti,
                run=r["run_number"],
                result=res_val,
            ))
            if ti not in task_text:
                for msg in r.get("messages", []):
                    if msg["role"] == "user":
                        task_text[ti] = msg["content"].lower()
                        break

df = pd.DataFrame(raw_rows)
MODELS = [m for m in MODEL_PREF_ORDER if m in df.model.unique()]
for m in sorted(df.model.unique()):
    if m not in MODELS:
        MODELS.append(m)

print(f"  {len(df)} records, {len(MODELS)} models: {[MODEL_SHORT.get(m,m) for m in MODELS]}")

# Colour palette (tab10)
_pal = plt.cm.tab10(np.linspace(0, 0.9, len(MODELS)))
MODEL_COLOR = {m: _pal[i] for i, m in enumerate(MODELS)}

# ── Error CSV Loading ─────────────────────────────────────────────────────────
print("Loading error CSVs…")
err_frames = []
for fname, model in CSV_TO_MODEL.items():
    fp = RESULTS_DIR / fname
    if fp.exists():
        d = pd.read_csv(fp)
        d["model"] = model
        d["task"]  = d["task_index"].astype(int)
        err_frames.append(d)

df_err = pd.concat(err_frames, ignore_index=True)
ALL_ERR = sorted(df_err["classification_type"].unique())
N_ERR   = len(ALL_ERR)

_ep = plt.cm.Set2(np.linspace(0, 1, N_ERR))
ERR_COLOR = {e: _ep[i] for i, e in enumerate(ALL_ERR)}
print(f"  {len(df_err)} records, {N_ERR} error types: {ALL_ERR}")

# ── Data Preparation ──────────────────────────────────────────────────────────
print("Building summary table…")
ALL_TASKS = list(range(N_TASKS))

def _get_run(model, task, run_no):
    v = df[(df.model == model) & (df.task == task) & (df.run == run_no)]["result"].values
    return int(v[0]) if len(v) else np.nan

summary_rows = []
for model in MODELS:
    for task in ALL_TASKS:
        r1 = _get_run(model, task, 1)
        r2 = _get_run(model, task, 2)
        valid = [v for v in (r1, r2) if not (isinstance(v, float) and np.isnan(v))]
        rate  = float(np.mean(valid)) if valid else np.nan
        both_known = not (np.isnan(r1) if isinstance(r1, float) else False) and \
                     not (np.isnan(r2) if isinstance(r2, float) else False)
        consistent = bool(r1 == r2) if both_known else np.nan
        flip       = bool(r1 != r2) if both_known else False
        err_row    = df_err[(df_err.model == model) & (df_err.task == task)]
        err_type   = err_row.iloc[0]["classification_type"] if len(err_row) else None
        summary_rows.append(dict(
            model=model, task=task, r1=r1, r2=r2,
            rate=rate, consistent=consistent, flip=flip, err_type=err_type,
        ))

sm = pd.DataFrame(summary_rows)

# Per-model stats
def _safe_mean(s):
    v = s.dropna()
    return float(v.mean()) if len(v) else np.nan

mstats = sm.groupby("model").agg(
    success_rate=("rate",       lambda x: _safe_mean(x)),
    consistency =("consistent", lambda x: _safe_mean(x.dropna())),
    n_flip      =("flip",       "sum"),
).reindex(MODELS)

# Task-level difficulty: number of models with rate > 0
task_n_solve = sm[sm.rate > 0].groupby("task")["model"].nunique().reindex(ALL_TASKS, fill_value=0)
task_mean    = sm.groupby("task")["rate"].mean()
task_var     = sm.groupby("task")["rate"].var().fillna(0)

# Task category inference
def _infer_cat(task_idx):
    txt = task_text.get(task_idx, "")
    for cat, kws in TASK_CATEGORIES_DEF.items():
        if any(kw in txt for kw in kws):
            return cat
    return "Other"

task_cat = {t: _infer_cat(t) for t in ALL_TASKS}
TASK_CATS = sorted(set(task_cat.values()))
print(f"  Task categories: {TASK_CATS}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(fig, tag):
    path = OUT_DIR / f"STD_{tag}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def ms(m):
    """Short model label."""
    return MODEL_SHORT.get(m, m)

def radar(ax, vals, labels, color, label=None, alpha=0.15, lw=2.2):
    """Draw one filled polygon on a polar axis."""
    N = len(labels)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    v2  = list(vals) + [vals[0]]
    a2  = ang + [ang[0]]
    ax.plot(a2, v2, color=color, lw=lw, label=label)
    ax.fill(a2, v2, color=color, alpha=alpha)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=8)

def decompose(model):
    """Both-success / run1-only / run2-only / both-fail fractions."""
    s = sm[sm.model == model]
    n = max(len(s), 1)
    bs  = ((s.r1 == 1) & (s.r2 == 1)).sum()
    r1o = ((s.r1 == 1) & (s.r2 == 0)).sum()
    r2o = ((s.r1 == 0) & (s.r2 == 1)).sum()
    bf  = ((s.r1 == 0) & (s.r2 == 0)).sum()
    return bs/n, r1o/n, r2o/n, bf/n


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP A — Model Performance Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def plot_A1():
    """Ranked Success Rate Bar Chart with run-shading decomposition."""
    ordered = sorted(MODELS, key=lambda m: mstats.loc[m, "success_rate"], reverse=True)
    x = np.arange(len(ordered))

    bs_arr, r1o_arr, r2o_arr = [], [], []
    for m in ordered:
        bs, r1o, r2o, _ = decompose(m)
        bs_arr.append(bs); r1o_arr.append(r1o); r2o_arr.append(r2o)

    fig, ax = plt.subplots(figsize=(12, 6))
    w = 0.65
    ax.bar(x, bs_arr,  width=w, color="#1565C0", label="Both runs succeed (reliable)", alpha=0.92)
    ax.bar(x, r1o_arr, width=w, bottom=bs_arr, color="#64B5F6", label="Run 1 only", alpha=0.92)
    bot2 = [a+b for a, b in zip(bs_arr, r1o_arr)]
    ax.bar(x, r2o_arr, width=w, bottom=bot2, color="#BBDEFB", label="Run 2 only", alpha=0.92)
    totals = [a+b+c for a, b, c in zip(bs_arr, r1o_arr, r2o_arr)]
    ax.plot(x, totals, "k^", ms=8, zorder=5, label="Aggregate success rate")

    ax.set_xticks(x)
    ax.set_xticklabels([ms(m) for m in ordered], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Fraction of 144 Tasks"); ax.set_ylim(0, 1.12)
    ax.set_title("A1 — Ranked Success Rate: Run-Shading Decomposition\n"
                 "(dark blue = reliable; light blue = single-run lucky — "
                 "same height, different shading = different reliability)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.35)
    save(fig, "A1_ranked_success")


def plot_A2():
    """Model × Task Success Heatmap (rows sorted by success rate)."""
    ordered = sorted(MODELS, key=lambda m: mstats.loc[m, "success_rate"], reverse=True)
    mat = np.full((len(ordered), N_TASKS), np.nan)
    for i, m in enumerate(ordered):
        for j in ALL_TASKS:
            row = sm[(sm.model == m) & (sm.task == j)]
            if len(row):
                mat[i, j] = row.iloc[0]["rate"]

    fig, ax = plt.subplots(figsize=(22, 4))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([ms(m) for m in ordered], fontsize=9)
    ax.set_xticks(range(0, N_TASKS, 10))
    ax.set_xlabel("Task Index (0–143)")
    ax.set_title("A2 — Model × Task Success Heatmap  (rows: strong→weak; sorted by overall rate)\n"
                 "Vertical dark stripes = universally hard tasks  |  "
                 "Block patterns = task clusters testing same capability",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.015, label="Success Rate (0/0.5/1.0)")
    save(fig, "A2_model_task_heatmap")


def plot_A3():
    """Pairwise Model Overlap: 'both-solve' redundancy + complementarity."""
    n = len(MODELS)
    solved_by = {m: set(sm[(sm.model == m) & (sm.rate > 0)]["task"].values)
                 for m in MODELS}
    N = len(ALL_TASKS)

    both_mat = np.zeros((n, n))
    comp_mat = np.zeros((n, n))
    for i, mi in enumerate(MODELS):
        for j, mj in enumerate(MODELS):
            ai, aj = solved_by[mi], solved_by[mj]
            both_mat[i, j] = len(ai & aj) / N
            comp_mat[i, j] = (len(ai - aj) + len(aj - ai)) / N

    labs = [ms(m) for m in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, mat, title, cmap, vmax in [
        (axes[0], both_mat, "Both Solve (Redundancy)",      "Blues",  1.0),
        (axes[1], comp_mat, "Unique Coverage (Complementarity)", "YlOrRd", 1.0),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if mat[i,j] > vmax*0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle("A3 — Pairwise Model Overlap Matrix\n"
                 "(left: high = redundant pair; right: high = complementary → ensemble value)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "A3_pairwise_overlap")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP B — Run Stability
# ═══════════════════════════════════════════════════════════════════════════════

def plot_B1():
    """Per-Model Consistency vs. Success Rate."""
    ordered = sorted(MODELS, key=lambda m: mstats.loc[m, "consistency"] or 0, reverse=True)
    x = np.arange(len(ordered))
    cons = [mstats.loc[m, "consistency"] or 0 for m in ordered]
    succ = [mstats.loc[m, "success_rate"] or 0 for m in ordered]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.19, cons, width=0.36, color="#43A047", label="Consistency (both runs agree)", alpha=0.9)
    ax.bar(x + 0.19, succ, width=0.36, color="#1976D2", label="Overall Success Rate",          alpha=0.9, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([ms(m) for m in ordered], rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Rate")
    ax.axhline(0.8, color="gray", lw=1, ls="--", alpha=0.55)
    ax.text(len(ordered)-0.5, 0.81, "80% threshold", fontsize=8, color="gray")
    ax.set_title("B1 — Per-Model Consistency vs. Success Rate\n"
                 "(green bar taller than blue = reliable model; "
                 "blue taller than green = lucky/unstable — dangerous for production)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.35)
    save(fig, "B1_consistency_rate")


def plot_B2():
    """Flip Matrix Heatmap: Model × Task  (4 states)."""
    # 0=both fail, 1=run2-only, 2=run1-only, 3=both succeed
    mat = np.full((len(MODELS), N_TASKS), np.nan)
    for i, m in enumerate(MODELS):
        for j in ALL_TASKS:
            row = sm[(sm.model == m) & (sm.task == j)]
            if not len(row): continue
            r = row.iloc[0]
            r1, r2 = r.r1, r.r2
            if isinstance(r1, float) and np.isnan(r1): continue
            if isinstance(r2, float) and np.isnan(r2): continue
            if   r1 == 1 and r2 == 1: mat[i, j] = 3
            elif r1 == 1 and r2 == 0: mat[i, j] = 2
            elif r1 == 0 and r2 == 1: mat[i, j] = 1
            else:                      mat[i, j] = 0

    cmap  = mcolors.ListedColormap(["#B71C1C", "#FF8F00", "#64B5F6", "#2E7D32"])
    norm  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(24, 4))
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([ms(m) for m in MODELS], fontsize=9)
    ax.set_xticks(range(0, N_TASKS, 10)); ax.set_xlabel("Task Index")
    ax.set_title("B2 — Flip Matrix: Model × Task\n"
                 "(orange/blue columns across many models = task-level noise; "
                 "isolated flips = model-specific instability)",
                 fontsize=11, fontweight="bold")
    patches = [Patch(color="#B71C1C", label="Both Fail"),
               Patch(color="#FF8F00", label="Run2 Only"),
               Patch(color="#64B5F6", label="Run1 Only"),
               Patch(color="#2E7D32", label="Both Succeed")]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=4,
              bbox_to_anchor=(1.0, 1.18))
    save(fig, "B2_flip_matrix")


def plot_B3():
    """Run 1 vs. Run 2 Success Rate Scatter."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for m in MODELS:
        s = sm[sm.model == m]
        r1r = s["r1"].apply(lambda v: 0 if isinstance(v, float) and np.isnan(v) else v).mean()
        r2r = s["r2"].apply(lambda v: 0 if isinstance(v, float) and np.isnan(v) else v).mean()
        ax.scatter(r1r, r2r, s=140, color=MODEL_COLOR[m], zorder=4)
        ax.annotate(ms(m), (r1r, r2r), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Run 1 Success Rate"); ax.set_ylabel("Run 2 Success Rate")
    ax.set_title("B3 — Run 1 vs. Run 2 Success Rate per Model\n"
                 "(above diagonal = improved; far off-diagonal = run-order sensitive)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.35)
    save(fig, "B3_run_scatter")


def plot_B4():
    """Task Instability Ranking."""
    flip_per_task = sm.groupby("task")["flip"].sum().reindex(ALL_TASKS, fill_value=0)
    top25 = flip_per_task.sort_values(ascending=False).head(25)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histogram
    axes[0].hist(flip_per_task.values, bins=range(0, len(MODELS)+2), align="left",
                 rwidth=0.8, color="#7B1FA2", edgecolor="white", alpha=0.9)
    axes[0].set_xlabel(f"Models flipping (run1≠run2)  [max={len(MODELS)}]")
    axes[0].set_ylabel("Number of Tasks")
    axes[0].set_title("B4a — Task Instability Distribution\n"
                       "(tasks at 5+ = benchmark noise; tasks at 0 = clean signal)")
    axes[0].axvline(5, color="red", lw=1.2, ls="--", alpha=0.7)
    axes[0].grid(axis="y", alpha=0.35)

    # Top-25 horizontal bar
    colors25 = ["#C62828" if v >= 5 else "#7B1FA2" for v in top25.values]
    axes[1].barh([f"Task {t}" for t in top25.index], top25.values,
                 color=colors25, edgecolor="white", alpha=0.9)
    axes[1].axvline(5, color="red", lw=1.2, ls="--", alpha=0.7)
    axes[1].set_xlabel("Number of Models Flipping")
    axes[1].set_title("B4b — Top-25 Most Unstable Tasks\n"
                       "(red = 5+ flips → candidate for benchmark redesign)")
    axes[1].grid(axis="x", alpha=0.35)

    fig.suptitle("B4 — Task Instability Ranking", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "B4_task_instability")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP C — Error Taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

def plot_C1():
    """Stacked Error Distribution per Model (sorted by success rate)."""
    ordered = sorted(MODELS, key=lambda m: mstats.loc[m, "success_rate"], reverse=True)
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(13, 6))

    bottoms = np.zeros(len(ordered))
    for e in ALL_ERR:
        vals = []
        for m in ordered:
            sub = df_err[df_err.model == m]
            n   = max(len(sub), 1)
            vals.append((sub.classification_type == e).sum() / n)
        ax.bar(x, vals, bottom=bottoms, width=0.65,
               color=ERR_COLOR[e], label=ERR_SHORT.get(e, e), alpha=0.9)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([ms(m) for m in ordered], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Proportion of Classified Failures"); ax.set_ylim(0, 1.15)
    ax.set_title("C1 — Stacked Error Distribution per Model\n"
                 "(each bar = failure fingerprint — same height, different colors = different failure personality;\n"
                 "same color dominant everywhere = benchmark tests one failure mode only)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    save(fig, "C1_error_distribution")


def plot_C2():
    """Error Type × Model Heatmap."""
    mat = np.zeros((N_ERR, len(MODELS)))
    for j, m in enumerate(MODELS):
        sub = df_err[df_err.model == m]
        n   = max(len(sub), 1)
        for i, e in enumerate(ALL_ERR):
            mat[i, j] = (sub.classification_type == e).sum() / n

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(N_ERR))
    ax.set_yticklabels([ERR_SHORT.get(e, e) for e in ALL_ERR], fontsize=10)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([ms(m) for m in MODELS], rotation=25, ha="right", fontsize=9)
    ax.set_title("C2 — Error Type × Model Heatmap\n"
                 "(same high-value in 2–3 columns = shared architectural flaw; "
                 "unique column = model-specific weakness)",
                 fontsize=12, fontweight="bold")
    for i in range(N_ERR):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if mat[i,j] > 0.55 else "black")
    plt.colorbar(im, ax=ax, fraction=0.02, label="Fraction of Failures")
    save(fig, "C2_error_model_heatmap")


def plot_C3():
    """Global Error Type Frequency."""
    global_counts = {e: (df_err.classification_type == e).sum() for e in ALL_ERR}
    total = sum(global_counts.values())
    sorted_e  = sorted(ALL_ERR, key=lambda e: global_counts[e], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh([ERR_SHORT.get(e,e) for e in sorted_e],
                   [global_counts[e] for e in sorted_e],
                   color=[ERR_COLOR[e] for e in sorted_e], alpha=0.9, edgecolor="white")
    for bar, e in zip(bars, sorted_e):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{global_counts[e]/total*100:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Total count (all models × all tasks)")
    ax.set_xlim(0, max(global_counts.values()) * 1.25)
    ax.set_title("C3 — Global Error Type Frequency (All Models, All Tasks)\n"
                 "(if one bar > 70% of total → benchmark error diversity is poor)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.35)
    save(fig, "C3_global_error_freq")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP D — Radar Graphs
# ═══════════════════════════════════════════════════════════════════════════════

def plot_D1():
    """Multi-Model Error Fingerprint Radar — 8 overlaid polygons."""
    labels = [ERR_SHORT.get(e, e) for e in ALL_ERR]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for m in MODELS:
        sub = df_err[df_err.model == m]
        n   = max(len(sub), 1)
        vals = [(sub.classification_type == e).sum() / n for e in ALL_ERR]
        radar(ax, vals, labels, color=MODEL_COLOR[m], label=ms(m), alpha=0.12, lw=2.5)

    ax.set_title("D1 — Multi-Model Error Fingerprint Radar\n"
                 "(similar polygon shape = same failure personality; "
                 "same shape different size = same why, different how often)\n"
                 "⟵ most information-dense graph in the suite",
                 fontsize=10, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.12), fontsize=9)
    save(fig, "D1_error_radar_all")


def plot_D2():
    """Per-Model Multi-Metric Profile Radar."""
    metrics = ["Success\nRate", "Run\nConsistency", "Error\nEntropy",
               "Task\nCoverage", "Cross-Model\nUniqueness", "Dominant Error\nConc."]

    def entropy_norm(probs):
        p = np.array([v for v in probs if v > 0])
        if len(p) < 2: return 0.0
        return float(-np.sum(p * np.log(p)) / np.log(len(p)))

    solved_by = {m: set(sm[(sm.model == m) & (sm.rate > 0)]["task"].values) for m in MODELS}
    N = len(ALL_TASKS)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for m in MODELS:
        sub = df_err[df_err.model == m]
        n   = max(len(sub), 1)
        probs = [(sub.classification_type == e).sum() / n for e in ALL_ERR]
        others_solved = set().union(*(solved_by[m2] for m2 in MODELS if m2 != m))
        unique = len(solved_by[m] - others_solved) / max(len(solved_by[m]), 1)
        vals = [
            mstats.loc[m, "success_rate"] or 0,
            mstats.loc[m, "consistency"]  or 0,
            entropy_norm(probs),
            len(solved_by[m]) / N,
            unique,
            max(probs),
        ]
        radar(ax, vals, metrics, color=MODEL_COLOR[m], label=ms(m), alpha=0.12, lw=2.2)

    ax.set_title("D2 — Per-Model Multi-Metric Profile Radar\n"
                 "(specialist = strong on 1–2 axes; generalist = balanced polygon;\n"
                 "high 'Dominant Error Conc.' = fails for one reason only)",
                 fontsize=10, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.12), fontsize=9)
    save(fig, "D2_multi_metric_radar")


def plot_D3():
    """Small Multiple Radars — 8 panels, each model vs. global average."""
    labels = [ERR_SHORT.get(e, e) for e in ALL_ERR]
    n_total = max(len(df_err), 1)
    global_avg = [(df_err.classification_type == e).sum() / n_total for e in ALL_ERR]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax  = axes[idx]
        sub = df_err[df_err.model == m]
        n   = max(len(sub), 1)
        vals = [(sub.classification_type == e).sum() / n for e in ALL_ERR]
        radar(ax, global_avg, labels, color="gray",          alpha=0.08, lw=1.2)
        radar(ax, vals,       labels, color=MODEL_COLOR[m],  alpha=0.25, lw=2.5)
        ax.set_title(ms(m), fontsize=10, fontweight="bold", pad=10)

    for idx in range(len(MODELS), len(axes)):
        axes[idx].set_visible(False)

    from matplotlib.patches import Patch as _Patch
    fig.legend(handles=[_Patch(color="gray", label="Global average"),
                        _Patch(color="#555", label="This model")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("D3 — Per-Model Error Radar vs. Global Average\n"
                 "(outward bulge = disproportionate weakness relative to peers)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "D3_small_multiple_radars")


def plot_D4():
    """Task-Category Success Rate Radar (if ≥ 3 categories)."""
    if len(TASK_CATS) < 3:
        print("  D4 skipped — fewer than 3 task categories inferred")
        return

    labels = TASK_CATS
    cat_rate = {}
    for m in MODELS:
        sub = sm[sm.model == m].copy()
        sub["cat"] = sub["task"].map(task_cat)
        cat_rate[m] = [sub[sub.cat == c]["rate"].mean() or 0.0 for c in labels]
        # replace nan
        cat_rate[m] = [0.0 if (isinstance(v, float) and np.isnan(v)) else v
                       for v in cat_rate[m]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for m in MODELS:
        radar(ax, cat_rate[m], labels, color=MODEL_COLOR[m],
              label=ms(m), alpha=0.12, lw=2)

    ax.set_title("D4 — Task-Category Success Rate Radar\n"
                 "(each axis = one task domain; reveals domain-specific model strengths)",
                 fontsize=10, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.12), fontsize=9)
    save(fig, "D4_task_category_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP E — Cross-Model Error Agreement
# ═══════════════════════════════════════════════════════════════════════════════

def plot_E1():
    """Cross-Model Error Agreement Heatmap (8×8)."""
    n = len(MODELS)
    mat = np.zeros((n, n))
    for i, mi in enumerate(MODELS):
        for j, mj in enumerate(MODELS):
            if i == j:
                mat[i, j] = 1.0
                continue
            fi = set(df_err[df_err.model == mi]["task"].values)
            fj = set(df_err[df_err.model == mj]["task"].values)
            shared = fi & fj
            if not shared:
                continue
            agree = sum(
                1 for t in shared
                if (df_err[(df_err.model == mi) & (df_err.task == t)]["classification_type"].values[:1]
                    == df_err[(df_err.model == mj) & (df_err.task == t)]["classification_type"].values[:1]).all()
                and len(df_err[(df_err.model == mi) & (df_err.task == t)]) > 0
                and len(df_err[(df_err.model == mj) & (df_err.task == t)]) > 0
            )
            mat[i, j] = agree / len(shared)

    labs = [ms(m) for m in MODELS]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_title("E1 — Cross-Model Error Agreement Heatmap\n"
                 "(green = same root cause → training lineage signal; "
                 "red = different root cause → ideal ensemble pair)",
                 fontsize=12, fontweight="bold")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Fraction of shared failures with same error type")
    save(fig, "E1_cross_model_agreement")


def plot_E2():
    """Error Co-occurrence Heatmap (error types that co-occur on same task)."""
    mat = np.zeros((N_ERR, N_ERR))
    for task in ALL_TASKS:
        sub  = df_err[df_err.task == task]
        etypes = set(sub["classification_type"].unique())
        for i, ei in enumerate(ALL_ERR):
            for j, ej in enumerate(ALL_ERR):
                if i != j and ei in etypes and ej in etypes:
                    mat[i, j] += 1
    mat /= max(len(ALL_TASKS), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="Blues", vmin=0)
    short = [ERR_SHORT.get(e, e) for e in ALL_ERR]
    ax.set_xticks(range(N_ERR)); ax.set_yticks(range(N_ERR))
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_title("E2 — Error Co-occurrence Heatmap\n"
                 "(dark = error types that appear together in the same task across different models;\n"
                 "correlated difficulty signals → may be measuring the same capability)",
                 fontsize=11, fontweight="bold")
    for i in range(N_ERR):
        for j in range(N_ERR):
            if mat[i, j] > 0.005:
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if mat[i,j] > 0.3 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Co-occurrence rate per task")
    save(fig, "E2_error_cooccurrence")


def plot_E3():
    """Per-Error-Type Model Ranking (one bar panel per error type)."""
    n_cols = min(N_ERR, 2)
    n_rows = (N_ERR + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 5))
    axes = np.array(axes).flatten()

    for idx, e in enumerate(ALL_ERR):
        ax = axes[idx]
        rates = [(df_err[(df_err.model == m)]["classification_type"] == e).sum() /
                 max(len(df_err[df_err.model == m]), 1)
                 for m in MODELS]
        pairs = sorted(zip(MODELS, rates), key=lambda x: x[1])
        ms_ord, r_ord = zip(*pairs) if pairs else ([], [])
        ax.barh([ms(m) for m in ms_ord], r_ord,
                color=[MODEL_COLOR[m] for m in ms_ord], alpha=0.88)
        ax.set_title(ERR_SHORT.get(e, e), fontsize=11, fontweight="bold")
        ax.set_xlabel("Fraction of that model's failures")
        ax.grid(axis="x", alpha=0.35)

    for idx in range(N_ERR, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("E3 — Per-Error-Type Model Ranking\n"
                 "(bottom of each chart = model best at avoiding that error type → "
                 "use for error-aware task routing)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "E3_per_error_ranking")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP F — Task-Level Deep Structure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F1():
    """Task Difficulty Spectrum."""
    vals = task_n_solve.values

    def _cat(v):
        if v == 0:               return "Unsolvable (0)"
        elif v <= 2:             return "Very Hard (1–2)"
        elif v <= 4:             return "Hard (3–4)"
        elif v <= 6:             return "Medium (5–6)"
        else:                    return "Easy (7–8)"

    cats     = ["Unsolvable (0)", "Very Hard (1–2)", "Hard (3–4)", "Medium (5–6)", "Easy (7–8)"]
    cat_cols = ["#B71C1C",        "#E53935",          "#FF8F00",    "#43A047",       "#1B5E20"]
    cat_cnts = {c: sum(1 for v in vals if _cat(v) == c) for c in cats}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(vals, bins=range(0, len(MODELS)+2), align="left", rwidth=0.8,
                 color="#512DA8", edgecolor="white", alpha=0.9)
    axes[0].set_xlabel(f"Number of models solving a task (0–{len(MODELS)})")
    axes[0].set_ylabel("Number of tasks")
    axes[0].set_title("F1a — Difficulty Spectrum Histogram\n"
                       "(U-shape = poor calibration; flat = ideal)")
    axes[0].grid(axis="y", alpha=0.35)

    bars = axes[1].bar(range(len(cats)), [cat_cnts[c] for c in cats],
                       color=cat_cols, edgecolor="white", alpha=0.9)
    axes[1].set_xticks(range(len(cats)))
    axes[1].set_xticklabels(cats, rotation=15, ha="right")
    for bar, c in zip(bars, cats):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(cat_cnts[c]), ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("Number of tasks")
    axes[1].set_title("F1b — Difficulty Category Distribution")
    axes[1].grid(axis="y", alpha=0.35)

    fig.suptitle("F1 — Task Difficulty Spectrum  (144 tasks)\n"
                 "(left-skewed = benchmark too hard; right-skewed = too easy; "
                 "flat bar heights = well-calibrated)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "F1_task_difficulty_spectrum")


def plot_F2():
    """Task Clustering by Error Signature (dendrogram + heatmap)."""
    # Feature: per-task, error frequency across all models (N_ERR-dim)
    feat = np.zeros((N_TASKS, N_ERR))
    for j, task in enumerate(ALL_TASKS):
        sub = df_err[df_err.task == task]
        total = max(len(sub), 1)
        for i, e in enumerate(ALL_ERR):
            feat[j, i] = (sub.classification_type == e).sum() / total

    has_err = feat.sum(axis=1) > 0
    feat_s  = feat[has_err]
    tids_s  = [t for t, h in zip(ALL_TASKS, has_err) if h]
    n_s     = len(feat_s)
    print(f"  F2: {n_s} tasks with error classifications")

    if n_s < 4:
        print("  F2: too few tasks with errors, skipping"); return

    try:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        link = hierarchy.linkage(pdist(feat_s, metric="cosine"), method="ward")
        fig  = plt.figure(figsize=(14, 9))
        ax_d = fig.add_axes([0.04, 0.08, 0.16, 0.78])
        ax_h = fig.add_axes([0.21, 0.08, 0.65, 0.78])
        ax_c = fig.add_axes([0.87, 0.08, 0.02, 0.78])
        dend  = hierarchy.dendrogram(link, ax=ax_d, orientation="left",
                                     no_labels=True, color_threshold=0)
        order = dend["leaves"]
        im = ax_h.imshow(feat_s[order], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax_h.set_xticks(range(N_ERR))
        ax_h.set_xticklabels([ERR_SHORT.get(e, e) for e in ALL_ERR],
                             rotation=30, ha="right", fontsize=9)
        ax_h.set_ylabel(f"Tasks (n={n_s} classified; clustered by ward linkage)")
        ax_h.set_title("F2 — Task Clustering by Error Signature\n"
                        "(same cluster = same underlying capability tested; "
                        "use to detect redundant tasks)",
                        fontsize=11, fontweight="bold")
        plt.colorbar(im, cax=ax_c, label="Error fraction")
        ax_d.axis("off")
    except Exception as exc:
        print(f"  F2 fallback to PCA ({exc})")
        cov = np.cov(feat_s.T)
        _, vecs = np.linalg.eigh(cov + np.eye(cov.shape[0]) * 1e-10)
        pcs = feat_s @ vecs[:, -2:]
        fig, ax = plt.subplots(figsize=(9, 7))
        dom = [ALL_ERR[int(np.argmax(feat_s[i]))] if feat_s[i].sum() > 0 else "none"
               for i in range(n_s)]
        for e in ALL_ERR + ["none"]:
            idx = [k for k, d in enumerate(dom) if d == e]
            if idx:
                ax.scatter(pcs[idx, 0], pcs[idx, 1], s=40, alpha=0.7,
                           color=ERR_COLOR.get(e, "gray"),
                           label=ERR_SHORT.get(e, e) if e != "none" else "No error")
        ax.set_title("F2 — Task Clustering (PCA)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)

    save(fig, "F2_task_clustering")


def plot_F3():
    """Universal Hard vs. Differential Hard Scatter."""
    means = [task_mean.get(t, 0) for t in ALL_TASKS]
    varis = [task_var.get(t, 0) for t in ALL_TASKS]
    dom_err = []
    for t in ALL_TASKS:
        sub = df_err[df_err.task == t]
        dom_err.append(sub["classification_type"].value_counts().index[0]
                       if len(sub) else None)

    fig, ax = plt.subplots(figsize=(12, 8))
    no_e_x = [m for m, e in zip(means, dom_err) if e is None]
    no_e_y = [v for v, e in zip(varis, dom_err) if e is None]
    ax.scatter(no_e_x, no_e_y, s=22, color="#CCCCCC", alpha=0.5,
               label="No classification", zorder=2)

    for e in ALL_ERR:
        ex = [m for m, de in zip(means, dom_err) if de == e]
        ey = [v for v, de in zip(varis, dom_err) if de == e]
        ax.scatter(ex, ey, s=45, color=ERR_COLOR[e], alpha=0.78,
                   label=ERR_SHORT.get(e, e), zorder=3)

    mid_m = float(np.median(means))
    mid_v = float(np.median(varis))
    ax.axvline(mid_m, color="gray", lw=1.1, ls="--", alpha=0.55)
    ax.axhline(mid_v, color="gray", lw=1.1, ls="--", alpha=0.55)

    quads = [
        (0.01, 0.98, "top", "left",    "Universally Hard\n(all models fail)",        "#B71C1C"),
        (0.99, 0.98, "top", "right",   "Differentially Hard\n(only some solve it)",  "#E65100"),
        (0.01, 0.02, "bottom","left",  "Universally Easy\n(all models solve it)",    "#1B5E20"),
        (0.99, 0.02, "bottom","right", "Differentially Easy",                        "#0D47A1"),
    ]
    for tx, ty, va, ha, label, col in quads:
        ax.text(tx, ty, label, transform=ax.transAxes, va=va, ha=ha,
                fontsize=9, color=col, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    ax.set_xlabel("Mean Success Rate across 8 Models")
    ax.set_ylabel("Variance of Success Rate across Models")
    ax.set_title("F3 — Universal-Hard vs. Differential-Hard Task Split\n"
                 "(bottom-right = most diagnostic tasks: only THESE reveal real capability differences;\n"
                 "colored by dominant error type)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)
    save(fig, "F3_universal_vs_differential")


def plot_F4():
    """Error-to-Success Transition Map per model."""
    # run1 fail → run2 succeed = recovered
    # run1 fail → run2 fail   = persistent
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax   = axes[idx]
        sub  = sm[sm.model == m]
        rec  = defaultdict(int)
        pers = defaultdict(int)
        for _, row in sub.iterrows():
            if row.err_type is None: continue
            if row.r1 == 0 and row.r2 == 1:   rec[row.err_type]  += 1
            elif row.r1 == 0 and row.r2 == 0:  pers[row.err_type] += 1

        x = np.arange(N_ERR)
        w = 0.38
        ax.bar(x - w/2, [rec[e]  for e in ALL_ERR], width=w,
               color="#2E7D32", alpha=0.88, label="Recovered (R2)")
        ax.bar(x + w/2, [pers[e] for e in ALL_ERR], width=w,
               color="#B71C1C", alpha=0.88, label="Persistent")
        ax.set_xticks(x)
        ax.set_xticklabels([ERR_SHORT.get(e, e) for e in ALL_ERR],
                           rotation=28, ha="right", fontsize=8)
        ax.set_title(ms(m), fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.35)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    for idx in range(len(MODELS), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("F4 — Error-to-Success Transition Map\n"
                 "(green = recovered on run 2 → prompting/retry issue fixable;\n"
                 "red = persistent failure → knowledge gap, not a prompting problem)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "F4_error_to_success_transition")


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP G — Model Similarity & Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def plot_G1():
    """Model Similarity Dendrogram (clustered by 144-dim task success vector)."""
    feat = np.zeros((len(MODELS), N_TASKS))
    for i, m in enumerate(MODELS):
        for j in ALL_TASKS:
            row = sm[(sm.model == m) & (sm.task == j)]
            feat[i, j] = row.iloc[0]["rate"] if len(row) and not np.isnan(row.iloc[0]["rate"]) else 0

    try:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
        link = hierarchy.linkage(pdist(feat, metric="euclidean"), method="ward")
        fig, ax = plt.subplots(figsize=(11, 5))
        hierarchy.dendrogram(link, labels=[ms(m) for m in MODELS],
                             ax=ax, leaf_rotation=20, leaf_font_size=11)
        ax.set_ylabel("Distance (ward linkage)")
        ax.set_title("G1 — Model Similarity Dendrogram\n"
                     "(same cluster = similar capability profile → redundant in comparison;\n"
                     "distant = genuinely different → both valuable in benchmark)",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.35)
    except Exception as exc:
        print(f"  G1 fallback ({exc})")
        corr = np.corrcoef(feat)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
        labs = [ms(m) for m in MODELS]
        ax.set_xticks(range(len(MODELS))); ax.set_yticks(range(len(MODELS)))
        ax.set_xticklabels(labs, rotation=30, ha="right"); ax.set_yticklabels(labs)
        ax.set_title("G1 — Model Similarity (Pearson correlation of success vectors)")
        plt.colorbar(im, ax=ax, fraction=0.04)

    save(fig, "G1_model_dendrogram")


def plot_G2():
    """Ensemble Potential Radar."""
    metrics = ["Individual\nSuccess", "Max Pair\nCoverage Gain",
               "Error\nComplementarity", "Task\nCoverage", "Run\nConsistency"]

    solved_by = {m: set(sm[(sm.model == m) & (sm.rate > 0)]["task"].values) for m in MODELS}
    N = len(ALL_TASKS)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for m in MODELS:
        sub  = df_err[df_err.model == m]
        n    = max(len(sub), 1)
        probs_m = [(sub.classification_type == e).sum() / n for e in ALL_ERR]

        # Max pair coverage gain
        max_gain = max(
            (len(solved_by[m] | solved_by[m2]) / N - len(solved_by[m]) / N)
            for m2 in MODELS if m2 != m
        )

        # Error complementarity (avg distance to all other model error profiles)
        dists = []
        for m2 in MODELS:
            if m2 != m:
                sub2 = df_err[df_err.model == m2]
                n2   = max(len(sub2), 1)
                p2   = [(sub2.classification_type == e).sum() / n2 for e in ALL_ERR]
                dists.append(sum(abs(a - b) for a, b in zip(probs_m, p2)) / (2 * N_ERR))
        ec = float(np.mean(dists)) if dists else 0

        vals = [
            mstats.loc[m, "success_rate"] or 0,
            max_gain,
            ec,
            len(solved_by[m]) / N,
            mstats.loc[m, "consistency"] or 0,
        ]
        radar(ax, vals, metrics, color=MODEL_COLOR[m], label=ms(m), alpha=0.12, lw=2.5)

    ax.set_title("G2 — Ensemble Potential Radar\n"
                 "(high 'Max Pair Coverage Gain' + high 'Error Complementarity' on a weak model\n"
                 "= highest-value second member of a 2-model pipeline)",
                 fontsize=10, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.12), fontsize=9)
    save(fig, "G2_ensemble_potential_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
PLOTS = [
    ("A1  Ranked Success Rate (run-shading decomposition)",  plot_A1),
    ("A2  Model × Task Success Heatmap",                     plot_A2),
    ("A3  Pairwise Model Overlap Matrix",                    plot_A3),
    ("B1  Consistency vs. Success Rate",                     plot_B1),
    ("B2  Flip Matrix Heatmap",                              plot_B2),
    ("B3  Run 1 vs. Run 2 Scatter",                          plot_B3),
    ("B4  Task Instability Ranking",                         plot_B4),
    ("C1  Stacked Error Distribution",                       plot_C1),
    ("C2  Error Type × Model Heatmap",                       plot_C2),
    ("C3  Global Error Frequency",                           plot_C3),
    ("D1  Multi-Model Error Fingerprint Radar",              plot_D1),
    ("D2  Per-Model Multi-Metric Profile Radar",             plot_D2),
    ("D3  Small Multiple Radars (8 panels)",                 plot_D3),
    ("D4  Task-Category Success Radar",                      plot_D4),
    ("E1  Cross-Model Error Agreement (8×8)",                plot_E1),
    ("E2  Error Co-occurrence Heatmap",                      plot_E2),
    ("E3  Per-Error-Type Model Ranking",                     plot_E3),
    ("F1  Task Difficulty Spectrum",                         plot_F1),
    ("F2  Task Clustering by Error Signature",               plot_F2),
    ("F3  Universal vs. Differential Hard Scatter",          plot_F3),
    ("F4  Error-to-Success Transition Map",                  plot_F4),
    ("G1  Model Similarity Dendrogram",                      plot_G1),
    ("G2  Ensemble Potential Radar",                         plot_G2),
]

for name, fn in PLOTS:
    print(f"Plotting {name}…")
    try:
        fn()
    except Exception as exc:
        import traceback
        print(f"  ERROR in {name}: {exc}")
        traceback.print_exc()

print(f"\nAll 23 plots complete → {OUT_DIR}/")
