# CS639 NLP Project

Team 3: Kris Yang, Skylar Hou, Xinyu Jessica Wang, Handan Hu, Linxiang Jin

## Project Goals

Replicating and extending failure-mode analysis of LLM agents on long-horizon OS tasks, built on top of [AgentBench](https://github.com/THUDM/AgentBench) + [AgentRL](https://github.com/THUDM/AgentRL).

- Replicate AgentBench OS-task results across **8 models** (the *standard* benchmark).
- Run an **extended horizon study** on `claude-sonnet-4` and `gpt-5-mini`.
- Use LLM-as-a-judge to classify every failed run into one of seven failure modes (HORIZON-style taxonomy).

## Failure Taxonomy

All failed traces are classified into exactly one of:

1. **Environment Disturbance / Unable to Detect Change** — agent ignores feedback (permission denied, missing file, …) and acts on a false belief of success.
2. **Instruction Error** — misreads or ignores explicit task instructions / exception clauses.
3. **Catastrophic Forgetting** — drops an earlier constraint or fact still visible in context.
4. **False Assumptions** — acts on unverified beliefs about files, users, ownership, or state.
5. **Planning Errors (Sub-plan & Action)** — understands the task but picks a bad method/order.
6. **History Error Accumulation** — assumes a failed earlier step succeeded; cascades downstream errors.
7. **Memory Limitations** — relevant earlier information has dropped out of the context window.

The judge prompt and scoring code live in [results/classify_failures.py](results/classify_failures.py).

## Datasets in this Repo

| Path | What it is |
| --- | --- |
| [results/os-std-*.jsonl](results/) | Standard OS-Bench traces (144 tasks × 2 runs) for 8 models: `gpt-3.5-turbo`, `GPT-4`, `gpt-4o`, `o3-mini`, `o4-mini`, `gpt-5-mini`, `claude-3`, `claude-sonnet-3.7` |
| [results/failure_classifications-*.csv](results/) | Per-task failure-mode label for each std-bench failure |
| [OS-Task/run_{0,1,2}/{model}/os-std-aug-clear-filtered-{1..10}/](OS-Task/) | Extended horizon traces — 3 runs × 10 horizons × 46 base tasks for `claude-sonnet-4` and `gpt-5-mini` |
| [Extension_Failure_Classification_Results/{model}/runs_{1,4,7,10}*](Extension_Failure_Classification_Results/) | Failure-mode labels for the extended study at horizons 1, 4, 7, 10 |
| [final_analysis_figures/](final_analysis_figures/) | ~86 PNG figures produced by the analysis scripts below |
| [horizon_analysis/](horizon_analysis/) | Per-model failure-type-by-horizon plots |

## Analysis Scripts

Run from the project root after activating `agent-bench`. All figures land in `final_analysis_figures/`.

| Script | Output | Purpose |
| --- | --- | --- |
| [analyze_os_std.py](analyze_os_std.py) | `A1`–`A6` (6 figures) | Quick-look success / consistency / turns analysis on the 8-model std-bench data |
| [analyze_os_std_extended.py](analyze_os_std_extended.py) | `G1`–`G15` (15 figures) | Deeper std-bench breakdown: tool transitions, latency proxies, difficulty tiers, rolling success |
| [analyze_os_std_comprehensive.py](analyze_os_std_comprehensive.py) | `STD_A1`…`STD_G2` (23 figures, Groups A–G) | Full publication set: ranked success, flip matrices, error fingerprints, radars, clustering, dendrograms |
| [analyze_horizon.py](analyze_horizon.py) | `H01`–`H15` (15 figures) | Cross-horizon comparison of `claude-sonnet-4` vs `gpt-5-mini` (success curves, head-to-head, action / think-length distributions) |
| [analyze_horizon_extended.py](analyze_horizon_extended.py) | `HA1`…`HF3` (21 figures, Groups A–F) | Horizon × failure-taxonomy view: error survival, transitions, radar, efficiency curves, clustering |
| [horizon_analysis.py](horizon_analysis.py) | `failure_type_percentage_by_horizon_*.png` | Stacked failure-type composition by horizon, per model |
| [result_analysis_methods.py](result_analysis_methods.py) | Helper plots / tables | Cross-CSV summarisation utilities |
| [scripts/analyze_eval.py](scripts/analyze_eval.py) | stdout | Per-task breakdown and printable failure traces for any single results JSONL |
| [results/classify_failures.py](results/classify_failures.py) | `failure_classifications-*.csv` | Calls the OpenAI judge to label every failed trace |

## Prerequisites

- Docker (with Docker Compose v2)
- Conda (Python 3.12)
- OpenAI API key (set in `.env`)
- For analysis only: `pip install pandas numpy matplotlib`

## Setup

Run once after cloning:

```bash
bash setup.sh
```

This will:

1. Pull the `AgentBench` and `AgentRL` git submodules.
2. Create an `agent-bench` conda env (Python 3.12) and install dependencies.
3. Build the OS task Docker images (`local-os/default`, `local-os/packages`, `local-os/ubuntu`).

Then copy the example env file and add your API key:

```bash
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY
```

---

## Running an Evaluation

### 1. Activate the conda env

```bash
conda activate agent-bench
```

### 2. Start the task environment (Docker)

```bash
bash scripts/start_env.sh
```

This starts the AgentBench controller (port `5020`) and the OS task workers. Keep it running in a separate terminal or in the background.

To stop it:

```bash
bash scripts/start_env.sh --down
```

### 3. Run the evaluation

```bash
bash scripts/run_eval.sh
```

Common options:

| Flag                  | Default                     | Description                            |
| --------------------- | --------------------------- | -------------------------------------- |
| `-m` / `--model`      | `gpt-5-mini`                | Model name                             |
| `-u` / `--url`        | `https://api.openai.com/v1` | API base URL                           |
| `-j` / `--jobs`       | `8`                         | Concurrent sessions                    |
| `-c` / `--controller` | `http://localhost:5020/api` | Controller URL                         |
| `--task`              | `os-std`                    | Task set: `os-std` or `os-dev`         |
| `--resume`            | —                           | Resume from a previous output `.jsonl` |

Any additional flags are passed directly to `server_agent.py`. See all options with:

```bash
python vendor/AgentRL/examples/eval/server_agent.py --help
```

Examples:

```bash
# Run with GPT-4o-mini, 2 runs per task, temperature 0
bash scripts/run_eval.sh -m gpt-4o-mini -t 0 -n 2 --task os-std

# Run with GPT-5-mini, 2 runs per task, temperature 1
bash scripts/run_eval.sh -m gpt-5-mini -t 1 -n 2 --task os-std
```

Results are saved as `.jsonl` files under `results/`.

### 4. Inspect a single results file

```bash
bash scripts/check_results.sh results/<output-file>.jsonl
# or, for a deeper per-task / per-trace view:
python scripts/analyze_eval.py results/<output-file>.jsonl --show-traces --limit 5
```

### 5. Classify failures

```bash
python results/classify_failures.py \
    --input  results/os-std-gpt-4o-0.0-04080012.jsonl \
    --output results/failure_classifications-gpt4o.csv
```

### 6. Regenerate figures

```bash
python analyze_os_std.py
python analyze_os_std_extended.py
python analyze_os_std_comprehensive.py
python analyze_horizon.py
python analyze_horizon_extended.py
python horizon_analysis.py
```

All figures land in [final_analysis_figures/](final_analysis_figures/) (and a couple in [horizon_analysis/](horizon_analysis/)).

---

## Project Structure

```
.
├── setup.sh                           # one-time setup
├── .env.example                       # template for API keys and defaults
├── scripts/
│   ├── start_env.sh                   # start/stop Docker task environment
│   ├── run_eval.sh                    # run evaluation
│   ├── check_results.sh               # quick summary of a results file
│   └── analyze_eval.py                # per-task breakdown + trace printer
├── results/
│   ├── os-std-*.jsonl                 # raw std-bench traces (8 models)
│   ├── failure_classifications-*.csv  # judge-labelled failure modes
│   └── classify_failures.py           # GPT judge for the 7-class taxonomy
├── OS-Task/
│   └── run_{0,1,2}/{model}/os-std-aug-clear-filtered-{1..10}/
│                                      # extended horizon traces (3 runs × 10 horizons)
├── Extension_Failure_Classification_Results/
│   └── {claude-sonnet-4,gpt-5-mini}/runs_{1,4,7,10}_failure_results.csv
├── analyze_os_std.py                  # std-bench: 6 figures
├── analyze_os_std_extended.py         # std-bench: 15 figures
├── analyze_os_std_comprehensive.py    # std-bench: 23 figures (Groups A–G)
├── analyze_horizon.py                 # horizon study: 15 figures
├── analyze_horizon_extended.py        # horizon × taxonomy: 21 figures (Groups A–F)
├── horizon_analysis.py                # failure-type-by-horizon stacked plots
├── result_analysis_methods.py         # shared analysis helpers
├── final_analysis_figures/            # ~86 PNGs produced by the scripts above
├── horizon_analysis/                  # per-model horizon failure-type plots
└── vendor/
    ├── AgentBench/                    # task definitions + Docker controller
    └── AgentRL/                       # eval harness (server_agent.py)
```
