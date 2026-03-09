# CS639 NLP Project

**Team 3**: Kris Yang, Skylar Hou, Xinyu Jessica Wang, Handan Hu, Linxiang Jin

Replicating and analyzing long-horizon OS task failures using [AgentBench](https://github.com/THUDM/AgentBench) + [AgentRL](https://github.com/THUDM/AgentRL).

## Project Goals

- Replicate AgentBench OS task results with multiple models
- Categorize execution traces into failure modes (per the HORIZON taxonomy)
- Analyze where and why agents fail on long-horizon OS tasks

## Prerequisites

- Docker (with Docker Compose)
- Conda (Python 3.10+)
- OpenAI API key (set in .env)

## Setup

Run once after cloning:

```bash
bash setup.sh
```

This will:

1. Pull the `AgentBench` and `AgentRL` git submodules
2. Create a `agent-bench` conda env (Python 3.12) and install dependencies
3. Build the OS task Docker images (`local-os/default`, `local-os/packages`, `local-os/ubuntu`)

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

### 4. Check results

```bash
bash scripts/check_results.sh results/<output-file>.jsonl
```

---

## Defaults via `.env`

Instead of passing flags every time, set defaults in `.env`:

```bash
MODEL=gpt-4o
BASE_URL=https://api.openai.com/v1
TASK=os-std
JOBS=16
```

---

## Project Structure

```
.
├── setup.sh                  # one-time setup
├── .env.example              # template for API keys and defaults
├── scripts/
│   ├── start_env.sh          # start/stop Docker task environment
│   ├── run_eval.sh           # run evaluation
│   └── check_results.sh      # summarise a results file
├── results/                  # evaluation output (gitignored)
└── vendor/
    ├── AgentBench/           # task definitions + Docker controller
    └── AgentRL/              # eval harness (server_agent.py)
```
