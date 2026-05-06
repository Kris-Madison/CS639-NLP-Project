import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Local folder
base_dir = Path("/Users/jinlinxiang/Desktop/gpt-5-mini")

files = {
    1: "runs_1_failure_results.csv",
    4: "runs_4_failure_results.csv",
    7: "runs_7_failure_results.csv",
    10: "runs_10_failure_results.csv",
}

# Same category order and colors as your model-level figure
failure_types = [
    "Environment Disturbance / Unable to Detect Change",
    "Instruction Error",
    "Catastrophic Forgetting",
    "False Assumptions",
    "Planning Errors (Sub-plan & Action)",
    "History Error Accumulation",
    "Memory Limitations",
]

colors = {
    "Environment Disturbance / Unable to Detect Change": "#1f77b4",  # blue
    "Instruction Error": "#ff7f0e",                                  # orange
    "Catastrophic Forgetting": "#2ca02c",                            # green
    "False Assumptions": "#d62728",                                  # red
    "Planning Errors (Sub-plan & Action)": "#9467bd",                # purple
    "History Error Accumulation": "#8c564b",                         # brown
    "Memory Limitations": "#e377c2",                                 # pink
}

all_rows = []

for horizon, filename in files.items():
    path = base_dir / filename
    df = pd.read_csv(path)

    if "classification_type" not in df.columns:
        raise ValueError(f"{filename} does not contain 'classification_type' column")

    df["horizon"] = horizon
    all_rows.append(df[["horizon", "classification_type"]])

data = pd.concat(all_rows, ignore_index=True)

# Count by horizon and failure type
counts = (
    data.groupby(["horizon", "classification_type"])
    .size()
    .reset_index(name="count")
)

pivot_counts = counts.pivot(
    index="horizon",
    columns="classification_type",
    values="count"
).fillna(0)

# Force all categories to appear in the same order
pivot_counts = pivot_counts.reindex(columns=failure_types, fill_value=0)
pivot_counts = pivot_counts.loc[[1, 4, 7, 10]]

# Convert to percentages
percentages = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

bottom = [0] * len(percentages.index)
x_labels = [str(h) for h in percentages.index]

for_type_order = failure_types

for error_type in failure_types:
    values = percentages[error_type].values
    ax.bar(
        x_labels,
        values,
        bottom=bottom,
        label=error_type,
        color=colors[error_type],
    )
    bottom = [b + v for b, v in zip(bottom, values)]

ax.set_xlabel("Intrinsic Horizon Length")
ax.set_ylabel("Percentage (%)")
ax.set_title("Failure Type Percentage by Horizon Length (Model: gpt-5-mini)")
ax.set_ylim(0, 100)

ax.legend(
    title="Classification Type",
    bbox_to_anchor=(1.03, 1),
    loc="upper left",
)

plt.tight_layout()

output_path = base_dir / "failure_type_percentage_by_horizon.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved figure to:", output_path)
print("\nPercentage table:")
print(percentages.round(2))