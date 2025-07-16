
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import sys

read_results_dir = "VOO_bb_bull_study"
results_dir = read_results_dir
save_results_dir = "VOO_bb_bull_study_post"

# List all CSV files in the directory
csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

# Initialize a list to store extracted data
data_list = []

pattern = re.compile(
     #r"port_return_B_(\d+\.?\d*)_new_invest0_k_2_q_(\d+\.?\d*)_lambda1_(\d+\.?\d*)_freq_(\d+)B_tfreq_(\d+)B_H_scale_1_solver_type(.*?)\.csv"
     r"port_return_B_(\d+\.?\d*)_new_invest(\d+\.?\d*)_k_2_q_(\d+\.?\d*)_lambda1_(\d+\.?\d*)_freq_(\d+)B_tfreq_(\d+)B_H_scale_1_solver_type(.*?)\.csv"
)

# Read all dataframes with q value and solver type from filenames
data_list = []
for file in csv_files:
    match = pattern.match(file)
    if match:
        full_path = os.path.join(results_dir, file)
        df = pd.read_csv(full_path)
        q_value = file.split("q_")[1].split("_")[0]
        solver = "QAOA" if "QAOA" in file else "Classic"
        df["q"] = float(q_value)
        df["Solver"] = solver
        data_list.append(df)

# Concatenate all dataframes
all_data = pd.concat(data_list, ignore_index=True)

# Rename return column for each dataframe and keep only relevant columns
for df in data_list:
    df.rename(columns={"0": "Return"}, inplace=True)

# Build a combined DataFrame with one return series per configuration
combined_df = pd.DataFrame()
for df in data_list:
    label = f"q={df['q'].iloc[0]} | {df['Solver'].iloc[0]}"
    combined_df[label] = df["Return"].reset_index(drop=True)

# --- Benchmark file --- #

benchmark_df = None  # Initialize
# Load the benchmark file
for file in csv_files:
    if file.strip().lower() == "benchmarks_returns.csv":
        full_path = os.path.join(results_dir, file)
        benchmark_df = pd.read_csv(full_path)


# Redefine styles for specific assignment
portfolio_styles = {
    "Classic": {"marker": "s", "linestyle": None, "linewidth": 2},  # Square
    "QAOA": {"marker": "o", "linestyle": None, "linewidth": 2},     # Dot
}
benchmark_styles = {
    "VOO": {"color": "black", "linestyle": "--", "linewidth": 2, "marker": None},
    "AAPL": {"color": "orange", "linestyle": "--", "linewidth": 1, "marker": None},
    "MSFT": {"color": "blue", "linestyle": "--", "linewidth": 1, "marker": None},
    "AMZN": {"color": "magenta", "linestyle": "--", "linewidth": 1, "marker": None},
}

# Ensure all benchmark styles have linewidth defined
for key in benchmark_styles:
    benchmark_styles[key].setdefault("linewidth", 2)

# Combine benchmark data with the previous optimizer result trajectories
# First, make sure the indices align by trimming or reindexing if necessary
min_len = min(len(combined_df), len(benchmark_df))
trimmed_combined_df = combined_df.iloc[:min_len].copy()
trimmed_benchmark_df = benchmark_df.iloc[:min_len].copy()

# Start plotting again with fully defined styles
plt.figure(figsize=(14, 7))

# Plot optimized portfolios
for column in trimmed_combined_df.columns:
    solver_type = "QAOA" if "QAOA" in column else "Classic"
    style = portfolio_styles[solver_type]
    plt.plot(
        trimmed_combined_df.index,
        trimmed_combined_df[column],
        label=column,
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        markersize=6
    )

# Plot benchmarks
for col in trimmed_benchmark_df.columns:
    style = benchmark_styles[col]
    plt.plot(
        trimmed_benchmark_df.index,
        trimmed_benchmark_df[col],
        label=f"Benchmark: {col}",
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        marker=style["marker"],
        markersize=6 if style["marker"] else 0
    )

# Formatting
plt.title("Optimized Portfolios vs VOO, AAPL, MSFT, AMZN")
plt.xlabel("Rebalance month")
plt.ylabel("Return (%)")
plt.legend(loc="best", fontsize=9, title="Configuration")
plt.grid(True)
plt.tight_layout()
#plt.show()

# Ensure output folder exists
os.makedirs(save_results_dir, exist_ok=True)
#png_path = os.path.join(save_results_dir, f"_VOO_bb_bull_study_post_new_invest.png")
#pdf_path = os.path.join(save_results_dir, f"_VOO_bb_bull_study_post.pdf")
pdf_path = os.path.join(save_results_dir, f"_VOO_bb_bull_study_post_new_invest.pdf")

# Save as PNG
#plt.savefig(png_path, dpi=300)

# Save as PDF
plt.savefig(pdf_path, format='pdf')
plt.show()
#sys.exit("exit pattern")