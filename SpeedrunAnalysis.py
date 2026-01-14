import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import numpy as np
import glob
import os

# ==========================
# Load & combine data
# ==========================

folder = "gamefiles"
game_csv_files = glob.glob(os.path.join(folder, "*.csv"))

game_dataframes = []
for file_path in game_csv_files:
    game_df = pd.read_csv(file_path)
    game_df["Game"] = os.path.basename(file_path)
    game_dataframes.append(game_df)

combined_df = pd.concat(game_dataframes, ignore_index=True)

# ==========================
# Add unix time & reorder
# ==========================

combined_df["Date (unix)"] = pd.to_datetime(combined_df["Date"]).apply(
    lambda dt: dt.timestamp()
)

column_order = [
    "Date",
    "Date (unix)",
    "Id",
    "Players",
    "Time",
    "Seconds",
    "Link",
    "Game",
]
combined_df = combined_df.reindex(columns=column_order)

# Relative time (seconds since previous run for that game)
combined_df["relative time"] = (
    combined_df.groupby("Game")["Seconds"].diff().round(2)
)

combined_df.to_csv("game_data.csv", index=False)

# ==========================
# Basic stats
# ==========================

analysis_df = combined_df.copy()
games_grouped = analysis_df.groupby("Game")
number_of_games = len(games_grouped)
number_of_runs = len(analysis_df)
number_of_unique_players = analysis_df["Players"].nunique()

# ==========================
# Data preprocessing
# ==========================

analysis_df["Seconds"] = pd.to_numeric(analysis_df["Seconds"], errors="coerce")
analysis_df["Date (unix)"] = pd.to_numeric(
    analysis_df["Date (unix)"], errors="coerce"
)

first_run_indices = analysis_df.groupby("Game")["Date (unix)"].idxmin()
first_run_dates = (
    analysis_df.loc[first_run_indices, ["Game", "Date (unix)"]]
    .set_index("Game")["Date (unix)"]
)

analysis_df["first_run_date"] = analysis_df["Game"].map(first_run_dates)
analysis_df["days_since_first_run"] = (
    analysis_df["Date (unix)"] - analysis_df["first_run_date"]
) / 86400  # seconds per day

analysis_df.to_csv("analysis.csv", index=False)

# ==========================
# Legend-friendly game labels
# ==========================

analysis_df["GameLabel"] = analysis_df["Game"].str.replace(".csv", "", regex=False)

game_name_map = {
    "crash4Any%": "Crash Bandicoot 4 (Any%)",
    "doometernalAny%(Restricted)": "Doom Eternal - Any% (Restricted)",
    "ghostoftsushimaAny%(NG+)": "Ghost of Tsushima - NG+",
    "hadesAnyHeatSeeded": "Hades (Seeded Any Heat)",
    "ittakestwoAny%": "It Takes Two (Any%)",
    "halflifealyx": "Half-Life: Alyx",
    "ori_wotwAllChallenges": "Ori and the Will of the Wisps – All Challenges",
    "rac_raNG+": "Ratchet & Clank: Rift Apart – NG+",
    "re3rAny%(PC)": "Resident Evil 3 Remake – Any% (PC)",
    "spider-manmilesmoralesAny%": "Spider-Man: Miles Morales – Any%",
}

analysis_df["GameLabel"] = analysis_df["GameLabel"].replace(game_name_map)

# ==========================
# Scatterplot: all games
# ==========================

plt.style.use("ggplot")
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=list(plt.cm.tab10.colors) + list(plt.cm.Set3.colors)
)

fig, ax = plt.subplots(figsize=(14, 6))

for game_label, game_subset in analysis_df.groupby("GameLabel"):
    ax.scatter(
        game_subset["days_since_first_run"],
        game_subset["Seconds"],
        s=20,
        alpha=0.7,
        label=game_label,
    )

ax.set_xlabel("Days since first run", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Run times vs. days since first run", fontsize=14)

ax.legend(
    title="Game",
    fontsize=10,
    title_fontsize=11,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.0,
    ncol=1,
    frameon=True,
    facecolor="white",
    edgecolor="gray",
)

plt.tight_layout()

# ==========================
# Player base bar chart
# ==========================

unique_players_per_game = (
    analysis_df.groupby("GameLabel")["Players"]
    .nunique()
    .sort_values(ascending=False)
)

fig2, ax2 = plt.subplots(figsize=(12, 6))
unique_players_per_game.plot(kind="bar", ax=ax2)

ax2.set_xlabel("Game", fontsize=12)
ax2.set_ylabel("Number of unique players", fontsize=12)
ax2.set_title("Player base by game (unique runners)", fontsize=14)
ax2.tick_params(axis="x", labelrotation=45)
for label in ax2.get_xticklabels():
    label.set_horizontalalignment("right")

plt.tight_layout()

# ==========================
# Total runs per game bar chart
# ==========================

total_runs_per_game = (
    analysis_df["GameLabel"]
    .value_counts()
    .sort_values(ascending=False)
)

fig3, ax3 = plt.subplots(figsize=(12, 6))
total_runs_per_game.plot(kind="bar", ax=ax3, color="indianred")

ax3.set_xlabel("Game", fontsize=12)
ax3.set_ylabel("Total number of runs", fontsize=12)
ax3.set_title("Total World Record Submissions since 2020", fontsize=14)
ax3.tick_params(axis="x", labelrotation=45)
for label in ax3.get_xticklabels():
    label.set_horizontalalignment("right")

plt.tight_layout()

# ==========================
# Exponential decay fit for most popular game
# ==========================

# Most popular game by total runs
runs_per_game = analysis_df["GameLabel"].value_counts()
most_popular_game = runs_per_game.idxmax()

# Filter to that game and compute daily best times
popular_game_df = analysis_df[analysis_df["GameLabel"] == most_popular_game].copy()
popular_game_df = popular_game_df.dropna(
    subset=["days_since_first_run", "Seconds"]
)

daily_best_times = (
    popular_game_df
    .groupby("days_since_first_run", as_index=False)["Seconds"]
    .min()
    .sort_values(by="days_since_first_run")
)

days_since_first_run_full = daily_best_times["days_since_first_run"].values.astype(
    float
)
best_times_full_seconds = daily_best_times["Seconds"].values.astype(float)

# Use only the "stable phase" to fit (after day 50)
is_stable_phase = days_since_first_run_full > 50
if is_stable_phase.sum() >= 5:
    days_for_fit = days_since_first_run_full[is_stable_phase]
    best_times_for_fit = best_times_full_seconds[is_stable_phase]
else:
    days_for_fit = days_since_first_run_full
    best_times_for_fit = best_times_full_seconds


def exponential_decay_model(days, amplitude, decay_rate, asymptote):
# Exponential decay: time(days) = amplitude * exp(-decay_rate * days) + asymptote.
    return amplitude * np.exp(-decay_rate * days) + asymptote


best_time_ever = best_times_for_fit.min()
worst_relevant_time = best_times_for_fit.max()

# Initial parameter guesses
initial_amplitude = worst_relevant_time - best_time_ever
initial_decay_rate = 0.001
initial_asymptote = best_time_ever

# Bounds for parameters:
#   amplitude >= 0
#   0 <= decay_rate <= 1
#   asymptote near current best (within +/- 300s, but not below 0)
asymptote_lower_bound = max(0.0, best_time_ever - 300.0)
asymptote_upper_bound = best_time_ever + 300.0

# Weights: later points get more weight
fit_weights = days_for_fit / days_for_fit.max()
fit_weights = np.clip(fit_weights, 0.2, None)
fit_sigma = 1.0 / fit_weights

fit_params, covariance_matrix = curve_fit(
    exponential_decay_model,
    days_for_fit,
    best_times_for_fit,
    p0=[initial_amplitude, initial_decay_rate, initial_asymptote],
    sigma=fit_sigma,
    absolute_sigma=False,
    bounds=(
        [0.0, 0.0, asymptote_lower_bound],
        [np.inf, 1.0, asymptote_upper_bound],
    ),
    maxfev=20000,
)

amplitude, decay_rate, asymptote = fit_params

# Predict best time 365 days after the last recorded run
last_recorded_day = days_since_first_run_full.max()
prediction_day = last_recorded_day + 365.0
predicted_best_time = exponential_decay_model(
    prediction_day, amplitude, decay_rate, asymptote
)

# Curve for plotting (0 to prediction_day)
curve_days = np.linspace(0, prediction_day, 400)
curve_times = exponential_decay_model(curve_days, amplitude, decay_rate, asymptote)

# Plot: all daily bests, fitted curve, and prediction point
plt.style.use("ggplot")
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(
    days_since_first_run_full,
    best_times_full_seconds,
    s=35,
    alpha=0.7,
    color="tab:orange",
    label=f"{most_popular_game} daily bests",
)

ax.plot(
    curve_days,
    curve_times,
    linewidth=2,
    color="dimgray",
    linestyle="--",
    label="Exponential decay fit",
)

ax.scatter(
    [prediction_day],
    [predicted_best_time],
    color="red",
    edgecolor="black",
    s=80,
    zorder=5,
    label="+1 year prediction",
)

ax.set_xlabel("Days since first run", fontsize=12)
ax.set_ylabel("Best run time (seconds)", fontsize=12)
ax.set_title(
    f"Best-time trend and +1-year prediction for {most_popular_game}"
)

ax.set_xlim(0, prediction_day + 100)
ax.legend()
plt.tight_layout()

# ==========================
# Print data function
# ==========================

def print_data():
    print(
    f"Contains {number_of_games} games run {number_of_runs} times "
    f"by {number_of_unique_players} runners."
    )
    print(f"Most popular game: {most_popular_game}")
    print(
    f"Predicted best time +1 year after last run: {predicted_best_time:.2f} seconds"
    )

# ==========================
# Trend analysis of all games
# ==========================

def trend_analysis(dataframe, game_filename=None):

    # Filter dataframe
    if game_filename:
        game_df = dataframe[dataframe["Game"] == game_filename].copy()
    else:
        game_df = dataframe.copy()

    # Ensure sorted chronologically
    game_df = game_df.sort_values("days_since_first_run").reset_index(drop=True)

    # ==========================
    # Improvement per day between runs
    # ==========================
    improvements_per_day = []

    for i in range(1, len(game_df)):
        previous_time = game_df.iloc[i - 1]["Seconds"]
        current_time = game_df.iloc[i]["Seconds"]
        delta_time = previous_time - current_time

        previous_day = game_df.iloc[i - 1]["days_since_first_run"]
        current_day = game_df.iloc[i]["days_since_first_run"]
        delta_days = current_day - previous_day

        if delta_days > 0:
            improvements_per_day.append(delta_time / delta_days)

    # Summary of improvements
    if improvements_per_day:
        avg_improvement = np.mean(improvements_per_day)
        std_improvement = np.std(improvements_per_day)

        print(f"\n=== Improvement Rate Statistics ===")
        print(f"Mean improvement rate: {avg_improvement:.4f} sec/day")
        print(f"Standard deviation:    {std_improvement:.4f} sec/day\n")

        # Compare first half vs second half (for diminishing returns)
        split_index = len(improvements_per_day) // 2
        if split_index > 5:
            first_half = improvements_per_day[:split_index]
            second_half = improvements_per_day[split_index:]

            avg_first = np.mean(first_half)
            avg_second = np.mean(second_half)

            print(f"First half avg improvement:  {avg_first:.4f} sec/day")
            print(f"Second half avg improvement: {avg_second:.4f} sec/day")

            slowdown_factor = (avg_first / avg_second) if avg_second != 0 else float("inf")
            percent_reduction = (
                (avg_first - avg_second) / avg_first * 100 if avg_first != 0 else 0
            )

            print(f"Slowdown factor: {slowdown_factor:.2f}×")
            print(f"Percent reduction from early to late runs: {percent_reduction:.1f}%\n")

    # ==========================
    # Pearson Correlation
    # ==========================
    correlation, p_value = pearsonr(
        game_df["days_since_first_run"],
        game_df["Seconds"]
    )

    print("=== Correlation Analysis ===")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"P-value:            {p_value:.4e}")

    if correlation < -0.3 and p_value < 0.05:
        print("Significant improvement trend.\n")
    elif correlation < -0.2 and p_value < 0.05:
        print("Moderate improvement trend.\n")
    else:
        print("No statistically clear trend.\n")

    # ==========================
    # Power-law diminishing returns
    # ==========================
    positive_day_df = game_df[game_df["days_since_first_run"] > 0]

    if len(positive_day_df) > 10:
        try:
            log_days = np.log(positive_day_df["days_since_first_run"])
            log_times = np.log(positive_day_df["Seconds"])

            slope, intercept = np.polyfit(log_days, log_times, 1)

            print("=== Power-Law Fit (Diminishing Returns) ===")
            if abs(slope) < 0.3:
                print("Strong diminishing returns.")
            elif abs(slope) < 0.7:
                print("Moderate diminishing returns.")
            else:
                print("Nearly linear improvement.")
            print()

            # Plot power-law relationship
            fig_power, ax_power = plt.subplots(figsize=(10, 6))

            ax_power.scatter(
                positive_day_df["days_since_first_run"],
                positive_day_df["Seconds"],
                alpha=0.6,
                s=50,
                label="Data",
                color="steelblue",
            )

            days_fit_line = np.linspace(
                positive_day_df["days_since_first_run"].min(),
                positive_day_df["days_since_first_run"].max(),
                100,
            )
            times_fit_line = np.exp(intercept) * (days_fit_line ** slope)

            ax_power.plot(
                days_fit_line,
                times_fit_line,
                "r-",
                linewidth=2.5,
                label=f"Power-Law Fit: y = {np.exp(intercept):.2f} × x^({slope:.3f})",
            )

            ax_power.set_xscale("log")
            ax_power.set_yscale("log")
            ax_power.set_xlabel("Days since first run (log scale)", fontsize=12)
            ax_power.set_ylabel("Best time (seconds, log scale)", fontsize=12)
            ax_power.set_title("Power-Law Fit: Improvement Over Time", fontsize=14)

            ax_power.legend()
            ax_power.grid(True, alpha=0.3, which="both")

            plt.tight_layout()
            plt.show()
            plt.close(fig_power)

        except Exception:
            pass

# TO SEE ALL 5 GRAPHS AND DATA UNCOMMENT LAST LINE 3 LINES.
# ALL 5 WILL OPEN IN SEPARATE WINDOWS AT THE SAME TIME.

# print_data()
# trend_analysis(analysis_df)
# plt.show()