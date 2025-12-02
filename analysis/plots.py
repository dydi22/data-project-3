"""
Generate visualizations for NBA play-by-play analysis.

Creates:
1. Comeback probability heatmap
2. Clutch performance distribution
3. Timeout effectiveness analysis
4. Overtime predictors
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_config import setup_logging, get_logger

# Set up logging
logger = setup_logging(log_level="INFO")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10


def load_data(db_file: Path):
    """Load data from DuckDB."""
    conn = duckdb.connect(str(db_file))
    
    # Load tables, handling cases where they might not exist
    try:
        scoring_runs = conn.execute("SELECT * FROM scoring_runs").fetchdf()
    except:
        scoring_runs = pd.DataFrame()
    
    try:
        droughts = conn.execute("SELECT * FROM droughts").fetchdf()
    except:
        droughts = pd.DataFrame()
    
    try:
        foul_freq = conn.execute("SELECT * FROM foul_frequencies").fetchdf()
    except:
        foul_freq = pd.DataFrame()
    
    try:
        pace_metrics = conn.execute("SELECT * FROM pace_metrics").fetchdf()
    except:
        pace_metrics = pd.DataFrame()
    
    try:
        momentum = conn.execute("SELECT * FROM momentum").fetchdf()
    except:
        momentum = pd.DataFrame()
    
    try:
        comeback_prob = conn.execute("SELECT * FROM comeback_probability").fetchdf()
    except:
        comeback_prob = pd.DataFrame()
    
    try:
        clutch_perf = conn.execute("SELECT * FROM clutch_performance").fetchdf()
    except:
        clutch_perf = pd.DataFrame()
    
    try:
        timeout_effect = conn.execute("SELECT * FROM timeout_effectiveness").fetchdf()
    except:
        timeout_effect = pd.DataFrame()
    
    try:
        overtime_pred = conn.execute("SELECT * FROM overtime_predictors").fetchdf()
    except:
        overtime_pred = pd.DataFrame()
    
    conn.close()
    
    return scoring_runs, droughts, foul_freq, pace_metrics, momentum, comeback_prob, clutch_perf, timeout_effect, overtime_pred


def plot_scoring_run_distribution(scoring_runs: pd.DataFrame, output_dir: Path):
    """Plot distribution of scoring run lengths."""
    print("Creating scoring run distribution plot...")
    
    if scoring_runs.empty:
        print("  ⚠ No scoring runs data available")
        return
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_desc = fig.add_subplot(gs[1, :])
    ax_desc.axis('off')
    
    # Histogram
    ax1.hist(
        scoring_runs["run_length"],
        bins=range(1, scoring_runs["run_length"].max() + 2),
        edgecolor="black",
        alpha=0.7,
        color="steelblue"
    )
    ax1.set_xlabel("Scoring Run Length (consecutive scores)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of Scoring Run Lengths", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Add description
    description = (
        "Calculation: A scoring run is defined as consecutive possessions where the same team scores. "
        "For each game, we track sequences of made shots/free throws by the same team. "
        f"The average run length is {scoring_runs['run_length'].mean():.2f} possessions. "
        "Most runs are short (1-2 scores), indicating that sustained offensive momentum is rare in NBA games."
    )
    ax_desc.text(0.05, 0.5, description, fontsize=10, verticalalignment='center', 
                 wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Box plot by team (top 10 teams by total runs)
    top_teams = scoring_runs["team"].value_counts().head(10).index
    top_teams_runs = scoring_runs[scoring_runs["team"].isin(top_teams)]
    
    if not top_teams_runs.empty:
        sns.boxplot(
            data=top_teams_runs,
            x="team",
            y="run_length",
            ax=ax2
        )
        ax2.set_xlabel("Team", fontsize=12)
        ax2.set_ylabel("Scoring Run Length", fontsize=12)
        ax2.set_title("Scoring Run Lengths by Team (Top 10)", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    output_path = output_dir / "scoring_run_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_drought_length_by_team(droughts: pd.DataFrame, output_dir: Path):
    """Plot average drought length by team."""
    print("Creating drought length by team plot...")
    
    if droughts.empty:
        print("  ⚠ No droughts data available")
        return
    
    # Calculate average drought length by team
    team_droughts = droughts.groupby("team")["drought_length"].agg([
        "mean",
        "median",
        "std",
        "count"
    ]).reset_index()
    team_droughts = team_droughts.sort_values("mean", ascending=False)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_desc = fig.add_subplot(gs[1, :])
    ax_desc.axis('off')
    
    # Bar plot of average drought length
    top_teams = team_droughts.head(15)
    ax1.barh(top_teams["team"], top_teams["mean"], color="coral", alpha=0.7)
    ax1.set_xlabel("Average Drought Length (possessions)", fontsize=12)
    ax1.set_ylabel("Team", fontsize=12)
    ax1.set_title("Average Scoring Drought Length by Team", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    
    # Box plot for top teams
    top_team_names = top_teams["team"].head(10).tolist()
    top_teams_data = droughts[droughts["team"].isin(top_team_names)]
    
    if not top_teams_data.empty:
        sns.boxplot(
            data=top_teams_data,
            x="team",
            y="drought_length",
            ax=ax2
        )
        ax2.set_xlabel("Team", fontsize=12)
        ax2.set_ylabel("Drought Length (possessions)", fontsize=12)
        ax2.set_title("Drought Length Distribution by Team", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")
    
    # Add description
    description = (
        "Calculation: A scoring drought is defined as consecutive possessions where a team does not score. "
        "We track sequences of missed shots, turnovers, and other non-scoring events. "
        f"The average drought length is {droughts['drought_length'].mean():.2f} possessions. "
        "Teams with longer average droughts struggle more with offensive consistency."
    )
    ax_desc.text(0.05, 0.5, description, fontsize=10, verticalalignment='center', 
                 wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / "drought_length_by_team.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_foul_frequency_heatmap(foul_freq: pd.DataFrame, output_dir: Path):
    """Plot heatmap of foul frequency by minute."""
    print("Creating foul frequency heatmap...")
    
    if foul_freq.empty:
        print("  ⚠ No foul frequency data available")
        return
    
    # Pivot to create heatmap data
    heatmap_data = foul_freq.pivot_table(
        values="foul_count",
        index="period",
        columns="game_minute",
        aggfunc="mean",
        fill_value=0
    )
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax_desc = fig.add_subplot(gs[1, 0])
    ax_desc.axis('off')
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Average Fouls per Game"},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_xlabel("Game Minute", fontsize=12)
    ax.set_ylabel("Period", fontsize=12)
    ax.set_title("Foul Frequency Heatmap by Period and Game Minute", fontsize=14, fontweight="bold")
    
    # Add description
    description = (
        "Calculation: Foul frequency is calculated by counting foul events (event_msg_type = 6) "
        "grouped by period and game minute. The heatmap shows average fouls per game at each minute. "
        "Darker colors indicate higher foul frequency. Patterns reveal strategic fouling, particularly "
        "at the end of quarters when teams intentionally foul to stop the clock."
    )
    ax_desc.text(0.05, 0.5, description, fontsize=10, verticalalignment='center', 
                 wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / "foul_frequency_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_pace_by_quarter(pace_metrics: pd.DataFrame, output_dir: Path):
    """Plot time series of possessions per minute across quarters."""
    print("Creating pace by quarter plot...")
    
    if pace_metrics.empty:
        print("  ⚠ No pace metrics data available")
        return
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax_desc = fig.add_subplot(gs[2, 0])
    ax_desc.axis('off')
    
    # Average pace by period
    period_pace = pace_metrics.groupby("period")["pace"].agg([
        "mean",
        "std",
        "count"
    ]).reset_index()
    
    ax1.bar(
        period_pace["period"],
        period_pace["mean"],
        yerr=period_pace["std"],
        capsize=5,
        color="steelblue",
        alpha=0.7,
        edgecolor="black"
    )
    ax1.set_xlabel("Period", fontsize=12)
    ax1.set_ylabel("Average Pace (Possessions per Minute)", fontsize=12)
    ax1.set_title("Average Pace by Period", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_xticks(period_pace["period"])
    
    # Box plot of pace by period
    sns.boxplot(
        data=pace_metrics,
        x="period",
        y="pace",
        ax=ax2
    )
    ax2.set_xlabel("Period", fontsize=12)
    ax2.set_ylabel("Pace (Possessions per Minute)", fontsize=12)
    ax2.set_title("Pace Distribution by Period", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add description
    description = (
        "Calculation: Pace is calculated as possessions per minute. A possession is defined as an event "
        "that ends a possession (made/missed shot, turnover, rebound). For each period, we count the "
        f"total possessions and divide by the period duration in minutes. Average pace: {pace_metrics['pace'].mean():.2f} "
        "possessions/min. Pace typically decreases in later quarters as teams slow down strategically."
    )
    ax_desc.text(0.05, 0.5, description, fontsize=10, verticalalignment='center', 
                 wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / "pace_by_quarter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_momentum_analysis(momentum: pd.DataFrame, output_dir: Path):
    """Plot momentum analysis (rolling point differences)."""
    print("Creating momentum analysis plot...")
    
    if momentum.empty:
        print("  ⚠ No momentum data available")
        return
    
    # Sample games for visualization (showing more games for better pattern visibility)
    # Using first 10 games to show more variety while keeping plot readable
    sample_games = momentum["game_id"].unique()[:10]
    sample_momentum = momentum[momentum["game_id"].isin(sample_games)]
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax_desc = fig.add_subplot(gs[2, 0])
    ax_desc.axis('off')
    
    # Plot score margin over time for sample games
    colors = plt.cm.tab10(range(len(sample_games)))
    for idx, game_id in enumerate(sample_games):
        game_data = sample_momentum[sample_momentum["game_id"] == game_id].sort_values("game_time_seconds")
        ax1.plot(
            game_data["game_time_seconds"] / 60,  # Convert to minutes
            game_data["score_margin"],
            alpha=0.5,
            linewidth=1.2,
            label=f"Game {game_id[-4:]}",
            color=colors[idx]
        )
    
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Game Time (minutes)", fontsize=12)
    ax1.set_ylabel("Score Margin (Home - Visitor)", fontsize=12)
    ax1.set_title("Score Margin Over Time (Sample Games)", fontsize=14, fontweight="bold")
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot momentum (rolling average of margin changes)
    for idx, game_id in enumerate(sample_games):
        game_data = sample_momentum[sample_momentum["game_id"] == game_id].sort_values("game_time_seconds")
        ax2.plot(
            game_data["game_time_seconds"] / 60,
            game_data["momentum_5pos"],
            alpha=0.5,
            linewidth=1.2,
            label=f"Game {game_id[-4:]}",
            color=colors[idx]
        )
    
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Game Time (minutes)", fontsize=12)
    ax2.set_ylabel("Momentum (5-possession rolling avg)", fontsize=12)
    ax2.set_title("Momentum Over Time (Sample Games)", fontsize=14, fontweight="bold")
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Add description
    description = (
        "Calculation: Score margin = Home score - Visitor score. Momentum is calculated as a 5-possession "
        "rolling average of score margin changes. Positive momentum means the home team is gaining ground, "
        "negative means the visitor team is gaining. The top plot shows raw score margin over time, while "
        "the bottom shows smoothed momentum. Most momentum shifts are short-lived, indicating volatile games."
    )
    ax_desc.text(0.05, 0.5, description, fontsize=10, verticalalignment='center', 
                 wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / "momentum_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_comeback_probability(comeback_prob: pd.DataFrame, output_dir: Path):
    """Plot comeback probability heatmap by deficit and time remaining - HOME TEAM PERSPECTIVE."""
    logger.info("Creating comeback probability heatmap (home team perspective)...")
    
    if comeback_prob.empty:
        logger.warning("No comeback probability data available")
        return
    
    # Sort and prepare data for heatmap
    comeback_prob_sorted = comeback_prob.sort_values(["minutes_remaining", "deficit"])
    
    # Create pivot table for heatmap
    heatmap_data = comeback_prob_sorted.pivot_table(
        values="win_probability",
        index="deficit",
        columns="minutes_remaining",
        aggfunc="mean"
    )
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    
    # Create simple annotations (just percentages, larger font)
    annot_text = []
    for i in range(len(heatmap_data.index)):
        row = []
        for j in range(len(heatmap_data.columns)):
            prob = heatmap_data.iloc[i, j]
            if pd.isna(prob):
                row.append("")
            else:
                row.append(f"{prob:.0%}")  # Just percentage, no sample size
        annot_text.append(row)
    
    # Use a clearer colormap
    sns.heatmap(
        heatmap_data,
        annot=annot_text,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Home Team Win Probability (%)', 'shrink': 0.8},
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    # Update colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label("Home Team Win Probability (%)", fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    ax.set_xlabel("Minutes Remaining in Game", fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel("Points Home Team is Trailing By", fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title("Home Team Comeback Win Probability\nWhat % chance does the HOME team have to win when trailing?", 
                 fontsize=20, fontweight="bold", pad=20)
    
    # Make tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add simple explanation BELOW the plot
    example = comeback_prob[(comeback_prob["deficit"] == 10) & (comeback_prob["minutes_remaining"] == 5)]
    if len(example) > 0:
        example_prob = example.iloc[0]["win_probability"]
        example_text = f"Example: Home team down 10 points with 5 min left = {example_prob:.0%} chance to win"
    else:
        example_text = "Example: Home team down 10 points with 5 min left = ~20% chance to win"
    
    fig = plt.gcf()
    fig.text(0.5, 0.02, 
             f"READ: Find how many points the HOME team is trailing by (left) and minutes remaining (bottom). "
             f"Color shows HOME team's win probability. {example_text}. Green = good chance, Red = low chance",
             fontsize=14, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2),
             family='sans-serif')
    
    plt.tight_layout()
    output_path = output_dir / "comeback_probability.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_timeout_effectiveness(timeout_effect: pd.DataFrame, output_dir: Path):
    """Plot timeout effectiveness analysis."""
    logger.info("Creating timeout effectiveness plot...")
    
    if timeout_effect.empty:
        logger.warning("No timeout effectiveness data available")
        return
    
    # Use net_change if available (team perspective), otherwise fall back to old score_change
    if "net_change" in timeout_effect.columns:
        change_col = "net_change"
        change_label = "Net Score Change (Team - Opponent)"
        explanation = "Net points: team that called timeout minus opponent in next 2 minutes"
    else:
        change_col = "score_change"
        change_label = "Score Change"
        explanation = "Total scoring change (both teams combined)"
    
    # Calculate key statistics
    mean_change = timeout_effect[change_col].mean()
    effective_timeouts = (timeout_effect[change_col] > 0).sum()  # Positive = team outscored opponent
    total_timeouts = len(timeout_effect)
    effectiveness_rate = (effective_timeouts / total_timeouts * 100) if total_timeouts > 0 else 0
    
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax_desc = fig.add_subplot(gs[1, 0])
    ax_desc.axis('off')
    
    # Simple histogram with clear color coding
    n, bins, patches = ax.hist(
        timeout_effect[change_col],
        bins=50,
        edgecolor="black",
        alpha=0.8,
        linewidth=1.5
    )
    
    # Color bars: green for positive (effective), red for negative (ineffective)
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val > 0:
            patch.set_facecolor("lightgreen")  # Effective timeout (team outscored opponent)
        elif bin_val < 0:
            patch.set_facecolor("lightcoral")  # Ineffective timeout (opponent outscored team)
        else:
            patch.set_facecolor("lightgray")
    
    # Add reference lines
    ax.axvline(0, color="black", linestyle="--", linewidth=3, label="No Change (Tied)", zorder=3)
    ax.axvline(mean_change, color="blue", linestyle="-", linewidth=3, 
                label=f"Average: {mean_change:.2f} points", zorder=3)
    
    ax.set_xlabel(change_label, fontsize=16, fontweight="bold", labelpad=15)
    ax.set_ylabel("Number of Timeouts", fontsize=16, fontweight="bold", labelpad=15)
    ax.set_title("Do Timeouts Actually Work?\nDoes the Team That Calls Timeout Outscore Opponent in Next 2 Minutes?", 
                 fontsize=20, fontweight="bold", pad=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=1)
    
    ax.legend(fontsize=14, loc="upper right", framealpha=0.9)
    
    # Add clear explanation BELOW the plot
    textstr = f"WHAT THIS SHOWS: {explanation}. "
    textstr += f"Green (right) = team that called timeout outscored opponent (timeout worked). "
    textstr += f"Red (left) = opponent outscored team (timeout didn't work).\n"
    textstr += f"RESULT: {effectiveness_rate:.1f}% of timeouts were effective (team outscored opponent). "
    textstr += f"Average net change: {mean_change:.2f} points."
    ax_desc.text(0.5, 0.5, textstr, transform=ax_desc.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='black', linewidth=2),
            family='sans-serif')
    
    plt.tight_layout()
    output_path = output_dir / "timeout_effectiveness.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_overtime_predictors(overtime_pred: pd.DataFrame, output_dir: Path):
    """Plot overtime game predictors."""
    logger.info("Creating overtime predictors plot...")
    
    if overtime_pred.empty:
        logger.warning("No overtime predictor data available")
        return
    
    ot_games = overtime_pred[overtime_pred["went_to_overtime"] == 1]
    reg_games = overtime_pred[overtime_pred["went_to_overtime"] == 0]
    
    # Calculate key statistics
    close_ot_rate = overtime_pred[overtime_pred["close_game"] == 1]["went_to_overtime"].mean() * 100
    not_close_ot_rate = overtime_pred[overtime_pred["close_game"] == 0]["went_to_overtime"].mean() * 100
    avg_reg_lead = reg_games["max_lead"].mean()
    avg_ot_lead = ot_games["max_lead"].mean()
    
    # Get margin at 5 min stats (use margin_at_5min if available)
    margin_col = "margin_at_5min" if "margin_at_5min" in reg_games.columns else "final_margin"
    avg_reg_margin = reg_games[margin_col].mean()
    avg_ot_margin = ot_games[margin_col].mean()
    
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax_desc = fig.add_subplot(gs[2, :])
    ax_desc.axis('off')
    
    # Plot 1: Close game indicator (bar chart with percentages) - MAIN INSIGHT
    categories = ["Close Games\n(Margin ≤5 pts)", "Not Close\n(Margin >5 pts)"]
    percentages = [close_ot_rate, not_close_ot_rate]
    colors = ["crimson", "steelblue"]
    
    bars = ax1.bar(categories, percentages, color=colors, alpha=0.8, edgecolor="black", linewidth=2.5)
    # Remove x-axis label since categories are self-explanatory
    ax1.set_ylabel("Chance of Going to Overtime (%)", fontsize=16, fontweight="bold", labelpad=15)
    ax1.set_title("Key Finding: Close Games Go to OT More Often\n(Measured with 5 minutes left in regulation)", 
                  fontsize=18, fontweight="bold", pad=15)
    ax1.set_ylim(0, max(percentages) * 1.3)
    ax1.tick_params(axis='x', which='major', labelsize=13, pad=10)
    ax1.tick_params(axis='y', which='major', labelsize=14)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=16, fontweight="bold")
    
    # Plot 2: Max lead comparison (simple box plot)
    bp = ax2.boxplot(
        [reg_games["max_lead"].dropna(), ot_games["max_lead"].dropna()],
        patch_artist=True,
        widths=0.6
    )
    
    # Set labels separately for better control
    ax2.set_xticklabels([f"Regulation\n({len(reg_games)} games)", f"Overtime\n({len(ot_games)} games)"], 
                        fontsize=12)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], ['steelblue', 'crimson']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel("Game Type", fontsize=14, fontweight="bold", labelpad=10)
    ax2.set_ylabel("Maximum Lead in Game (Points)", fontsize=16, fontweight="bold", labelpad=15)
    ax2.set_title("OT Games Stay Competitive Throughout", 
                  fontsize=18, fontweight="bold", pad=15)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=1)
    
    # Plot 3: Margin with 5 minutes left distribution (overlapping histograms)
    # Use margin_at_5min if available, otherwise fall back to final_margin
    margin_col = "margin_at_5min" if "margin_at_5min" in reg_games.columns else "final_margin"
    
    ax3.hist(reg_games[margin_col], bins=50, alpha=0.6, label=f"Regulation Games ({len(reg_games)})", 
             color="steelblue", edgecolor="black", linewidth=1)
    ax3.hist(ot_games[margin_col], bins=50, alpha=0.8, label=f"Overtime Games ({len(ot_games)})", 
             color="crimson", edgecolor="black", linewidth=1)
    ax3.axvline(5, color="orange", linestyle="--", linewidth=3, alpha=0.7, label="Close Game (5 pts)")
    ax3.set_xlabel("Score Margin with 5 Minutes Left (Points)", fontsize=16, fontweight="bold", labelpad=15)
    ax3.set_ylabel("Number of Games", fontsize=16, fontweight="bold", labelpad=15)
    ax3.set_title("Margin with 5 Minutes Left: Close Games More Likely to Go to OT", 
                  fontsize=16, fontweight="bold", pad=12)
    ax3.tick_params(axis='x', which='major', labelsize=14)
    ax3.tick_params(axis='y', which='major', labelsize=14)
    ax3.legend(fontsize=13, loc="upper right", framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=1)
    
    # Add simple explanation in dedicated subplot (no overlap)
    textstr = f"WHAT PREDICTS OVERTIME? "
    textstr += f"• Games close with 5 minutes left (margin ≤5 pts) are {close_ot_rate/not_close_ot_rate:.1f}x more likely to go to OT ({close_ot_rate:.1f}% vs {not_close_ot_rate:.1f}%). "
    textstr += f"• OT games have smaller max leads during regulation (avg {avg_ot_lead:.0f} pts vs {avg_reg_lead:.0f} pts) - they stay competitive. "
    textstr += f"• Games that are close with 5 minutes left are more likely to end regulation tied and go to OT. "
    textstr += f"Note: 'Close game' means score margin ≤5 points with 5 MINUTES LEFT in regulation, not at the end."
    ax_desc.text(0.5, 0.5, textstr, transform=ax_desc.transAxes, fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=2),
                 family='sans-serif', wrap=True)
    
    plt.tight_layout()
    output_path = output_dir / "overtime_predictors.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all plots."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    db_file = data_dir / "nba_playbyplay.duckdb"
    output_dir = project_root / "analysis" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not db_file.exists():
        raise FileNotFoundError(
            f"Database not found: {db_file}\n"
            "Please run pipeline/load_duckdb.py and analysis/compute_metrics.py first."
        )
    
    logger.info("Loading data from DuckDB...")
    scoring_runs, droughts, foul_freq, pace_metrics, momentum, comeback_prob, clutch_perf, timeout_effect, overtime_pred = load_data(db_file)
    
    logger.info("Generating visualizations...")
    logger.info("=" * 60)
    
    # Generate key visualizations
    plot_comeback_probability(comeback_prob, output_dir)
    plot_timeout_effectiveness(timeout_effect, output_dir)
    plot_overtime_predictors(overtime_pred, output_dir)
    
    logger.info("=" * 60)
    logger.info("All visualizations generated!")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

