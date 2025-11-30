"""
Generate visualizations for NBA play-by-play analysis.

Creates:
1. Scoring run distribution
2. Average drought length by team
3. Heatmap of foul frequency by minute
4. Time series of possessions per minute across quarters
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

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
    
    conn.close()
    
    return scoring_runs, droughts, foul_freq, pace_metrics, momentum


def plot_scoring_run_distribution(scoring_runs: pd.DataFrame, output_dir: Path):
    """Plot distribution of scoring run lengths."""
    print("Creating scoring run distribution plot...")
    
    if scoring_runs.empty:
        print("  ⚠ No scoring runs data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
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
    print(f"✓ Saved: {output_path}")
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
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
    
    plt.tight_layout()
    output_path = output_dir / "drought_length_by_team.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
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
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
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
    
    plt.tight_layout()
    output_path = output_dir / "foul_frequency_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_pace_by_quarter(pace_metrics: pd.DataFrame, output_dir: Path):
    """Plot time series of possessions per minute across quarters."""
    print("Creating pace by quarter plot...")
    
    if pace_metrics.empty:
        print("  ⚠ No pace metrics data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
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
    
    plt.tight_layout()
    output_path = output_dir / "pace_by_quarter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_momentum_analysis(momentum: pd.DataFrame, output_dir: Path):
    """Plot momentum analysis (rolling point differences)."""
    print("Creating momentum analysis plot...")
    
    if momentum.empty:
        print("  ⚠ No momentum data available")
        return
    
    # Sample a few games for visualization
    sample_games = momentum["game_id"].unique()[:5]
    sample_momentum = momentum[momentum["game_id"].isin(sample_games)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot score margin over time for sample games
    for game_id in sample_games:
        game_data = sample_momentum[sample_momentum["game_id"] == game_id].sort_values("game_time_seconds")
        ax1.plot(
            game_data["game_time_seconds"] / 60,  # Convert to minutes
            game_data["score_margin"],
            alpha=0.6,
            linewidth=1.5,
            label=f"Game {game_id[-4:]}"
        )
    
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Game Time (minutes)", fontsize=12)
    ax1.set_ylabel("Score Margin (Home - Visitor)", fontsize=12)
    ax1.set_title("Score Margin Over Time (Sample Games)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot momentum (rolling average of margin changes)
    for game_id in sample_games:
        game_data = sample_momentum[sample_momentum["game_id"] == game_id].sort_values("game_time_seconds")
        ax2.plot(
            game_data["game_time_seconds"] / 60,
            game_data["momentum_5pos"],
            alpha=0.6,
            linewidth=1.5,
            label=f"Game {game_id[-4:]}"
        )
    
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Game Time (minutes)", fontsize=12)
    ax2.set_ylabel("Momentum (5-possession rolling avg)", fontsize=12)
    ax2.set_title("Momentum Over Time (Sample Games)", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "momentum_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
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
    
    print("Loading data from DuckDB...")
    scoring_runs, droughts, foul_freq, pace_metrics, momentum = load_data(db_file)
    
    print(f"\nGenerating visualizations...")
    print("="*60)
    
    # Generate all plots
    plot_scoring_run_distribution(scoring_runs, output_dir)
    plot_drought_length_by_team(droughts, output_dir)
    plot_foul_frequency_heatmap(foul_freq, output_dir)
    plot_pace_by_quarter(pace_metrics, output_dir)
    plot_momentum_analysis(momentum, output_dir)
    
    print("\n" + "="*60)
    print("✓ All visualizations generated!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

