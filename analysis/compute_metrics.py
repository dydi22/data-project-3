"""
Compute analytical metrics from play-by-play data.

This script computes:
- Scoring runs and drought lengths
- Foul frequencies by quarter
- Time-based pace metrics
- Momentum measures (rolling point differences)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


def compute_scoring_runs(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute scoring runs (consecutive scoring possessions) for each team.
    
    Returns DataFrame with run lengths and team information.
    """
    print("Computing scoring runs...")
    
    # First check what columns are available
    available_cols = conn.execute("DESCRIBE playbyplay").fetchdf()["column_name"].tolist()
    
    # Build query based on available columns
    select_cols = ["game_id", "period", "event_msg_type", "visitor_score", "home_score", "game_time_seconds"]
    
    # Add optional columns if they exist
    if "period_clock_seconds" in available_cols:
        select_cols.append("period_clock_seconds")
    if "home_description" in available_cols:
        select_cols.append("home_description")
    if "visitor_description" in available_cols:
        select_cols.append("visitor_description")
    
    query = f"""
        SELECT 
            {', '.join(select_cols)}
        FROM playbyplay
        WHERE event_msg_type IN (1, 2, 3)  -- Made shot, missed shot, free throw
            AND (visitor_score IS NOT NULL OR home_score IS NOT NULL)
        ORDER BY game_id, game_time_seconds
    """
    
    df = conn.execute(query).fetchdf()
    
    if df.empty:
        return pd.DataFrame()
    
    # Identify scoring events (made shots and made free throws)
    df["is_score"] = df["event_msg_type"].isin([1, 3])  # Made shot or free throw
    
    # Determine which team scored based on score changes
    df["score_change"] = 0
    df["team_scored"] = None
    
    runs = []
    
    for game_id in df["game_id"].unique():
        game_df = df[df["game_id"] == game_id].copy().sort_values("game_time_seconds")
        
        # Calculate score changes to determine which team scored
        game_df["prev_visitor_score"] = game_df["visitor_score"].shift(1).fillna(0)
        game_df["prev_home_score"] = game_df["home_score"].shift(1).fillna(0)
        game_df["visitor_change"] = game_df["visitor_score"] - game_df["prev_visitor_score"]
        game_df["home_change"] = game_df["home_score"] - game_df["prev_home_score"]
        
        # Determine which team scored (simplified: use score change)
        game_df["team_scored"] = None
        game_df.loc[game_df["visitor_change"] > 0, "team_scored"] = "VISITOR"
        game_df.loc[game_df["home_change"] > 0, "team_scored"] = "HOME"
        
        # Track runs for each team
        current_run = 0
        last_team = None
        
        for idx, row in game_df.iterrows():
            if row["is_score"] and row["team_scored"] is not None:
                team = row["team_scored"]
                if team == last_team:
                    current_run += 1
                else:
                    if last_team is not None and current_run > 0:
                        runs.append({
                            "game_id": game_id,
                            "team": last_team,
                            "run_length": current_run,
                            "period": row["period"]
                        })
                    current_run = 1
                    last_team = team
            else:
                if last_team is not None and current_run > 0:
                    runs.append({
                        "game_id": game_id,
                        "team": last_team,
                        "run_length": current_run,
                        "period": row["period"]
                    })
                current_run = 0
                last_team = None
    
    runs_df = pd.DataFrame(runs)
    print(f"✓ Computed {len(runs_df)} scoring runs")
    return runs_df


def compute_droughts(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute scoring droughts (consecutive non-scoring possessions).
    
    Returns DataFrame with drought lengths by team.
    """
    print("Computing scoring droughts...")
    
    # Check available columns
    available_cols = conn.execute("DESCRIBE playbyplay").fetchdf()["column_name"].tolist()
    
    select_cols = ["game_id", "period", "event_msg_type", "game_time_seconds"]
    if "visitor_score" in available_cols:
        select_cols.append("visitor_score")
    if "home_score" in available_cols:
        select_cols.append("home_score")
    
    query = f"""
        SELECT 
            {', '.join(select_cols)}
        FROM playbyplay
        WHERE event_msg_type IN (1, 2, 3, 5)  -- Shot attempts and free throws
        ORDER BY game_id, game_time_seconds
    """
    
    df = conn.execute(query).fetchdf()
    
    if df.empty:
        return pd.DataFrame()
    
    df["is_score"] = df["event_msg_type"].isin([1, 3])
    
    # Determine which team based on score changes (similar to scoring runs)
    droughts = []
    
    for game_id in df["game_id"].unique():
        game_df = df[df["game_id"] == game_id].copy().sort_values("game_time_seconds")
        
        # Calculate score changes to determine team
        if "visitor_score" in game_df.columns and "home_score" in game_df.columns:
            game_df["prev_visitor_score"] = game_df["visitor_score"].shift(1).fillna(0)
            game_df["prev_home_score"] = game_df["home_score"].shift(1).fillna(0)
            game_df["visitor_change"] = game_df["visitor_score"] - game_df["prev_visitor_score"]
            game_df["home_change"] = game_df["home_score"] - game_df["prev_home_score"]
            
            # Determine team (simplified approach)
            game_df["team"] = "UNKNOWN"
            game_df.loc[game_df["visitor_change"] > 0, "team"] = "VISITOR"
            game_df.loc[game_df["home_change"] > 0, "team"] = "HOME"
        else:
            # If no score columns, use a simplified approach
            game_df["team"] = "UNKNOWN"
        
        current_drought = 0
        last_team = None
        
        for idx, row in game_df.iterrows():
            team = row["team"]
            if team != last_team and team != "UNKNOWN":
                if last_team is not None and current_drought > 0:
                    droughts.append({
                        "game_id": game_id,
                        "team": last_team,
                        "drought_length": current_drought,
                        "period": row["period"]
                    })
                current_drought = 0
                last_team = team
            
            if team != "UNKNOWN":
                if not row["is_score"]:
                    current_drought += 1
                else:
                    if current_drought > 0:
                        droughts.append({
                            "game_id": game_id,
                            "team": team,
                            "drought_length": current_drought,
                            "period": row["period"]
                        })
                    current_drought = 0
    
    droughts_df = pd.DataFrame(droughts)
    print(f"✓ Computed {len(droughts_df)} scoring droughts")
    return droughts_df


def compute_foul_frequencies(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute foul frequencies by quarter and minute.
    
    Returns DataFrame with foul counts by period and game minute.
    """
    print("Computing foul frequencies...")
    
    # Check available columns
    available_cols = conn.execute("DESCRIBE playbyplay").fetchdf()["column_name"].tolist()
    
    select_cols = ["game_id", "period", "game_time_seconds", "event_msg_type"]
    if "period_clock_seconds" in available_cols:
        select_cols.append("period_clock_seconds")
    
    query = f"""
        SELECT 
            {', '.join(select_cols)}
        FROM playbyplay
        WHERE event_msg_type = 6  -- Foul
            AND period_clock_seconds IS NOT NULL
        ORDER BY game_id, game_time_seconds
    """
    
    df = conn.execute(query).fetchdf()
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate minute in game (0-48 for regulation)
    df["game_minute"] = (df["game_time_seconds"] / 60).astype(int)
    df["game_minute"] = df["game_minute"].clip(0, 48)
    
    # Group by period and minute
    foul_freq = df.groupby(["period", "game_minute"]).size().reset_index(name="foul_count")
    
    print(f"✓ Computed foul frequencies for {len(foul_freq)} period-minute combinations")
    return foul_freq


def compute_pace_metrics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute pace metrics (possessions per minute) by quarter.
    
    Returns DataFrame with pace metrics.
    """
    print("Computing pace metrics...")
    
    query = """
        SELECT 
            game_id,
            period,
            period_clock_seconds,
            game_time_seconds,
            event_msg_type
        FROM playbyplay
        WHERE event_msg_type IN (1, 2, 3, 4, 5, 6)  -- Scoring and possession-ending events
            AND period_clock_seconds IS NOT NULL
        ORDER BY game_id, game_time_seconds
    """
    
    df = conn.execute(query).fetchdf()
    
    if df.empty:
        return pd.DataFrame()
    
    # Identify possession-ending events
    df["is_possession_end"] = df["event_msg_type"].isin([1, 2, 4, 5])  # Made/missed shot, turnover, rebound
    
    pace_metrics = []
    
    for game_id in df["game_id"].unique():
        game_df = df[df["game_id"] == game_id].copy().sort_values("game_time_seconds")
        
        for period in game_df["period"].unique():
            period_df = game_df[game_df["period"] == period]
            
            if len(period_df) == 0:
                continue
            
            # Count possessions in period
            possessions = period_df["is_possession_end"].sum()
            
            # Calculate time in period (in minutes)
            period_start = period_df["period_clock_seconds"].max()
            period_end = period_df["period_clock_seconds"].min()
            period_duration_minutes = (period_start - period_end) / 60.0
            
            if period_duration_minutes > 0:
                pace = possessions / period_duration_minutes
                
                pace_metrics.append({
                    "game_id": game_id,
                    "period": period,
                    "possessions": possessions,
                    "period_duration_minutes": period_duration_minutes,
                    "pace": pace
                })
    
    pace_df = pd.DataFrame(pace_metrics)
    print(f"✓ Computed pace metrics for {len(pace_df)} game-periods")
    return pace_df


def compute_momentum(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute momentum measures (rolling point differences across possessions).
    
    Returns DataFrame with momentum metrics.
    """
    print("Computing momentum measures...")
    
    query = """
        SELECT 
            game_id,
            period,
            game_time_seconds,
            visitor_score,
            home_score,
            score_margin
        FROM playbyplay
        WHERE visitor_score IS NOT NULL 
            AND home_score IS NOT NULL
        ORDER BY game_id, game_time_seconds
    """
    
    df = conn.execute(query).fetchdf()
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate score margin (home - visitor)
    df["score_margin"] = df["home_score"] - df["visitor_score"]
    
    # Calculate rolling point difference (momentum)
    momentum_data = []
    
    for game_id in df["game_id"].unique():
        game_df = df[df["game_id"] == game_id].copy().sort_values("game_time_seconds")
        
        # Calculate change in margin (momentum)
        game_df["margin_change"] = game_df["score_margin"].diff()
        
        # Rolling average of margin changes (5-possession window)
        game_df["momentum_5pos"] = game_df["margin_change"].rolling(window=5, min_periods=1).mean()
        
        momentum_data.append(game_df[["game_id", "period", "game_time_seconds", 
                                      "score_margin", "margin_change", "momentum_5pos"]])
    
    momentum_df = pd.concat(momentum_data, ignore_index=True)
    print(f"✓ Computed momentum for {len(momentum_df)} events")
    return momentum_df


def main():
    """Main function to compute all metrics."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    db_file = data_dir / "nba_playbyplay.duckdb"
    
    if not db_file.exists():
        raise FileNotFoundError(
            f"Database not found: {db_file}\n"
            "Please run pipeline/load_duckdb.py first."
        )
    
    # Connect to DuckDB
    print(f"Connecting to DuckDB: {db_file}")
    conn = duckdb.connect(str(db_file))
    
    try:
        # Compute all metrics
        scoring_runs = compute_scoring_runs(conn)
        droughts = compute_droughts(conn)
        foul_freq = compute_foul_frequencies(conn)
        pace_metrics = compute_pace_metrics(conn)
        momentum = compute_momentum(conn)
        
        # Save to DuckDB
        print("\nSaving metrics to DuckDB...")
        conn.execute("DROP TABLE IF EXISTS scoring_runs")
        conn.execute("DROP TABLE IF EXISTS droughts")
        conn.execute("DROP TABLE IF EXISTS foul_frequencies")
        conn.execute("DROP TABLE IF EXISTS pace_metrics")
        conn.execute("DROP TABLE IF EXISTS momentum")
        
        if not scoring_runs.empty:
            conn.execute("CREATE TABLE scoring_runs AS SELECT * FROM scoring_runs")
        if not droughts.empty:
            conn.execute("CREATE TABLE droughts AS SELECT * FROM droughts")
        if not foul_freq.empty:
            conn.execute("CREATE TABLE foul_frequencies AS SELECT * FROM foul_freq")
        if not pace_metrics.empty:
            conn.execute("CREATE TABLE pace_metrics AS SELECT * FROM pace_metrics")
        if not momentum.empty:
            conn.execute("CREATE TABLE momentum AS SELECT * FROM momentum")
        
        # Save to CSV for easy inspection
        output_dir = data_dir
        if not scoring_runs.empty:
            scoring_runs.to_csv(output_dir / "scoring_runs.csv", index=False)
        if not droughts.empty:
            droughts.to_csv(output_dir / "droughts.csv", index=False)
        if not foul_freq.empty:
            foul_freq.to_csv(output_dir / "foul_frequencies.csv", index=False)
        if not pace_metrics.empty:
            pace_metrics.to_csv(output_dir / "pace_metrics.csv", index=False)
        if not momentum.empty:
            momentum.to_csv(output_dir / "momentum.csv", index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("Metrics Summary")
        print("="*60)
        if not scoring_runs.empty:
            print(f"Scoring runs: {len(scoring_runs):,}")
            print(f"  Average run length: {scoring_runs['run_length'].mean():.2f}")
        if not droughts.empty:
            print(f"Scoring droughts: {len(droughts):,}")
            print(f"  Average drought length: {droughts['drought_length'].mean():.2f}")
        if not foul_freq.empty:
            print(f"Foul frequency records: {len(foul_freq):,}")
        if not pace_metrics.empty:
            print(f"Pace metrics: {len(pace_metrics):,}")
            print(f"  Average pace: {pace_metrics['pace'].mean():.2f} possessions/min")
        if not momentum.empty:
            print(f"Momentum records: {len(momentum):,}")
        
        print(f"\n✓ Metrics saved to DuckDB and CSV files in {output_dir}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()

