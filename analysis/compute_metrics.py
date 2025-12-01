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


def compute_comeback_probability(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute win probability based on score deficit and time remaining.
    
    Returns DataFrame with comeback probabilities.
    """
    print("Computing comeback probabilities...")
    
    # Get final scores and game outcomes
    query = """
        SELECT 
            game_id,
            MAX(game_time_seconds) as final_time,
            MAX(visitor_score) as final_visitor_score,
            MAX(home_score) as final_home_score
        FROM playbyplay
        WHERE visitor_score IS NOT NULL AND home_score IS NOT NULL
        GROUP BY game_id
    """
    
    final_scores = conn.execute(query).fetchdf()
    
    # Determine winner (1 = home won, 0 = visitor won)
    final_scores["home_won"] = (final_scores["final_home_score"] > final_scores["final_visitor_score"]).astype(int)
    
    # Get score margins at different time points
    comeback_data = []
    
    for game_id in final_scores["game_id"].unique():
        game_final = final_scores[final_scores["game_id"] == game_id].iloc[0]
        home_won = game_final["home_won"]
        
        # Get score margins at different time points
        time_query = f"""
            SELECT 
                game_time_seconds,
                home_score - visitor_score as score_margin,
                (48 * 60 - game_time_seconds) / 60.0 as minutes_remaining
            FROM playbyplay
            WHERE game_id = '{game_id}'
                AND visitor_score IS NOT NULL 
                AND home_score IS NOT NULL
                AND game_time_seconds <= 48 * 60  -- Only regulation
            ORDER BY game_time_seconds
        """
        
        game_data = conn.execute(time_query).fetchdf()
        
        # Sample at 5-minute intervals
        for minutes_remaining in [5, 10, 15, 20, 25, 30, 35, 40]:
            target_time = (48 - minutes_remaining) * 60
            closest = game_data.iloc[(game_data["game_time_seconds"] - target_time).abs().argsort()[:1]]
            
            if len(closest) > 0:
                margin = closest.iloc[0]["score_margin"]
                # Margin is home - visitor, so negative = home losing
                # If home won and was losing (negative margin), that's a comeback
                # If visitor won and home was winning (positive margin), that's a comeback for visitor
                if home_won:
                    # Home team perspective: were they losing?
                    was_losing = margin < 0
                    deficit = abs(margin) if was_losing else 0
                    came_back = 1 if was_losing else 0
                else:
                    # Visitor team perspective: was home winning?
                    was_losing = margin > 0
                    deficit = abs(margin) if was_losing else 0
                    came_back = 1 if was_losing else 0
                
                comeback_data.append({
                    "game_id": game_id,
                    "minutes_remaining": minutes_remaining,
                    "deficit": deficit,
                    "home_won": home_won,
                    "came_back": came_back
                })
    
    comeback_df = pd.DataFrame(comeback_data)
    
    # Calculate probabilities
    if not comeback_df.empty:
        prob_df = comeback_df.groupby(["minutes_remaining", "deficit"])["came_back"].agg([
            "mean", "count"
        ]).reset_index()
        prob_df.columns = ["minutes_remaining", "deficit", "win_probability", "sample_size"]
        prob_df = prob_df[prob_df["sample_size"] >= 5]  # Only include where we have enough data
    else:
        prob_df = pd.DataFrame()
    
    print(f"✓ Computed comeback probabilities for {len(prob_df)} time-deficit combinations")
    return prob_df


def compute_clutch_performance(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute team performance in clutch situations (last 5 minutes, score within 5).
    
    Returns DataFrame with clutch performance metrics.
    """
    print("Computing clutch performance...")
    
    # Get games that were close in the last 5 minutes
    query = """
        SELECT 
            game_id,
            period,
            game_time_seconds,
            visitor_score,
            home_score,
            ABS(home_score - visitor_score) as score_diff,
            event_msg_type
        FROM playbyplay
        WHERE game_time_seconds >= (48 * 60 - 5 * 60)  -- Last 5 minutes
            AND game_time_seconds <= 48 * 60  -- Regulation only
            AND visitor_score IS NOT NULL 
            AND home_score IS NOT NULL
            AND ABS(home_score - visitor_score) <= 5  -- Within 5 points
        ORDER BY game_id, game_time_seconds
    """
    
    clutch_data = conn.execute(query).fetchdf()
    
    if clutch_data.empty:
        return pd.DataFrame()
    
    # Determine which team scored (simplified - using score changes)
    clutch_data["home_scored"] = 0
    clutch_data["visitor_scored"] = 0
    
    for game_id in clutch_data["game_id"].unique():
        game_data = clutch_data[clutch_data["game_id"] == game_id].copy().sort_values("game_time_seconds")
        game_data["prev_home"] = game_data["home_score"].shift(1).fillna(0)
        game_data["prev_visitor"] = game_data["visitor_score"].shift(1).fillna(0)
        
        home_change = game_data["home_score"] - game_data["prev_home"]
        visitor_change = game_data["visitor_score"] - game_data["prev_visitor"]
        
        game_data.loc[home_change > 0, "home_scored"] = 1
        game_data.loc[visitor_change > 0, "visitor_scored"] = 1
        
        clutch_data.loc[clutch_data["game_id"] == game_id, "home_scored"] = game_data["home_scored"]
        clutch_data.loc[clutch_data["game_id"] == game_id, "visitor_scored"] = game_data["visitor_scored"]
    
    # Aggregate by game
    clutch_performance = clutch_data.groupby("game_id").agg({
        "home_scored": "sum",
        "visitor_scored": "sum"
    }).reset_index()
    
    print(f"✓ Computed clutch performance for {len(clutch_performance)} games")
    return clutch_performance


def compute_timeout_effectiveness(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Analyze if timeouts effectively stop opponent scoring runs.
    
    Returns DataFrame with timeout effectiveness metrics.
    """
    print("Computing timeout effectiveness...")
    
    # Find timeout events (event_msg_type 9 = timeout)
    query = """
        SELECT 
            game_id,
            period,
            game_time_seconds,
            event_msg_type,
            visitor_score,
            home_score
        FROM playbyplay
        WHERE event_msg_type = 9  -- Timeout
            OR (home_description LIKE '%Timeout%' OR visitor_description LIKE '%Timeout%')
        ORDER BY game_id, game_time_seconds
    """
    
    timeouts = conn.execute(query).fetchdf()
    
    if timeouts.empty:
        return pd.DataFrame()
    
    # Get scoring before and after timeouts
    timeout_effectiveness = []
    
    for idx, timeout in timeouts.iterrows():
        game_id = timeout["game_id"]
        timeout_time = timeout["game_time_seconds"]
        
        # Get scores before and after timeout
        before_query = f"""
            SELECT 
                MAX(visitor_score) as visitor_score,
                MAX(home_score) as home_score
            FROM playbyplay
            WHERE game_id = '{game_id}'
                AND game_time_seconds >= {max(0, timeout_time - 120)}  -- 2 minutes before
                AND game_time_seconds < {timeout_time}
                AND visitor_score IS NOT NULL AND home_score IS NOT NULL
        """
        
        after_query = f"""
            SELECT 
                MAX(visitor_score) as visitor_score,
                MAX(home_score) as home_score
            FROM playbyplay
            WHERE game_id = '{game_id}'
                AND game_time_seconds > {timeout_time}
                AND game_time_seconds <= {timeout_time + 120}  -- 2 minutes after
                AND visitor_score IS NOT NULL AND home_score IS NOT NULL
        """
        
        before = conn.execute(before_query).fetchdf()
        after = conn.execute(after_query).fetchdf()
        
        if len(before) > 0 and len(after) > 0:
            before_score = before.iloc[0]["visitor_score"] + before.iloc[0]["home_score"]
            after_score = after.iloc[0]["visitor_score"] + after.iloc[0]["home_score"]
            
            timeout_effectiveness.append({
                "game_id": game_id,
                "timeout_time": timeout_time,
                "period": timeout["period"],
                "score_before": before_score,
                "score_after": after_score,
                "score_change": after_score - before_score
            })
    
    effectiveness_df = pd.DataFrame(timeout_effectiveness)
    
    print(f"✓ Computed timeout effectiveness for {len(effectiveness_df)} timeouts")
    return effectiveness_df


def compute_overtime_predictors(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Analyze what game characteristics predict overtime games.
    
    Returns DataFrame with overtime game characteristics.
    """
    print("Computing overtime predictors...")
    
    # Find games that went to overtime (period > 4)
    query = """
        SELECT 
            game_id,
            MAX(period) as max_period,
            COUNT(DISTINCT period) as num_periods,
            MAX(visitor_score) as final_visitor_score,
            MAX(home_score) as final_home_score,
            ABS(MAX(home_score) - MAX(visitor_score)) as final_margin
        FROM playbyplay
        WHERE visitor_score IS NOT NULL AND home_score IS NOT NULL
        GROUP BY game_id
    """
    
    game_stats = conn.execute(query).fetchdf()
    game_stats["went_to_overtime"] = (game_stats["max_period"] > 4).astype(int)
    
    # Get additional game characteristics
    characteristics = []
    
    for game_id in game_stats["game_id"].unique():
        game_stat = game_stats[game_stats["game_id"] == game_id].iloc[0]
        
        # Get lead changes, close quarters, etc.
        detail_query = f"""
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT period) as periods_played,
                MAX(ABS(home_score - visitor_score)) as max_lead,
                MIN(ABS(home_score - visitor_score)) as min_lead
            FROM playbyplay
            WHERE game_id = '{game_id}'
                AND visitor_score IS NOT NULL AND home_score IS NOT NULL
                AND game_time_seconds <= 48 * 60  -- Regulation only
        """
        
        details = conn.execute(detail_query).fetchdf()
        
        if len(details) > 0:
            characteristics.append({
                "game_id": game_id,
                "went_to_overtime": game_stat["went_to_overtime"],
                "final_margin": game_stat["final_margin"],
                "max_lead": details.iloc[0]["max_lead"],
                "min_lead": details.iloc[0]["min_lead"],
                "total_events": details.iloc[0]["total_events"],
                "close_game": 1 if game_stat["final_margin"] <= 5 else 0
            })
    
    predictors_df = pd.DataFrame(characteristics)
    
    print(f"✓ Computed overtime predictors for {len(predictors_df)} games")
    return predictors_df


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
        comeback_prob = compute_comeback_probability(conn)
        clutch_perf = compute_clutch_performance(conn)
        timeout_effect = compute_timeout_effectiveness(conn)
        overtime_pred = compute_overtime_predictors(conn)
        
        # Save to DuckDB
        print("\nSaving metrics to DuckDB...")
        conn.execute("DROP TABLE IF EXISTS scoring_runs")
        conn.execute("DROP TABLE IF EXISTS droughts")
        conn.execute("DROP TABLE IF EXISTS foul_frequencies")
        conn.execute("DROP TABLE IF EXISTS pace_metrics")
        conn.execute("DROP TABLE IF EXISTS momentum")
        conn.execute("DROP TABLE IF EXISTS comeback_probability")
        conn.execute("DROP TABLE IF EXISTS clutch_performance")
        conn.execute("DROP TABLE IF EXISTS timeout_effectiveness")
        conn.execute("DROP TABLE IF EXISTS overtime_predictors")
        
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
        if not comeback_prob.empty:
            conn.execute("CREATE TABLE comeback_probability AS SELECT * FROM comeback_prob")
        if not clutch_perf.empty:
            conn.execute("CREATE TABLE clutch_performance AS SELECT * FROM clutch_perf")
        if not timeout_effect.empty:
            conn.execute("CREATE TABLE timeout_effectiveness AS SELECT * FROM timeout_effect")
        if not overtime_pred.empty:
            conn.execute("CREATE TABLE overtime_predictors AS SELECT * FROM overtime_pred")
        
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
        if not comeback_prob.empty:
            comeback_prob.to_csv(output_dir / "comeback_probability.csv", index=False)
        if not clutch_perf.empty:
            clutch_perf.to_csv(output_dir / "clutch_performance.csv", index=False)
        if not timeout_effect.empty:
            timeout_effect.to_csv(output_dir / "timeout_effectiveness.csv", index=False)
        if not overtime_pred.empty:
            overtime_pred.to_csv(output_dir / "overtime_predictors.csv", index=False)
        
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
        if not comeback_prob.empty:
            print(f"Comeback probability records: {len(comeback_prob):,}")
        if not clutch_perf.empty:
            print(f"Clutch performance records: {len(clutch_perf):,}")
        if not timeout_effect.empty:
            print(f"Timeout effectiveness records: {len(timeout_effect):,}")
        if not overtime_pred.empty:
            ot_games = overtime_pred["went_to_overtime"].sum()
            print(f"Overtime predictors: {len(overtime_pred):,} games ({ot_games} went to OT)")
        
        print(f"\n✓ Metrics saved to DuckDB and CSV files in {output_dir}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()

