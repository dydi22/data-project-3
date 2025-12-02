"""
Compute analytical metrics from play-by-play data.

This script computes:
- Scoring runs and drought lengths
- Foul frequencies by quarter
- Time-based pace metrics
- Momentum measures (rolling point differences)
- Comeback probabilities
- Clutch performance
- Timeout effectiveness
- Overtime predictors
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging_config import setup_logging, get_logger

# Set up logging
logger = setup_logging(log_level="INFO")


def compute_scoring_runs(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute scoring runs (consecutive scoring possessions) for each team.
    
    Returns DataFrame with run lengths and team information.
    """
    logger.info("Computing scoring runs...")
    
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
    logger.info(f"Computed {len(runs_df)} scoring runs")
    return runs_df


def compute_droughts(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute scoring droughts (consecutive non-scoring possessions).
    
    Returns DataFrame with drought lengths by team.
    """
    logger.info("Computing scoring droughts...")
    
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
    logger.info(f"Computed {len(droughts_df)} scoring droughts")
    return droughts_df


def compute_foul_frequencies(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute foul frequencies by quarter and minute.
    
    Returns DataFrame with foul counts by period and game minute.
    """
    logger.info("Computing foul frequencies...")
    
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
    
    logger.info(f"Computed foul frequencies for {len(foul_freq)} period-minute combinations")
    return foul_freq


def compute_pace_metrics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute pace metrics (possessions per minute) by quarter.
    
    Returns DataFrame with pace metrics.
    """
    logger.info("Computing pace metrics...")
    
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
    logger.info(f"Computed pace metrics for {len(pace_df)} game-periods")
    return pace_df


def compute_momentum(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute momentum measures (rolling point differences across possessions).
    
    Returns DataFrame with momentum metrics.
    """
    logger.info("Computing momentum measures...")
    
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
    logger.info(f"Computed momentum for {len(momentum_df)} events")
    return momentum_df


def compute_comeback_probability(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute win probability based on score deficit and time remaining.
    
    Returns DataFrame with comeback probabilities.
    """
    logger.info("Computing comeback probabilities...")
    
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
    
    # Determine winner
    final_scores["home_won"] = (final_scores["final_home_score"] > final_scores["final_visitor_score"]).astype(int)
    
    # Get score margins at different time points for ALL games
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
        
        if len(game_data) == 0:
            continue
        
        # Sample at 5-minute intervals
        for minutes_remaining in [5, 10, 15, 20, 25, 30, 35, 40]:
            target_time = (48 - minutes_remaining) * 60
            closest_idx = (game_data["game_time_seconds"] - target_time).abs().idxmin()
            closest = game_data.loc[[closest_idx]]
            
            if len(closest) > 0:
                margin = closest.iloc[0]["score_margin"]
                
                # Only track HOME team's win probability when they're trailing
                # margin = home_score - visitor_score, so negative means home is losing
                if margin < 0:  # Home is losing
                    deficit = abs(margin)
                    home_won_despite_trailing = 1 if home_won else 0
                    comeback_data.append({
                        "game_id": game_id,
                        "minutes_remaining": minutes_remaining,
                        "deficit": deficit,
                        "home_won": home_won_despite_trailing  # Did home team win despite trailing?
                    })
    
    comeback_df = pd.DataFrame(comeback_data)
    
    # Calculate win probabilities for trailing teams
    if not comeback_df.empty:
        # Round deficits to 2-point buckets to increase sample sizes (e.g., 8-9 points -> 8 points)
        comeback_df["deficit_bucket"] = (comeback_df["deficit"] // 2) * 2
        
        prob_df = comeback_df.groupby(["minutes_remaining", "deficit_bucket"])["home_won"].agg([
            "mean", "count"
        ]).reset_index()
        prob_df.columns = ["minutes_remaining", "deficit", "win_probability", "sample_size"]
        # Only include combinations with at least 5 samples for reliability
        prob_df = prob_df[prob_df["sample_size"] >= 5]
    else:
        prob_df = pd.DataFrame()
    
        logger.info(f"Computed comeback probabilities for {len(prob_df)} time-deficit combinations")
        if not prob_df.empty:
            logger.debug(f"  Sample sizes range from {prob_df['sample_size'].min()} to {prob_df['sample_size'].max()}")
            logger.debug(f"  Win probabilities range from {prob_df['win_probability'].min():.2%} to {prob_df['win_probability'].max():.2%}")
    
    return prob_df


def compute_clutch_performance(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute team performance in clutch situations (last 5 minutes, score within 5).
    
    Returns DataFrame with clutch performance metrics.
    """
    logger.info("Computing clutch performance...")
    
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
    
    logger.info(f"Computed clutch performance for {len(clutch_performance)} games")
    return clutch_performance


def compute_timeout_effectiveness(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Analyze if timeouts effectively stop opponent scoring runs.
    
    Returns DataFrame with timeout effectiveness metrics.
    """
    logger.info("Computing timeout effectiveness...")
    
    # Find timeout events (event_msg_type 9 = timeout)
    # Check available columns first
    available_cols = conn.execute("DESCRIBE playbyplay").fetchdf()["column_name"].tolist()
    select_cols = ["game_id", "period", "game_time_seconds", "event_msg_type", "visitor_score", "home_score"]
    if "home_description" in available_cols:
        select_cols.append("home_description")
    if "visitor_description" in available_cols:
        select_cols.append("visitor_description")
    
    query = f"""
        SELECT 
            {', '.join(select_cols)}
        FROM playbyplay
        WHERE event_msg_type = 9  -- Timeout
            OR (home_description LIKE '%Timeout%' OR visitor_description LIKE '%Timeout%')
        ORDER BY game_id, game_time_seconds
    """
    
    timeouts = conn.execute(query).fetchdf()
    
    if timeouts.empty:
        return pd.DataFrame()
    
    # Get scoring before and after timeouts from the perspective of the team that called it
    timeout_effectiveness = []
    
    for idx, timeout in timeouts.iterrows():
        game_id = timeout["game_id"]
        timeout_time = timeout["game_time_seconds"]
        
        # Determine which team called the timeout
        # If home_description has timeout, home team called it; if visitor_description has it, visitor called it
        home_called = False
        visitor_called = False
        if "home_description" in timeout and pd.notna(timeout["home_description"]):
            if "Timeout" in str(timeout["home_description"]):
                home_called = True
        if "visitor_description" in timeout and pd.notna(timeout["visitor_description"]):
            if "Timeout" in str(timeout["visitor_description"]):
                visitor_called = True
        
        # If we can't determine, skip (or default to home if both are None)
        if not home_called and not visitor_called:
            continue
        
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
            before_home = before.iloc[0]["home_score"]
            before_visitor = before.iloc[0]["visitor_score"]
            after_home = after.iloc[0]["home_score"]
            after_visitor = after.iloc[0]["visitor_score"]
            
            # Calculate score change from the perspective of the team that called the timeout
            if home_called:
                team_score_before = before_home
                team_score_after = after_home
                opponent_score_before = before_visitor
                opponent_score_after = after_visitor
                team_type = "home"
            else:  # visitor_called
                team_score_before = before_visitor
                team_score_after = after_visitor
                opponent_score_before = before_home
                opponent_score_after = after_home
                team_type = "visitor"
            
            # Net change: positive = team that called timeout scored more than opponent in next 2 min
            team_change = team_score_after - team_score_before
            opponent_change = opponent_score_after - opponent_score_before
            net_change = team_change - opponent_change  # Positive = timeout worked (team outscored opponent)
            
            timeout_effectiveness.append({
                "game_id": game_id,
                "timeout_time": timeout_time,
                "period": timeout["period"],
                "team_type": team_type,
                "team_score_before": team_score_before,
                "team_score_after": team_score_after,
                "opponent_score_before": opponent_score_before,
                "opponent_score_after": opponent_score_after,
                "team_change": team_change,
                "opponent_change": opponent_change,
                "net_change": net_change  # Positive = team that called timeout did better
            })
    
    effectiveness_df = pd.DataFrame(timeout_effectiveness)
    
    logger.info(f"Computed timeout effectiveness for {len(effectiveness_df)} timeouts")
    return effectiveness_df


def compute_overtime_predictors(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Analyze what game characteristics predict overtime games.
    
    Returns DataFrame with overtime game characteristics.
    """
    logger.info("Computing overtime predictors...")
    
    # Find games that went to overtime (period > 4)
    query = """
        SELECT 
            game_id,
            MAX(period) as max_period,
            COUNT(DISTINCT period) as num_periods
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
        
        # Get score with 5 MINUTES LEFT in regulation (at 43 minutes) - this is what predicts OT!
        # We want to predict based on game state BEFORE we know the outcome
        five_min_left_time = 43 * 60  # 43 minutes = 5 minutes before end of regulation
        
        margin_at_5min_query = f"""
            SELECT 
                visitor_score,
                home_score,
                ABS(home_score - visitor_score) as margin
            FROM playbyplay
            WHERE game_id = '{game_id}'
                AND visitor_score IS NOT NULL AND home_score IS NOT NULL
                AND game_time_seconds <= {five_min_left_time}
                AND game_time_seconds >= {five_min_left_time - 60}  -- Within 1 minute of target
            ORDER BY ABS(game_time_seconds - {five_min_left_time})
            LIMIT 1
        """
        
        margin_5min = conn.execute(margin_at_5min_query).fetchdf()
        
        # Get final score (including OT if it happened) for reference
        final_query = f"""
            SELECT 
                MAX(visitor_score) as final_visitor_score,
                MAX(home_score) as final_home_score
            FROM playbyplay
            WHERE game_id = '{game_id}'
                AND visitor_score IS NOT NULL AND home_score IS NOT NULL
        """
        
        final = conn.execute(final_query).fetchdf()
        
        # Get lead changes, close quarters, etc. during REGULATION
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
        
        if len(margin_5min) > 0 and len(final) > 0 and len(details) > 0:
            # Margin with 5 minutes left in regulation - this predicts OT!
            margin_at_5min = margin_5min.iloc[0]["margin"]
            
            # Final margin (after OT if it happened) - for reference
            final_margin = abs(final.iloc[0]["final_home_score"] - final.iloc[0]["final_visitor_score"])
            
            characteristics.append({
                "game_id": game_id,
                "went_to_overtime": game_stat["went_to_overtime"],
                "margin_at_5min": margin_at_5min,  # Margin with 5 min left (predictor)
                "final_margin": final_margin,  # Final margin including OT (reference)
                "max_lead": details.iloc[0]["max_lead"],
                "min_lead": details.iloc[0]["min_lead"],
                "total_events": details.iloc[0]["total_events"],
                "close_game": 1 if margin_at_5min <= 5 else 0  # Close with 5 min left (≤5 pts)
            })
    
    predictors_df = pd.DataFrame(characteristics)
    
    logger.info(f"Computed overtime predictors for {len(predictors_df)} games")
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
    logger.info(f"Connecting to DuckDB: {db_file}")
    conn = duckdb.connect(str(db_file))
    
    try:
        # Compute all metrics
        logger.info("Starting metric computation...")
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
        logger.info("Saving metrics to DuckDB...")
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
        logger.info("=" * 60)
        logger.info("Metrics Summary")
        logger.info("=" * 60)
        if not scoring_runs.empty:
            logger.info(f"Scoring runs: {len(scoring_runs):,}")
            logger.info(f"  Average run length: {scoring_runs['run_length'].mean():.2f}")
        if not droughts.empty:
            logger.info(f"Scoring droughts: {len(droughts):,}")
            logger.info(f"  Average drought length: {droughts['drought_length'].mean():.2f}")
        if not foul_freq.empty:
            logger.info(f"Foul frequency records: {len(foul_freq):,}")
        if not pace_metrics.empty:
            logger.info(f"Pace metrics: {len(pace_metrics):,}")
            logger.info(f"  Average pace: {pace_metrics['pace'].mean():.2f} possessions/min")
        if not momentum.empty:
            logger.info(f"Momentum records: {len(momentum):,}")
        if not comeback_prob.empty:
            logger.info(f"Comeback probability records: {len(comeback_prob):,}")
        if not clutch_perf.empty:
            logger.info(f"Clutch performance records: {len(clutch_perf):,}")
        if not timeout_effect.empty:
            logger.info(f"Timeout effectiveness records: {len(timeout_effect):,}")
        if not overtime_pred.empty:
            ot_games = overtime_pred["went_to_overtime"].sum()
            logger.info(f"Overtime predictors: {len(overtime_pred):,} games ({ot_games} went to OT)")
        
        logger.info(f"Metrics saved to DuckDB and CSV files in {output_dir}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()

