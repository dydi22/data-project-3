"""
Normalize and clean play-by-play data into a flat table structure.

This script processes raw JSON play-by-play files and converts them
into a normalized pandas DataFrame, handling inconsistent schemas.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def load_playbyplay_file(file_path: Path) -> Optional[Dict]:
    """Load a single play-by-play JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Error loading {file_path.name}: {e}")
        return None


def normalize_playbyplay(pbp_data: Dict) -> pd.DataFrame:
    """
    Normalize play-by-play data into a flat DataFrame.
    
    Args:
        pbp_data: Dictionary containing play-by-play data
        
    Returns:
        DataFrame with normalized events
    """
    game_id = pbp_data.get("game_id", "unknown")
    headers = pbp_data.get("headers", [])
    # Try both "events" and "rowSet" for compatibility
    events = pbp_data.get("events", pbp_data.get("rowSet", []))
    
    if not events:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(events, columns=headers)
    
    # Standardize column names (NBA API can be inconsistent)
    column_mapping = {
        "GAME_ID": "game_id",
        "EVENTNUM": "event_num",
        "EVENTMSGTYPE": "event_msg_type",
        "EVENTMSGACTIONTYPE": "event_msg_action_type",
        "PERIOD": "period",
        "WCTIMESTRING": "wall_clock_time",
        "PCTIMESTRING": "period_clock",
        "HOMEDESCRIPTION": "home_description",
        "NEUTRALDESCRIPTION": "neutral_description",
        "VISITORDESCRIPTION": "visitor_description",
        "SCORE": "score",
        "SCOREMARGIN": "score_margin",
        "PERSON1TYPE": "person1_type",
        "PLAYER1_ID": "player1_id",
        "PLAYER1_NAME": "player1_name",
        "PLAYER1_TEAM_ID": "player1_team_id",
        "PLAYER1_TEAM_CITY": "player1_team_city",
        "PLAYER1_TEAM_NICKNAME": "player1_team_nickname",
        "PLAYER1_TEAM_ABBREVIATION": "player1_team_abbreviation",
        "PERSON2TYPE": "person2_type",
        "PLAYER2_ID": "player2_id",
        "PLAYER2_NAME": "player2_name",
        "PLAYER2_TEAM_ID": "player2_team_id",
        "PLAYER2_TEAM_CITY": "player2_team_city",
        "PLAYER2_TEAM_NICKNAME": "player2_team_nickname",
        "PLAYER2_TEAM_ABBREVIATION": "player2_team_abbreviation",
        "PERSON3TYPE": "person3_type",
        "PLAYER3_ID": "player3_id",
        "PLAYER3_NAME": "player3_name",
        "POSSESSION": "possession",
    }
    
    # Rename columns that exist (this will rename GAME_ID to game_id if it exists)
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Ensure game_id exists and is a string (use the game_id from function param if column doesn't exist)
    if "game_id" not in df.columns:
        df["game_id"] = game_id
    else:
        # If game_id column exists from GAME_ID rename, ensure it's set correctly
        df["game_id"] = df["game_id"].fillna(game_id)
    df["game_id"] = df["game_id"].astype(str)
    
    # Remove any duplicate columns (in case of any issues)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Parse period clock time (format: "MM:SS" or "M:SS")
    if "period_clock" in df.columns:
        df["period_clock_seconds"] = df["period_clock"].apply(parse_time_string)
    
    # Parse score
    if "score" in df.columns:
        score_df = df["score"].str.split(" - ", expand=True)
        if len(score_df.columns) >= 2:
            df["visitor_score"] = pd.to_numeric(score_df[0], errors="coerce")
            df["home_score"] = pd.to_numeric(score_df[1], errors="coerce")
    
    # Calculate total game time in seconds
    if "period" in df.columns and "period_clock_seconds" in df.columns:
        df["game_time_seconds"] = (
            (df["period"] - 1) * 12 * 60 +  # 12 minutes per period
            (12 * 60 - df["period_clock_seconds"])  # Time elapsed in period
        )
    
    return df


def parse_time_string(time_str: str) -> float:
    """Parse time string (MM:SS or M:SS) to seconds."""
    if pd.isna(time_str) or time_str == "":
        return np.nan
    
    try:
        parts = str(time_str).split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
    except:
        pass
    
    return np.nan


def main():
    """Main function to normalize all play-by-play data."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "raw" / "playbyplay"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Play-by-play directory not found: {input_dir}\n"
            "Please run ingest/fetch_playbyplay.py first."
        )
    
    # Get all JSON files
    pbp_files = list(input_dir.glob("*.json"))
    print(f"Found {len(pbp_files)} play-by-play files to process")
    
    # Process each file
    all_events = []
    successful = 0
    failed = 0
    
    for pbp_file in pbp_files:
        pbp_data = load_playbyplay_file(pbp_file)
        if not pbp_data:
            failed += 1
            continue
        
        try:
            df = normalize_playbyplay(pbp_data)
            if not df.empty:
                all_events.append(df)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ⚠ Error processing {pbp_file.name}: {e}")
            failed += 1
    
    if not all_events:
        print("⚠ No events to process")
        return
    
    # Combine all events
    print(f"\nCombining {len(all_events)} game dataframes...")
    combined_df = pd.concat(all_events, ignore_index=True)
    
    print(f"✓ Normalized {len(combined_df):,} events from {successful} games")
    print(f"  Failed: {failed}")
    
    # Save to Parquet
    output_file = output_dir / "playbyplay_normalized.parquet"
    print(f"\nSaving to {output_file}...")
    combined_df.to_parquet(output_file, index=False, compression="snappy")
    
    print(f"✓ Saved {len(combined_df):,} events to Parquet")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Total events: {len(combined_df):,}")
    print(f"Unique games: {combined_df['game_id'].nunique()}")
    if "event_msg_type" in combined_df.columns:
        print(f"\nEvent types:")
        print(combined_df["event_msg_type"].value_counts().head(10))
    if "period" in combined_df.columns:
        print(f"\nPeriods: {combined_df['period'].min()} to {combined_df['period'].max()}")


if __name__ == "__main__":
    main()

