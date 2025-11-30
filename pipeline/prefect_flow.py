"""
Prefect workflow to orchestrate the NBA play-by-play data pipeline.

This workflow runs:
1. Fetch games for the season
2. Fetch play-by-play data for each game
3. Normalize and clean the data
4. Load into DuckDB
"""

from prefect import flow, task
from pathlib import Path
import subprocess
import sys


@task(name="fetch_games", log_prints=True)
def fetch_games_task():
    """Fetch list of all games for the season."""
    script_path = Path(__file__).parent.parent / "ingest" / "fetch_games.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"fetch_games failed: {result.stderr}")
    print(result.stdout)
    return "games.json created"


@task(name="fetch_playbyplay", log_prints=True)
def fetch_playbyplay_task():
    """Fetch play-by-play data for each game."""
    script_path = Path(__file__).parent.parent / "ingest" / "fetch_playbyplay.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"fetch_playbyplay failed: {result.stderr}")
    print(result.stdout)
    return "play-by-play data fetched"


@task(name="normalize_data", log_prints=True)
def normalize_data_task():
    """Normalize and clean play-by-play data."""
    script_path = Path(__file__).parent.parent / "pipeline" / "normalize_data.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"normalize_data failed: {result.stderr}")
    print(result.stdout)
    return "data normalized to Parquet"


@task(name="load_duckdb", log_prints=True)
def load_duckdb_task():
    """Load data into DuckDB."""
    script_path = Path(__file__).parent.parent / "pipeline" / "load_duckdb.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"load_duckdb failed: {result.stderr}")
    print(result.stdout)
    return "data loaded into DuckDB"


@flow(name="nba_playbyplay_pipeline", log_prints=True)
def nba_playbyplay_pipeline():
    """
    Main pipeline flow for NBA play-by-play analysis.
    
    This flow orchestrates:
    1. Fetching games for the season
    2. Fetching play-by-play data for each game
    3. Normalizing and cleaning the data
    4. Loading data into DuckDB for analysis
    """
    print("Starting NBA play-by-play pipeline...")
    
    # Step 1: Fetch games
    games_result = fetch_games_task()
    print(f"✓ {games_result}")
    
    # Step 2: Fetch play-by-play data
    pbp_result = fetch_playbyplay_task()
    print(f"✓ {pbp_result}")
    
    # Step 3: Normalize data
    normalize_result = normalize_data_task()
    print(f"✓ {normalize_result}")
    
    # Step 4: Load into DuckDB
    load_result = load_duckdb_task()
    print(f"✓ {load_result}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python analysis/compute_metrics.py")
    print("2. Run: python analysis/plots.py")
    
    return {
        "games": games_result,
        "playbyplay": pbp_result,
        "normalize": normalize_result,
        "load": load_result,
    }


if __name__ == "__main__":
    # Run the flow
    nba_playbyplay_pipeline()

