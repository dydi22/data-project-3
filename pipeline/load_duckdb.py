"""
Load normalized play-by-play data into DuckDB for analytical queries.

This script creates a DuckDB database and loads the Parquet files
for fast analytical queries.
"""

import duckdb
import pandas as pd
from pathlib import Path
import json


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create database schema."""
    print("Creating database schema...")
    
    # Drop existing tables if they exist
    conn.execute("DROP TABLE IF EXISTS playbyplay")
    conn.execute("DROP TABLE IF EXISTS games")
    
    # Create playbyplay table (will be loaded from Parquet)
    # We'll use DuckDB's ability to query Parquet directly, but also create a table
    print("✓ Schema ready (using Parquet files directly)")


def load_playbyplay(conn: duckdb.DuckDBPyConnection, parquet_file: Path) -> None:
    """Load play-by-play data from Parquet into DuckDB."""
    print(f"Loading play-by-play data from {parquet_file}...")
    
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
    
    # Create table from Parquet file
    conn.execute(f"""
        CREATE TABLE playbyplay AS
        SELECT * FROM read_parquet('{parquet_file}')
    """)
    
    # Get row count
    count = conn.execute("SELECT COUNT(*) FROM playbyplay").fetchone()[0]
    print(f"✓ Loaded {count:,} events into playbyplay table")


def load_games(conn: duckdb.DuckDBPyConnection, games_file: Path) -> None:
    """Load games metadata into DuckDB."""
    print(f"Loading games metadata from {games_file}...")
    
    if not games_file.exists():
        print("⚠ Games file not found, skipping...")
        return
    
    # Load JSON and convert to DataFrame
    with open(games_file, "r") as f:
        games_data = json.load(f)
    
    if not games_data:
        print("⚠ No games data found")
        return
    
    df = pd.DataFrame(games_data)
    
    # Create table
    conn.execute("CREATE TABLE games AS SELECT * FROM df")
    
    count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"✓ Loaded {count} games into games table")


def main():
    """Main function to load data into DuckDB."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_file = data_dir / "playbyplay_normalized.parquet"
    games_file = project_root / "data" / "raw" / "games.json"
    db_file = data_dir / "nba_playbyplay.duckdb"
    
    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_file}\n"
            "Please run pipeline/normalize_data.py first."
        )
    
    # Connect to DuckDB
    print(f"Connecting to DuckDB: {db_file}")
    conn = duckdb.connect(str(db_file))
    
    try:
        # Create schema
        create_schema(conn)
        
        # Load data
        load_playbyplay(conn, parquet_file)
        load_games(conn, games_file)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Database Summary")
        print("="*60)
        
        event_count = conn.execute("SELECT COUNT(*) FROM playbyplay").fetchone()[0]
        game_count = conn.execute("SELECT COUNT(DISTINCT game_id) FROM playbyplay").fetchone()[0]
        
        print(f"Total events: {event_count:,}")
        print(f"Unique games: {game_count}")
        
        # Show sample query
        print("\nSample query - Event types:")
        sample = conn.execute("""
            SELECT 
                event_msg_type,
                COUNT(*) as count
            FROM playbyplay
            WHERE event_msg_type IS NOT NULL
            GROUP BY event_msg_type
            ORDER BY count DESC
            LIMIT 10
        """).fetchdf()
        print(sample.to_string(index=False))
        
    finally:
        conn.close()
    
    print(f"\n✓ Database created successfully: {db_file}")


if __name__ == "__main__":
    main()
 
