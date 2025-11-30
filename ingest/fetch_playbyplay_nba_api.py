"""
Fetch play-by-play data using the nba_api Python package.

This is an alternative implementation that uses the nba_api package
which handles API complexities better than direct requests.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import time

try:
    from nba_api.stats.endpoints import PlayByPlayV2
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠ nba_api package not installed. Install with: pip install nba-api")


def fetch_playbyplay_nba_api(game_id: str, timeout: int = 30) -> Optional[Dict]:
    """
    Fetch play-by-play data using nba_api package.
    
    Args:
        game_id: NBA game ID (e.g., "0022300001")
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing play-by-play data or None if failed
    """
    if not NBA_API_AVAILABLE:
        return None
    
    try:
        # Use nba_api package
        pbp = PlayByPlayV2(game_id=game_id, timeout=timeout)
        
        # Get the data
        data = pbp.get_dict()
        result_sets = data.get("resultSets", [])
        
        if not result_sets:
            return None
        
        # The play-by-play data is typically in the first result set
        playbyplay_data = result_sets[0]
        events = playbyplay_data.get("rowSet", [])
        
        if not events:
            return None
        
        return {
            "game_id": game_id,
            "headers": playbyplay_data.get("headers", []),
            "events": events,
            "name": playbyplay_data.get("name", "PlayByPlay"),
        }
        
    except Exception as e:
        # Silently fail - we'll log summary at the end
        return None


def main():
    """Main function to fetch play-by-play data for all games using nba_api."""
    if not NBA_API_AVAILABLE:
        print("Please install nba_api: pip install nba-api")
        return
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "nba_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load games
    games_file = Path(__file__).parent.parent / "data" / "raw" / "games.json"
    if not games_file.exists():
        raise FileNotFoundError(
            f"Games file not found: {games_file}\n"
            "Please run ingest/fetch_games.py first."
        )
    
    with open(games_file, "r") as f:
        games = json.load(f)
    
    print(f"Found {len(games)} games to process")
    
    # Extract game IDs
    game_ids = []
    for game in games:
        game_id = game.get("GAME_ID") or game.get("game_id")
        if game_id:
            game_ids.append(str(game_id))
    
    if not game_ids:
        print("⚠ No game IDs found")
        return
    
    print(f"Processing {len(game_ids)} games using nba_api package")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "playbyplay"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch play-by-play data
    successful = 0
    failed = 0
    
    for game_id in tqdm(game_ids, desc="Fetching play-by-play"):
        pbp_data = fetch_playbyplay_nba_api(game_id, timeout=config["api"]["timeout"])
        
        if pbp_data:
            # Save to JSON file
            output_file = output_dir / f"{game_id}.json"
            with open(output_file, "w") as f:
                json.dump(pbp_data, f, indent=2)
            successful += 1
        else:
            failed += 1
        
        # Rate limiting
        time.sleep(config["api"]["rate_limit_delay"])
        
        # Print progress every 100 games
        if (successful + failed) % 100 == 0:
            print(f"\nProgress: {successful} successful, {failed} failed")
    
    print(f"\n✓ Completed fetching play-by-play data")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

