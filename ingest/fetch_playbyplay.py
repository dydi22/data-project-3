"""
Fetch play-by-play data for each game.

This script iterates over all games and fetches detailed play-by-play
event data, handling rate limits and large responses.
"""

import os
import json
import time
import requests
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tqdm import tqdm
import random

# Load environment variables
load_dotenv()

# NBA Stats API endpoints
NBA_API_BASE = "https://stats.nba.com/stats"


def get_headers() -> Dict[str, str]:
    """Get headers for NBA Stats API requests."""
    config_path = Path(__file__).parent.parent / "config" / "nba_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["api"]["headers"]


def fetch_playbyplay(game_id: str, max_retries: int = 3, session: Optional[requests.Session] = None) -> Optional[Dict]:
    """
    Fetch play-by-play data for a single game.
    
    Args:
        game_id: NBA game ID (e.g., "0022300001")
        max_retries: Maximum number of retry attempts
        session: Optional requests session for connection pooling
        
    Returns:
        Dictionary containing play-by-play data or None if failed
    """
    headers = get_headers()
    config_path = Path(__file__).parent.parent / "config" / "nba_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Try multiple endpoint variations
    endpoints = [
        f"{NBA_API_BASE}/playbyplayv2",
        f"{NBA_API_BASE}/playbyplay",
    ]
    
    params_variations = [
        {
            "GameID": game_id,
            "StartPeriod": "0",
            "EndPeriod": "0",
        },
        {
            "GameID": game_id,
            "StartPeriod": 0,
            "EndPeriod": 0,
        },
        {
            "gameId": game_id,
            "startPeriod": 0,
            "endPeriod": 0,
        },
    ]
    
    # Use session if provided, otherwise create a new request
    request_func = session.get if session else requests.get
    
    # Try all endpoint/parameter combinations
    for endpoint in endpoints:
        for params in params_variations:
            for attempt in range(max_retries):
                try:
                    response = request_func(
                        endpoint,
                        headers=headers,
                        params=params,
                        timeout=config["api"]["timeout"]
                    )
                    
                    # Check for various error status codes
                    if response.status_code == 404:
                        # Game not found or no play-by-play data
                        break  # Try next parameter combination
                    
                    if response.status_code == 403:
                        # Forbidden - API is blocking
                        break  # Try next parameter combination
                    
                    if response.status_code != 200:
                        if attempt < max_retries - 1:
                            wait_time = config["processing"]["retry_delay"] * (attempt + 1)
                            time.sleep(wait_time)
                            continue
                        else:
                            break  # Try next parameter combination
                    
                    # Try to parse JSON
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        if attempt < max_retries - 1:
                            wait_time = config["processing"]["retry_delay"] * (attempt + 1)
                            time.sleep(wait_time)
                            continue
                        else:
                            break  # Try next parameter combination
                    
                    # Extract result sets
                    result_sets = data.get("resultSets", [])
                    
                    if not result_sets:
                        break  # Try next parameter combination
                    
                    # The play-by-play data is typically in the first result set
                    playbyplay_data = result_sets[0]
                    events = playbyplay_data.get("rowSet", [])
                    
                    # Check if we actually got data
                    if not events:
                        break  # Try next parameter combination
                    
                    return {
                        "game_id": game_id,
                        "headers": playbyplay_data.get("headers", []),
                        "events": events,
                        "name": playbyplay_data.get("name", "PlayByPlay"),
                    }
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        wait_time = config["processing"]["retry_delay"] * (attempt + 1)
                        time.sleep(wait_time)
                        continue
                    break  # Try next parameter combination
                except requests.exceptions.RequestException:
                    if attempt < max_retries - 1:
                        wait_time = config["processing"]["retry_delay"] * (attempt + 1)
                        time.sleep(wait_time)
                        continue
                    break  # Try next parameter combination
    
    # If we get here, all combinations failed
    return None


def main():
    """Main function to fetch play-by-play data for all games."""
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
        game_id = game.get("GAME_ID") or game.get("game_id") or game.get("GAME_ID")
        if game_id:
            game_ids.append(str(game_id))
    
    if not game_ids:
        # Try alternative field names
        print("⚠ No game IDs found with standard field names. Trying alternatives...")
        if games:
            print(f"Sample game keys: {list(games[0].keys())}")
    
    print(f"Processing {len(game_ids)} games")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "playbyplay"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which games already exist
    existing_files = {f.stem for f in output_dir.glob("*.json")}
    games_to_fetch = [gid for gid in game_ids if gid not in existing_files]
    games_skipped = len(game_ids) - len(games_to_fetch)
    
    if games_skipped > 0:
        print(f"✓ Skipping {games_skipped} games that are already fetched")
    print(f"Fetching {len(games_to_fetch)} remaining games...\n")
    
    if not games_to_fetch:
        print("All games have already been fetched!")
        return
    
    # Fetch play-by-play data
    successful = 0
    failed = 0
    
    # Use a session for connection pooling
    session = requests.Session()
    session.headers.update(get_headers())
    
    # Test with first game to see if API is working
    if games_to_fetch:
        print(f"Testing API with first game: {games_to_fetch[0]}")
        test_result = fetch_playbyplay(games_to_fetch[0], max_retries=1, session=session)
        if test_result:
            print(f"✓ API is working! Fetched {len(test_result.get('events', []))} events")
        else:
            print("⚠ API test failed. The NBA Stats API may be blocking requests.")
            print("  This is common - the API sometimes blocks automated requests.")
            print("  You may need to:")
            print("  1. Wait and try again later")
            print("  2. Use a VPN or different network")
            print("  3. Check if the API endpoint has changed")
            print("\nContinuing anyway...\n")
        time.sleep(1)
    
    for game_id in tqdm(games_to_fetch, desc="Fetching play-by-play"):
        pbp_data = fetch_playbyplay(game_id, max_retries=config["processing"]["max_retries"], session=session)
        
        if pbp_data:
            # Save to JSON file
            output_file = output_dir / f"{game_id}.json"
            with open(output_file, "w") as f:
                json.dump(pbp_data, f, indent=2)
            successful += 1
        else:
            failed += 1
        
        # Rate limiting - increase delay if many failures
        if failed > successful and failed > 10:
            # If we're getting many failures, slow down more
            time.sleep(config["api"]["rate_limit_delay"] * 2)
        else:
            time.sleep(config["api"]["rate_limit_delay"])
    
    session.close()
    
    print(f"\n✓ Completed fetching play-by-play data")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

