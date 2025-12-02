"""
Fetch list of all games for a given NBA season.

This script uses the NBA Stats API to get all games for a season,
handling rate limits and pagination.
"""

import os
import json
import time
import requests
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime

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


def fetch_season_games(season_year: int, season_type: str = "Regular Season") -> List[Dict]:
    """
    Fetch all games for a given season.
    
    Args:
        season_year: Year of the season (e.g., 2023 for 2023-24 season)
        season_type: "Regular Season", "Playoffs", or "All Star"
        
    Returns:
        List of game dictionaries
    """
    headers = get_headers()
    config_path = Path(__file__).parent.parent / "config" / "nba_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    all_games = []
    
    print(f"Fetching games for {season_year}-{season_year+1} {season_type}")
    
    # Use the LeagueGameFinder endpoint which is more reliable
    endpoint = f"{NBA_API_BASE}/leaguegamefinder"
    
    # Season format: "2023-24" for 2023-24 season
    season_str = f"{season_year}-{str(season_year+1)[-2:]}"
    
    # Map season type to API format
    season_type_map = {
        "Regular Season": "Regular Season",
        "Playoffs": "Playoffs",
        "All Star": "All Star"
    }
    api_season_type = season_type_map.get(season_type, "Regular Season")
    
    params = {
        "LeagueID": "00",
        "Season": season_str,
        "SeasonType": api_season_type,
        "PlayerOrTeam": "T",  # Team
    }
    
    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=config["api"]["timeout"]
        )
        response.raise_for_status()
        
        data = response.json()
        result_sets = data.get("resultSets", [])
        
        if result_sets:
            games_data = result_sets[0]
            headers_list = games_data.get("headers", [])
            rows = games_data.get("rowSet", [])
            
            # Convert to list of dictionaries
            for row in rows:
                game = dict(zip(headers_list, row))
                all_games.append(game)
        
        print(f"✓ Fetched {len(all_games)} games")
        
        # Rate limiting
        time.sleep(config["api"]["rate_limit_delay"])
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching games: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}")
        raise
    
    return all_games


def main():
    """Main function to fetch and save games."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "nba_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    season_year = config["season"]["year"]
    season_type = config["season"]["season_type"]
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "games.json"
    
    print("="*60)
    print("NBA Games Fetcher")
    print("="*60)
    
    # Fetch games
    games = fetch_season_games(season_year, season_type)
    
    if not games:
        print("⚠ No games found. Check season year and type.")
        return
    
    # Save to JSON
    print(f"\nSaving {len(games)} games to {output_file}")
    with open(output_file, "w") as f:
        json.dump(games, f, indent=2)
    
    # Print summary
    print(f"\n✓ Saved {len(games)} games")
    print(f"  Output: {output_file}")
    print(f"\nSample game IDs:")
    for game in games[:5]:
        game_id = game.get("GAME_ID") or game.get("game_id")
        if game_id:
            print(f"  - {game_id}")


if __name__ == "__main__":
    main()
 
