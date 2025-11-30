"""
Test script to verify NBA Stats API is accessible.

This script tests fetching play-by-play data for a single game
to verify the API is working before running the full pipeline.
"""

import requests
import json
import yaml
from pathlib import Path

NBA_API_BASE = "https://stats.nba.com/stats"

def get_headers():
    """Get headers for NBA Stats API requests."""
    config_path = Path(__file__).parent.parent / "config" / "nba_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["api"]["headers"]

def test_playbyplay_api(game_id: str = "0022300001"):
    """Test fetching play-by-play data for a single game."""
    headers = get_headers()
    endpoint = f"{NBA_API_BASE}/playbyplayv2"
    
    params = {
        "GameID": game_id,
        "StartPeriod": "0",
        "EndPeriod": "0",
    }
    
    print(f"Testing NBA Stats API...")
    print(f"Endpoint: {endpoint}")
    print(f"Game ID: {game_id}")
    print(f"Headers: {headers.get('User-Agent', 'N/A')[:50]}...")
    print()
    
    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(list(response.headers.items())[:5])}")
        print()
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"  Full response keys: {list(data.keys())}")
                print(f"  Response preview: {str(data)[:500]}")
                
                result_sets = data.get("resultSets", [])
                
                if result_sets:
                    pbp_data = result_sets[0]
                    events = pbp_data.get("rowSet", [])
                    print(f"✓ SUCCESS! Fetched {len(events)} events")
                    print(f"  Headers: {len(pbp_data.get('headers', []))} columns")
                    if events:
                        print(f"  Sample event: {events[0][:3] if len(events[0]) > 3 else events[0]}")
                    return True
                else:
                    print("⚠ Response has no result sets")
                    if not data:
                        print("  Response is empty JSON object {}")
                        print("  This usually means:")
                        print("    - Game ID format is wrong")
                        print("    - Game doesn't have play-by-play data")
                        print("    - API endpoint/parameters need adjustment")
                    return False
            except json.JSONDecodeError as e:
                print(f"✗ Failed to parse JSON: {e}")
                print(f"  Response text (first 500 chars): {response.text[:500]}")
                return False
        elif response.status_code == 403:
            print("✗ FORBIDDEN (403) - API is blocking requests")
            print("  The NBA Stats API may be blocking automated requests.")
            print("  Try:")
            print("  1. Waiting and trying again later")
            print("  2. Using a VPN")
            print("  3. Checking if the API endpoint has changed")
            return False
        elif response.status_code == 404:
            print(f"✗ NOT FOUND (404) - Game {game_id} not found")
            print("  Try a different game ID")
            return False
        else:
            print(f"✗ Error: HTTP {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ TIMEOUT - Request took too long")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ REQUEST ERROR: {e}")
        return False

if __name__ == "__main__":
    # Try a known game ID from 2023-24 season
    # You can change this to any game ID from your games.json
    test_game_id = "0022300001"  # First game of 2023-24 season
    
    # Try to get a game ID from games.json if it exists
    games_file = Path(__file__).parent.parent / "data" / "raw" / "games.json"
    if games_file.exists():
        with open(games_file, "r") as f:
            games = json.load(f)
            if games:
                test_game_id = games[0].get("GAME_ID", test_game_id)
                print(f"Using game ID from games.json: {test_game_id}\n")
    
    success = test_playbyplay_api(test_game_id)
    
    if success:
        print("\n✓ API is working! You can proceed with the full pipeline.")
    else:
        print("\n✗ API test failed. Please check the error messages above.")
        print("  The NBA Stats API may require different headers or may be blocking requests.")

