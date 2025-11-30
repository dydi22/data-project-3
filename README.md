# NBA Play-by-Play Data Analysis Pipeline

## Project Overview

This project builds a complete data engineering pipeline that ingests NBA play-by-play data at scale, processes it into a structured analytical format, stores it in a local analytical database, and generates meaningful insights about team momentum, scoring volatility, foul patterns, and pace.

**Key Questions:**
- Which teams experience long scoring droughts?
- Which teams rely on burst scoring for momentum?
- Do referee crews show consistent foul calling patterns?
- How does team pace vary by quarter?

## Project Structure

```
nba-playbyplay-pipeline/
├── ingest/
│   ├── fetch_games.py          # Fetch list of all games for a season
│   └── fetch_playbyplay.py     # Fetch play-by-play data for each game
├── pipeline/
│   ├── normalize_data.py       # Normalize and clean play-by-play data
│   ├── load_duckdb.py          # Load data into DuckDB
│   └── prefect_flow.py         # Orchestrate the entire pipeline
├── analysis/
│   ├── compute_metrics.py      # Compute analytical metrics
│   └── plots.py                # Generate visualizations
├── data/
│   ├── raw/                    # Raw JSON files from API
│   │   └── playbyplay/         # Individual game play-by-play files
│   └── processed/              # Parquet files and DuckDB database
├── config/
│   └── nba_config.yml          # Configuration file
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Internet connection for API access

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Season

Edit `config/nba_config.yml` to set the season year:

```yaml
season:
  year: 2023  # Season year (e.g., 2023 for 2023-24 season)
  season_type: "Regular Season"  # Regular Season, Playoffs, or All Star
```

### 4. Verify Setup

You can test the setup by running:

```bash
python -c "import requests, pandas, duckdb, prefect; print('All dependencies installed!')"
```

## Usage

### Option 1: Run Individual Scripts (Recommended for First Run)

Run each step sequentially to see progress and debug any issues:

```bash
# Step 1: Fetch list of all games for the season
python ingest/fetch_games.py

# Step 2: Fetch play-by-play data for each game
python ingest/fetch_playbyplay.py

# Step 3: Normalize and clean the data
python pipeline/normalize_data.py

# Step 4: Load data into DuckDB
python pipeline/load_duckdb.py

# Step 5: Compute analytical metrics
python analysis/compute_metrics.py

# Step 6: Generate visualizations
python analysis/plots.py
```

### Option 2: Run Prefect Pipeline (Automated)

Run the entire pipeline with Prefect orchestration:

```bash
python pipeline/prefect_flow.py
```

This will execute all steps in sequence with logging and error handling.

## Data Pipeline Details

### 1. Ingestion (`ingest/`)

**`fetch_games.py`**
- Uses NBA Stats API to fetch all games for a specified season
- Handles API rate limiting and errors
- Outputs: `data/raw/games.json`

**`fetch_playbyplay.py`**
- Iterates over each game and fetches detailed play-by-play event data
- Each game contains hundreds of events (shots, fouls, turnovers, etc.)
- Handles rate limits, pagination, and large responses
- Outputs: `data/raw/playbyplay/{game_id}.json`

### 2. Processing (`pipeline/`)

**`normalize_data.py`**
- Processes raw JSON play-by-play files
- Normalizes inconsistent schemas into a flat table structure
- Handles missing values and data type conversions
- Calculates derived fields (game time, score margins, etc.)
- Outputs: `data/processed/playbyplay_normalized.parquet`

**`load_duckdb.py`**
- Creates DuckDB database: `data/processed/nba_playbyplay.duckdb`
- Loads Parquet files into analytical database
- Creates tables for fast querying
- Outputs: DuckDB database with `playbyplay` and `games` tables

### 3. Analysis (`analysis/`)

**`compute_metrics.py`**
- Computes scoring runs (consecutive scoring possessions)
- Computes scoring droughts (consecutive non-scoring possessions)
- Calculates foul frequencies by quarter and minute
- Computes pace metrics (possessions per minute) by quarter
- Calculates momentum measures (rolling point differences)
- Outputs: DuckDB tables + CSV files

**`plots.py`**
- Generates 5 visualizations:
  1. **Scoring run distribution**: Histogram and box plot of scoring run lengths
  2. **Drought length by team**: Average drought length and distribution by team
  3. **Foul frequency heatmap**: Fouls by period and game minute
  4. **Pace by quarter**: Possessions per minute across quarters
  5. **Momentum analysis**: Score margin and momentum over time
- Outputs: PNG files in `analysis/plots/`

## Key Metrics & Analysis

### Scoring Runs
- **Definition**: Consecutive scoring possessions by a team
- **Insight**: Identifies teams that rely on burst scoring for momentum
- **Metric**: Run length distribution, average run length by team

### Scoring Droughts
- **Definition**: Consecutive non-scoring possessions
- **Insight**: Identifies teams that experience long scoring droughts
- **Metric**: Drought length distribution, average drought length by team

### Foul Frequencies
- **Definition**: Foul counts by period and game minute
- **Insight**: Reveals patterns in referee foul calling (e.g., end-of-quarter fouls)
- **Metric**: Average fouls per game by period-minute combination

### Pace Metrics
- **Definition**: Possessions per minute by quarter
- **Insight**: Shows how team pace varies by quarter
- **Metric**: Average pace, pace distribution by period

### Momentum Measures
- **Definition**: Rolling point differences across possessions
- **Insight**: Tracks game momentum shifts
- **Metric**: 5-possession rolling average of score margin changes

## Results & Insights

### Expected Findings

1. **Scoring Patterns**: Teams with high variance in scoring runs vs. consistent scoring
2. **Drought Identification**: Teams that frequently experience long scoring droughts
3. **Foul Patterns**: Referee crews showing consistent foul calling patterns (e.g., more fouls at end of quarters)
4. **Pace Variation**: Teams whose pace varies significantly by quarter (e.g., slower in 4th quarter)

### Visualizations

All plots are saved to `analysis/plots/`:

- `scoring_run_distribution.png`: Distribution of scoring run lengths
- `drought_length_by_team.png`: Average drought length by team
- `foul_frequency_heatmap.png`: Foul frequency by period and minute
- `pace_by_quarter.png`: Pace metrics across quarters
- `momentum_analysis.png`: Momentum and score margin over time

## Challenges & Solutions

### Rate Limiting

**Challenge**: NBA Stats API has rate limits and may block requests.

**Solution**:
- Implemented configurable delays between requests
- Added retry logic with exponential backoff
- Respects API response headers

### Inconsistent Schemas

**Challenge**: NBA API responses can have inconsistent field names and structures.

**Solution**:
- Normalization script handles multiple field name variations
- Robust parsing of time strings and scores
- Handles missing values gracefully

### Large Response Sizes

**Challenge**: Play-by-play data for a full season contains 200,000+ events.

**Solution**:
- Use Parquet format for efficient storage
- DuckDB for fast analytical queries
- Batch processing where possible

### Data Quality

**Challenge**: Some games may have incomplete or missing play-by-play data.

**Solution**:
- Error handling for missing games
- Validation of data completeness
- Logging of failed requests for manual review

## Configuration

Edit `config/nba_config.yml` to customize:

- Season year and type
- API rate limit delays
- Retry settings
- Batch sizes

## Reproducibility

This project is fully reproducible:

1. All random seeds are set (where applicable)
2. All data is saved to disk (JSON + Parquet + DuckDB)
3. All scripts are deterministic (no manual steps required)
4. Environment is specified (`requirements.txt`)

To reproduce:

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Configure season in config/nba_config.yml

# 3. Run pipeline
python pipeline/prefect_flow.py

# 4. Run analysis
python analysis/compute_metrics.py
python analysis/plots.py
```

## Data Source

**NBA Stats API**: https://stats.nba.com/

The API provides:
- Game schedules and metadata
- Detailed play-by-play event data
- Real-time and historical statistics

**Note**: The NBA Stats API is a public API but may have usage restrictions. This project uses reasonable rate limiting to respect the API.

## Technical Stack

- **Python**: Data processing and analysis
- **Requests**: API calls
- **Pandas**: Data manipulation
- **DuckDB**: Analytical database
- **Prefect**: Workflow orchestration
- **Matplotlib/Seaborn**: Visualizations
- **Parquet**: Efficient data storage

## Future Enhancements

Potential improvements:

1. **Real-time updates**: Schedule periodic data refreshes
2. **More metrics**: Advanced statistics (e.g., shot quality, defensive metrics)
3. **Player-level analysis**: Individual player momentum and impact
4. **Predictive modeling**: Predict momentum shifts or game outcomes
5. **Interactive dashboards**: Web-based visualization tools

## License

This project is for educational purposes as part of DS3022 Data Engineering course.

## Acknowledgments

- NBA Stats API for providing access to play-by-play data
- DuckDB for fast analytical queries
- Prefect for workflow orchestration

## Contact

For questions or issues, please open an issue on GitHub.
