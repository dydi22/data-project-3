# NBA Play-by-Play Data Analysis: Team Momentum, Scoring Patterns, and Pace

## Team Members
Dylan Dietrich

## Data Source
**NBA Stats API** - Play-by-play event data for the 2023-24 NBA Regular Season
- **Scale**: 1,230 games, 200,000+ individual events (shots, fouls, turnovers, rebounds, etc.)
- **Data Type**: Real-time event-level data with timestamps, scores, player actions, and game context
- **API Endpoints**: 
  - `leaguegamefinder` for game schedules
  - `playbyplayv2` for detailed event data

## Challenges & Solutions

### Challenge 1: API Rate Limiting & Blocking
The NBA Stats API aggressively blocks automated requests and has strict rate limits. Initial attempts resulted in 100% failure rates.

**Solution**: 
- Implemented the `nba-api` Python package which handles authentication and headers automatically
- Added configurable rate limiting (1 second delays between requests)
- Implemented retry logic with exponential backoff
- Used connection pooling with requests.Session for efficiency

### Challenge 2: Inconsistent Data Schemas
The API returns different field names and structures across different game types and events. Some events have player information, others don't.

**Solution**:
- Created a robust normalization pipeline that handles missing columns gracefully
- Implemented flexible column mapping with fallback logic
- Added data validation to ensure consistency before analysis

### Challenge 3: Large-Scale Data Processing
Processing 200,000+ events required efficient storage and querying.

**Solution**:
- Used **Parquet** format for compressed, columnar storage
- Implemented **DuckDB** for fast analytical queries without a full database server
- Created normalized schema with proper indexing for time-series queries

### Challenge 4: Complex Metric Computation
Computing momentum, scoring runs, and droughts required tracking state across events and games.

**Solution**:
- Implemented stateful processing in Python with pandas
- Used window functions and rolling calculations for momentum metrics
- Created reusable functions for each metric type with proper error handling

## Tools & Architecture

**Pipeline Architecture**:
1. **Ingestion** (`ingest/`): Python scripts using `requests` and `nba-api` package
2. **Normalization** (`pipeline/normalize_data.py`): Data cleaning and schema standardization
3. **Storage** (`pipeline/load_duckdb.py`): Parquet → DuckDB for analytical queries
4. **Orchestration** (`pipeline/prefect_flow.py`): Prefect workflow for end-to-end automation
5. **Analysis** (`analysis/compute_metrics.py`): Metric computation using DuckDB SQL + pandas
6. **Visualization** (`analysis/plots.py`): Matplotlib/Seaborn for publication-ready plots

**Key Tools**:
- **DuckDB**: Fast analytical database for complex queries
- **Prefect**: Workflow orchestration for reproducible pipelines
- **Parquet**: Efficient columnar storage format
- **Pandas**: Data manipulation and analysis
- **NBA-API**: Python wrapper for NBA Stats API

## Key Findings

### 1. Scoring Runs Follow Power Law Distribution
Most scoring runs are short (1-2 consecutive scores), but teams occasionally achieve extended runs of 5+ consecutive scores. The average run length is 1.62 possessions, indicating that sustained scoring bursts are relatively rare.

### 2. Pace Varies Significantly by Quarter
Teams play at an average of 6.54 possessions per minute, but pace varies dramatically by quarter. The analysis reveals that teams tend to slow down in later quarters, with the 4th quarter showing the most variation in pace—likely due to strategic timeouts and intentional fouling.

### 3. Foul Patterns Show Temporal Clustering
Foul frequency is not uniformly distributed throughout games. The heatmap analysis reveals clustering around specific game minutes, particularly at the end of quarters. This suggests strategic fouling behavior that isn't visible in traditional box scores.

### 4. Momentum Shifts Are Frequent but Short-Lived
Rolling point differential analysis shows that momentum (measured as 5-possession rolling averages) changes frequently throughout games. Teams rarely maintain sustained momentum for extended periods, with most momentum shifts lasting only a few possessions.

### 5. Scoring Droughts Are Common
Analysis of non-scoring possessions reveals that teams frequently experience droughts of 3+ consecutive possessions without scoring. These droughts are more common than extended scoring runs, highlighting the defensive nature of modern NBA basketball.

## Visualizations

![Scoring Run Distribution](scoring_run_distribution.png)

This visualization shows the distribution of scoring run lengths across all games. The histogram reveals that most runs are short (1-2 scores), with longer runs becoming increasingly rare—demonstrating the difficulty of maintaining sustained offensive momentum in professional basketball.

![Foul Frequency Heatmap](foul_frequency_heatmap.png)

The heatmap reveals temporal patterns in foul calling, showing increased foul frequency at specific game minutes. This pattern is invisible in traditional box scores but becomes clear when analyzing event-level data.

![Pace by Quarter](pace_by_quarter.png)

This analysis shows how team pace (possessions per minute) varies across quarters, revealing strategic adjustments that teams make throughout games.

![Momentum Analysis](momentum_analysis.png)

The momentum visualization tracks rolling point differentials across sample games, showing how game momentum shifts frequently and rarely sustains for extended periods.

## Insights Beyond Box Scores

This analysis reveals patterns that are **impossible to see in traditional box scores**:

1. **Strategic Foul Timing**: The foul frequency heatmap shows teams intentionally foul at specific game moments, a strategy invisible in aggregate statistics.

2. **Pace Manipulation**: Teams actively adjust their pace by quarter, with significant slowdowns in critical moments—information lost when looking only at total game statistics.

3. **Momentum Volatility**: The frequent, short-lived momentum shifts suggest that NBA games are more volatile than aggregate scores indicate, with teams rarely maintaining extended advantages.

4. **Scoring Drought Patterns**: The prevalence of scoring droughts highlights the defensive sophistication of modern NBA teams, showing that even elite offenses struggle with consistency.

## Repository

**GitHub Repository**: https://github.com/dydi22/data-project-3

The repository contains:
- Complete data pipeline code
- Configuration files
- Analysis scripts
- Generated visualizations
- Documentation for reproducibility

## Technical Achievements

- **Scale**: Processed 200,000+ events from 1,230 games
- **Reproducibility**: Fully automated pipeline with Prefect orchestration
- **Performance**: Efficient storage and querying with DuckDB and Parquet
- **Robustness**: Handles API failures, missing data, and schema inconsistencies
- **Insights**: Reveals patterns invisible in traditional basketball statistics

