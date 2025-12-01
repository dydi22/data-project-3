Team DODA
Team members
Dylan Dietrich and Oliver Andress
Data Source

What data source did you work with?
NBA Stats API - Play-by-play event data for the 2023-24 NBA Regular Season. The dataset includes 1,230 games with 200,000+ individual events (shots, fouls, turnovers, rebounds, etc.) with timestamps, scores, player actions, and game context. Data was accessed via the leaguegamefinder and playbyplayv2 endpoints.
Challenges / Obstacles

What challenges did this data choice present in data gathering, processing and analysis, and how did you work through them? What methods and tools did you use to work with this data?
The NBA Stats API blocks automated requests and has strict rate limits, causing initial 100% failure rates. We addressed this by using the nba-api Python package for authentication, implementing configurable rate limiting (1 second delays), retry logic with exponential backoff, and connection pooling. 
The API also returns inconsistent schemas across game types—some events include player information, others don't. We created a normalization pipeline with flexible column mapping and fallback logic to handle missing columns. Processing 200,000+ events required efficient storage and querying, so we used Parquet for compressed columnar storage and DuckDB for fast analytical queries without a full database server. 
For complex metrics like comeback probabilities and timeout effectiveness, we implemented stateful processing in Python with pandas and used SQL window functions in DuckDB to track game state over time.
Tools used: Python, pandas, DuckDB, Parquet, Prefect (orchestration), nba-api package, matplotlib/seaborn (visualization), requests library

Analysis
Offer a brief analysis of the data with your findings. Keep it to one brief, clear, and meaningful paragraph.
Our analysis reveals four insights: 
(1) Comeback win probability decreases exponentially with larger deficits and less time remaining—teams down 10 points with 5 minutes left have only ~20% chance of winning, while teams down 5 points with 10 minutes left have ~40-50% chance.
(2) Clutch performance shows high variability: in the final 5 minutes of close games, total scoring averages 20-25 points but ranges from 0 to 40+ points, indicating some games have explosive finishes while others become defensive battles. 
(3) Timeout effectiveness is mixed—only 50-55% of timeouts result in the calling team outscoring their opponent in the next 2 minutes, suggesting timeout strategy may need improvement. 
(4) Games that are within 5 points with 5 minutes remaining in regulation are 3-4x more likely to go to overtime, and overtime games have smaller maximum leads throughout regulation, showing they stay competitive from start to finish.
The information that we uncovered can be used my mutiple stakeholders. The first group is your avid sportsbetter, they can see the likelihood of different types of comebacks and can potentially find arbitrage between 

Plot / Visualization
Include at least one compelling plot or visualization of your work. Add images in your subdirectory and then display them using markdown in your README.md file.
We created four visualizations:
    1    Comeback Win Probability Heatmap (analysis/plots/comeback_probability.png): Shows the probability of winning when trailing by a specific deficit with a certain number of minutes remaining. Reveals that comeback probability decreases exponentially with both larger deficits and less time remaining.
    1    Clutch Performance Distribution (analysis/plots/clutch_performance.png): Histogram showing total points scored by both teams in the final 5 minutes when the game is within 5 points. Reveals high variability in clutch moment intensity.
    1    Timeout Effectiveness Analysis (analysis/plots/timeout_effectiveness.png): Histogram showing net score change (team that called timeout minus opponent) in the 2 minutes after each timeout. Green bars indicate effective timeouts, red bars indicate ineffective ones.
    1    Overtime Predictors (analysis/plots/overtime_predictors.png): Multi-panel visualization showing what predicts overtime games, including margin with 5 minutes left, maximum lead during regulation, and final margin distribution.
GitHub Repository
https://github.com/dydi22/data-project-3

