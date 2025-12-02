# Team DODA

**Team Members:** Dylan Dietrich and Oliver Andress

## Data Source
Our project uses play by play event data from the NBA Stats API for the 2023 to 2024 NBA Regular Season. The dataset covers 1,230 games and includes more than 200,000 individual events such as shots, fouls, turnovers, rebounds, score changes, timestamps, player actions, and detailed game context. Data was retrieved using the `leaguegamefinder` and `playbyplayv2` endpoints, allowing us to reconstruct game flow at a granular, event by event level and analyze how games evolve over time.

## Challenges and Obstacles
Working with the NBA Stats API was not as easy as it seemed. The API has strict automated requests blocks and therefore we used the nba-api package for authentication and implemented one second delays between requests. Once we are able to run the API it turned out that it was way more data than we expected and it would have taken around a day to load. It turned out we were downloading the data twice, all the stats for the home team, and also for the away team, and therefore were double counting games. We then started by only fetching a few games to get familiar with the data and how it will come before downloading all the data points.
Another challenge was inconsistent schemas across different game events, where some records contained player level details and others did not. We solved this by building a normalization pipeline with flexible column mapping and fallback logic to handle missing values gracefully. Managing over 200,000 events also required efficient data storage and querying, so we stored processed data in Parquet format and used DuckDB f without needing a full database server as we learned it in class. For more the more complex metrics such as comeback probabilities and timeout effectiveness, we used pandas with SQL window functions in DuckDB to track game state continuously over time.

## Analysis and Findings
Our analysis produced three meaningful insights with clear practical implications for teams, analysts, and fans. First, the comeback win probability heatmap shows that win likelihood declines sharply as deficits grow and time remaining decreases. This reinforces the strategic importance of early scoring runs and preventing large deficits before late game situations, since even strong teams face rapidly diminishing chances of winning when trailing by double digits late in games. Analysts and broadcasters can use this to contextualize late game “comeback” narratives with actual probabilities rather than intuition.
Sports betters can also use this information to find arbitrage between probabilities that are given in a given game and the overall likelihood of a comeback happening. 
Second, overtime analysis reveals that games within five points with five minutes remaining in regulation are several times more likely to go to overtime than games with larger margins. Although this is intuitive it allows for specific figures for sports betters and viewers alike to see the probability of games going into overtime. These games also tend to remain competitive throughout regulation, as shown by smaller maximum leads. This insight is useful for league planners, broadcasters, and fans, as it explains why certain games feel competitive well before the final buzzer and why late game tension is a strong predictor of overtime rather than final margin alone.
Third, timeouts are not as effective as assumed. Fewer than half of timeouts result in the calling team outscoring their opponent in the following two minutes, and the average net score change after a timeout is slightly negative. This suggests that timeouts do not reliably shift momentum and may not provide the expected scoring advantage. Coaches and teams could use this finding to reconsider timeout timing, potentially saving timeouts for defensive stops, rest, or late game clock management rather than expecting immediate offensive improvement.

Overall, these findings highlight how play by play data can challenge conventional basketball assumptions, offering data driven insights into momentum, strategy, and game management at the professional level.

## Visualizations

### Home Team Comeback Win Probability
This heatmap shows the probability that the home team eventually wins when trailing, based on point deficit and minutes remaining. It highlights the steep decline in comeback likelihood as both deficit and time pressure increase.

![Home Team Comeback Win Probability](analysis/plots/comeback_probability.png)

### Overtime Predictors
These plots demonstrate that games within five points with five minutes remaining are several times more likely to go to overtime. Overtime games also show smaller maximum leads during regulation, indicating sustained competitiveness from start to finish.

![Overtime Predictors](analysis/plots/overtime_predictors.png)

### Timeout Effectiveness
This histogram shows net score change in the two minutes following a timeout. Positive values indicate the team that called timeout outscored their opponent, while negative values indicate the opposite. The distribution shows that timeouts are effective less than half the time.

![Timeout Effectiveness](analysis/plots/timeout_effectiveness.png)

## Repository
All code, data pipelines, and visualizations are available at:  
https://github.com/dydi22/data-project-3
