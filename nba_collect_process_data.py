import pandas as pd
import numpy as np

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.http import NBAStatsHTTP

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)  # print all dataframe columns, regardless of how many there are
pd.set_option('display.width', 1000)  # prevent horizontal wrapping in print by setting a very large display width
pd.set_option('display.max_rows', 5)  # set the maximum number of rows to display before truncation (...) starts
pd.options.mode.chained_assignment = None  # done to prevent warnings later on reformatted columns

from utility_functions import get_opponent_id

# -------------------------------------------------------------------------------------------------------------

# COLLECT DATA

"""
We base our model off of regular season games played by all NBA teams.
Let's collect game data from the nba_api and store in a dataframe.
"""

nba_teams = teams.get_teams()  # provides a list of 30 team dictionaries containing team id, name, abbreviation, city, state, etc.
team_abbr_to_id = {team["abbreviation"]: team["id"] for team in nba_teams}  # make a map from team abbreviation to team id

# data was collected via a script which does a one-time pull from nba_api into a parquet file
all_games = pd.read_parquet('nba_all_team_games.parquet', engine='pyarrow')

print(all_games.sample(n=5))
print()

# -------------------------------------------------------------------------------------------------------------

# CLEAN AND REFORMAT DATA

"""
Let's reformat and clean some of the data to make it easier to work with and more trainable.
"""

# Convert GAME_DATE to pandas datetime
all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])

# Order by date earliest to latest to make viewing and working with stats easier
all_games.sort_values(by="GAME_DATE", inplace=True)

# Add binary "WIN" column, remove categorical WL column (optional not done yet)
all_games["WIN"] = all_games["WL"].apply(lambda x: 1 if x == "W" else 0)

print(all_games.tail(5))
print()

# Add opponent IDs to the dataframe
all_games["OPP_TEAM_ID"] = all_games.apply(
    lambda row: get_opponent_id(row["MATCHUP"], team_abbr_to_id, row["TEAM_ID"]), axis=1
)

print(all_games.tail(5))
print()

# Convert int stat columns to float type for more accurate data analysis
all_games["MIN"] = all_games["MIN"].astype(float)  # minutes
all_games["PTS"] = all_games["PTS"].astype(float)  # points
all_games["FGM"] = all_games["FGM"].astype(float)  # field goals made
all_games["FGA"] = all_games["FGA"].astype(float)  # field goals attempted
all_games["FG3M"] = all_games["FG3M"].astype(float)  # 3s made
all_games["FG3A"] = all_games["FG3A"].astype(float)  # 3s attempted
all_games["FTM"] = all_games["FTM"].astype(float)  # free throws made
all_games["FTA"] = all_games["FTA"].astype(float)  # free throws attempted
all_games["REB"] = all_games["REB"].astype(float)  # rebounds
all_games["OREB"] = all_games["OREB"].astype(float)  # offensive rebounds
all_games["DREB"] = all_games["DREB"].astype(float)  # defensive rebounds
all_games["AST"] = all_games["AST"].astype(float)  # assists
all_games["BLK"] = all_games["BLK"].astype(float)  # blocks
all_games["TOV"] = all_games["TOV"].astype(float)  # turnovers
all_games["PF"] = all_games["PF"].astype(float)  # personal fouls

print(all_games.tail(5))
print()

# all_games.to_parquet('nba_all_team_games_cleaned.parquet', index=False)

# -------------------------------------------------------------------------------------------------------------

# FEATURE ENGINEER

"""
Let's add a few new statistics that will serve as better features and possibly improve our model.
We indicate to the model if a team had home game advantage, which has an impact on the away team.
We indicate to the model the outcome of the team's last game, as a prior win or loss could momentum going into the next game.
We then calculate the "Four Factors". These statistics track a team's shooting, turnovers, rebounding, and free throws,
and are arguably strong metrics for predicting team success in basketball.
Finally, we calculate a team's true shooting percentage in each game, which is a more nuanced measure of shooting efficiency.
"""

# Define 'HGA' (Home Game Advantage)
all_games["HGA"] = all_games["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)

# Define 'LAST_GAME_OUTCOME'
all_games["LAST_GAME_OUTCOME"] = (
    all_games.groupby("TEAM_ID")["WIN"].shift(1).fillna(0)
)

# Define 'EFG%' (Effective Field Goal Percentage)
all_games["EFG%"] = (
    all_games["FGM"] + (0.5 * all_games["FG3M"])
) / all_games["FGA"]

# Define 'TOV%' (Turnover Percentage)
all_games["TOV%"] = all_games["TOV"] / (
    all_games["FGA"] + 0.44 * all_games["FTA"] + all_games["TOV"]
)

# Define 'FTR' (Free Throw Attempt Rate)
all_games["FTR"] = all_games["FTA"] / all_games["FGA"]

# Define 'TS%' (True Shooting Percentage)
all_games["TS%"] = all_games["PTS"] / (
    2 * (all_games["FGA"] + (0.44 * all_games["FTA"]))
)


print(all_games.head(5))


# all_games.to_parquet('nba_all_team_games_enriched.parquet', index=False)


# nba_games = pd.read_parquet('nba_all_team_games_enriched.parquet', engine='pyarrow')


# print(nba_games.head(5))


"""
Adjust the games we train on to only be more modern games, as the NBA has changed greatly over time.
"""

# Get games where season is in modern_seasons
# modern_games = all_games[
#     (all_games.SEASON_ID.str[-4:] == "2019")
#     | (all_games.SEASON_ID.str[-4:] == "2020")
#     | (all_games.SEASON_ID.str[-4:] == "2021")
#     | (all_games.SEASON_ID.str[-4:] == "2022")
#     | (all_games.SEASON_ID.str[-4:] == "2023")
#     | (all_games.SEASON_ID.str[-4:] == "2024")
# ]