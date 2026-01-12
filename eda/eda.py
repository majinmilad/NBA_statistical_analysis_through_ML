import pandas as pd
import matplotlib.pyplot as plt
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
Collect game data from the nba_api and store in a dataframe.
"""

nba_teams = teams.get_teams()  # provides a list of 30 team dictionaries containing team id, name, abbreviation, city, state, etc.
team_abbr_to_id = {team["abbreviation"]: team["id"] for team in nba_teams}  # make a map from team abbreviation to team id

# data was collected via a script which does a one-time pull from nba_api into a parquet file
all_games = pd.read_parquet('nba_all_team_games.parquet', engine='pyarrow')

print(all_games.sample(n=5))
print()

# -------------------------------------------------------------------------------------------------------------

# CLEAN AND REFORMAT DATA

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


def load_data(path="nba_all_team_games_enriched.parquet"):
    df = pd.read_parquet(path)

    # make sure GAME_DATE is a datetime if present
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # create SEASON_YEAR from SEASON_ID if needed
    if "SEASON_YEAR" not in df.columns and "SEASON_ID" in df.columns:
        # SEASON_ID is something like 21998, 22023, etc.
        # Take the LAST 4 digits -> 1998, 2023, ...
        season_str = df["SEASON_ID"].astype(str)
        year_str = season_str.str[-4:]  # last 4 characters

        # convert to numeric, drop rows where this fails
        year_numeric = pd.to_numeric(year_str, errors="coerce")
        df = df[~year_numeric.isna()].copy()
        df["SEASON_YEAR"] = year_numeric.astype(int)

    return df

def win_rates(df):
    print(" WIN RATES \n")

    # overall win rate
    overall_win_rate = df["WIN"].mean()
    print(f"Overall win rate: {overall_win_rate:.3f}")

    # home vs away (assuming HGA: 1 = home, 0 = away)
    if "HGA" in df.columns:
        home_mask = df["HGA"] == 1
        away_mask = df["HGA"] == 0

        home_win_rate = df.loc[home_mask, "WIN"].mean()
        away_win_rate = df.loc[away_mask, "WIN"].mean()

        print(f"Home win rate: {home_win_rate:.3f}")
        print(f"Away win rate: {away_win_rate:.3f}")

        # bar chart of home vs away win rate
        plt.figure(figsize=(5, 4))
        categories = ["Away", "Home"]
        values = [away_win_rate, home_win_rate]
        plt.bar(categories, values)
        plt.ylim(0, 1)
        plt.ylabel("Win rate")
        plt.title("Win rate by home/away")
        plt.tight_layout()
        plt.show()

    # back-to-back win rate
    if "BACK_TO_BACK" in df.columns:
        b2b_mask = df["BACK_TO_BACK"] == 1
        non_b2b_mask = df["BACK_TO_BACK"] == 0

        # win rate out of *all back-to-back games*
        b2b_win_rate = df.loc[b2b_mask, "WIN"].mean()
        non_b2b_win_rate = df.loc[non_b2b_mask, "WIN"].mean()

        print(f"\nBack-to-back win rate (only B2B games): {b2b_win_rate:.3f}")
        print(f"Non-back-to-back win rate:             {non_b2b_win_rate:.3f}")

        # bar chart comparing B2B vs non-B2B win rates
        plt.figure(figsize=(5, 4))
        categories = ["Non-B2B", "B2B"]
        values = [non_b2b_win_rate, b2b_win_rate]
        plt.bar(categories, values)
        plt.ylim(0, 1)
        plt.ylabel("Win rate")
        plt.title("Win rate: Back-to-back vs Non-back-to-back")
        plt.tight_layout()
        plt.show()

def plot_histograms(df):
    # box-score stats
    box_score_cols = [
        "PTS", "REB", "OREB", "DREB", "AST",
        "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "MIN"
    ]
    box_score_cols = [c for c in box_score_cols if c in df.columns]

    # advanced stats
    adv_cols = ["EFG%", "TS%", "TOV%", "FTR"]
    adv_cols = [c for c in adv_cols if c in df.columns]

    # histograms for box-score stats
    if box_score_cols:
        n = len(box_score_cols)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        plt.figure(figsize=(4 * ncols, 3 * nrows))
        for i, col in enumerate(box_score_cols, start=1):
            plt.subplot(nrows, ncols, i)
            data = df[col].dropna()
            plt.hist(data, bins=30)
            plt.title(col)
        plt.suptitle("Box-score stat distributions", y=1.02)
        plt.tight_layout()
        plt.show()

    # histograms for advanced stats
    if adv_cols:
        n = len(adv_cols)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        plt.figure(figsize=(4 * ncols, 3 * nrows))
        for i, col in enumerate(adv_cols, start=1):
            plt.subplot(nrows, ncols, i)
            data = df[col].dropna()
            plt.hist(data, bins=30)
            plt.title(col)
        plt.suptitle("Advanced stat distributions", y=1.02)
        plt.tight_layout()
        plt.show()


def temporal_trends(df):
    if "SEASON_YEAR" not in df.columns:
        print("\nNo SEASON_YEAR column available, skipping temporal trends.")
        return

    group_cols = ["PTS", "TS%", "REB"]
    group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols:
        print("\nNo numeric stat columns for trends, skipping temporal trends.")
        return

    seasonal = (
        df.groupby("SEASON_YEAR")[group_cols]
        .mean()
        .reset_index()
        .sort_values("SEASON_YEAR")
    )

    # restrict to 1985–2024
    seasonal = seasonal[
        (seasonal["SEASON_YEAR"] >= 1985) & (seasonal["SEASON_YEAR"] <= 2024)
    ]

    print("\nSeasonal league averages (first few rows):")
    print(seasonal.head())

    # one line chart per metric
    for col in group_cols:
        plt.figure(figsize=(8, 4))
        plt.plot(seasonal["SEASON_YEAR"], seasonal[col], marker="o")
        years = seasonal["SEASON_YEAR"].values
        # set year labels explicitly as strings
        plt.xticks(years, [str(y) for y in years], rotation=45)
        plt.xlabel("Season year")
        plt.ylabel(col)
        plt.title(f"League-average {col} by season (1985–2024)")
        plt.tight_layout()
        plt.show()


def main():
    df = load_data()
    print("Data loaded. Shape:", df.shape)

    win_rates(df)
    plot_histograms(df)
    temporal_trends(df)


if __name__ == "__main__":
    main()

