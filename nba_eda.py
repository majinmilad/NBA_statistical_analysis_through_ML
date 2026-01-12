import pandas as pd
import matplotlib.pyplot as plt

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
