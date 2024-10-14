import numpy as np
import pandas as pd
import nfl_data_py as nfl
import category_encoders as ce
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# get schedule info to assign home/away
schedule_22 = nfl.import_schedules(years=[2022])
schedule_23 = nfl.import_schedules(years=[2023])
home_22 = schedule_22[["week", "home_team"]]
home_23 = schedule_23[["week", "home_team"]]

# encodes whether player is playing at home (1) or away (0)
#   - with this we hope to capture the effects of home-field advantage on fantasy performance
# returns DataFrame with new column indicating home/away
def encode_home_away(df, home):
    for week in pd.unique(home["week"]):
        homes = [team for team in pd.unique(home.loc[home.week==week]["home_team"])]
        df.loc[df.week==week, "home_away"] = df.loc[df.week==week].apply(lambda x: 1 if x["recent_team"] in homes else 0, axis=1)
    return df

# applies binary encoding to player's team and opponent team
# each team has two different encodings:
#   - one corresponding to the team a player is on, technically representing that team's offense
#   - one corresponding to the team a player is against, technically representing that team's defense
#   - with this we hope to capture how a player's fantasy performance is affected by the skill of the offense around them and the skill of the defense facing them
# returns DataFrame with binary columns corresponding to this encoding
def encode_teams(df, encoder):
    try:
        check_is_fitted(encoder)
        encoded = encoder.transform(df)
    except NotFittedError as exc:
        encoded = encoder.fit_transform(df)
    return encoded

# rolling average function shifted so we are using stats from the weeks prior, not from the current week, to predict current week performance
def rolling_average(df, window):
    return df.rolling(min_periods=1, window=window).mean().shift(1)

# applies rolling average to each numerical stat
# returns average over season, average over last 4 weeks, and last week's stats
# as well as names of averaged columns to be used as features
def get_rolling_averages(df, stats):
    feature_names = []
    for stat in df[stats]:
        result1 = df.groupby("player_display_name")[stat].apply(lambda x: rolling_average(x, 16))
        df['season_{}'.format(stat)] = result1.droplevel(0)
        result2 = df.groupby("player_display_name")[stat].apply(lambda x: rolling_average(x, 4))
        df['recent_{}'.format(stat)] = result2.droplevel(0)
        result3 = df.groupby("player_display_name")[stat].apply(lambda x: rolling_average(x, 1))
        df['prev_{}'.format(stat)] = result3.droplevel(0)
        feature_names = feature_names + [time + "_" + stat for time in ['season', 'recent', 'prev']]
    return df, feature_names

# applies categorical encoding and rolling averages to stats
# returns processed DataFrame and full list of features
def data_processing(file_name, home, encoder):
    df = pd.read_csv(file_name, index_col=0)
    df.sort_values(by=["player_display_name", "week"])
    df = encode_home_away(df, home)
    df = encode_teams(df, encoder)
    numerical_stats = [column for column in df.columns if df[column].dtype != "object"
                       and not "recent_team" in column and not "opponent_team" in column
                       and not column in ["season", "week", "home_away"]]
    df, averaged_features = get_rolling_averages(df, numerical_stats)
    encoded_features = [column for column in df.columns if "recent_team" in column
                        or "opponent_team" in column] + ["home_away"]
    full_features = averaged_features + encoded_features
    df = df.fillna(0)
    return df, encoded_features, averaged_features, full_features

team_encoder = ce.BinaryEncoder(cols=["recent_team", "opponent_team"])
processed_22, encoded_feat_22, averaged_feat_22, full_feat_22 = data_processing("~/stuff/all_2022.csv", home_22, team_encoder)
processed_23, encoded_feat_23, averaged_feat_23, full_feat_23 = data_processing("~/stuff/all_2023.csv", home_23, team_encoder)

# processed_22.to_csv("full_22_processed.csv", index=True, header=True)
# processed_23.to_csv("full_23_processed.csv", index=True, header=True)
