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
# returns DataFrame with new column indicating home/away
def encode_home_away(df, home):
    for week in pd.unique(home["week"]):
        homes = [team for team in pd.unique(home.loc[home.week==week]["home_team"])]
        df.loc[df.week==week, "home_away"] = df.loc[df.week==week].apply(lambda x: 1 if x["recent_team"] in homes else 0, axis=1)
    return df

# applies binary encoding to player's team and opponent team
# returns DataFrame with binary columns representing own team and opponent team
def encode_teams(df, encoder):
    try:
        check_is_fitted(encoder)
        encoded = encoder.transform(df)
    except NotFittedError as exc:
        encoded = encoder.fit_transform(df)
    return encoded

# rolling average function shifted so we are using stats 
# from the weeks prior to predict this week's fantasy
# point total
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

# all_22_encoded = get_encoded_categoricals(all_2022, home_22)
# all_23_encoded = get_encoded_categoricals(all_2023, home_23)
# all_22_encoded.to_csv("22_encoded_v1.csv", header=True, index=True)
# all_23_encoded.to_csv("23_encoded_v1.csv", header=True, index=True)


"""
team_encoder = ce.BinaryEncoder()
own_team_22 = all_2022[["recent_team"]]
own_team_22_binary = team_encoder.fit_transform(own_team_22["recent_team"].values)
own_team_22 = own_team_22.join(own_team_22_binary)
own_team_22 = own_team_22.drop_duplicates("recent_team").set_index("recent_team")
own_team_22 = own_team_22.rename(columns={"0_0": "recent_0", "0_1": "recent_1", "0_2": "recent_2", "0_3": "recent_3", "0_4": "recent_4", "0_5": "recent_5"})
all_2022["recent_0"] = own_team_22.loc[all_2022.recent_team].recent_0
all_2022["recent_1"] = own_team_22.loc[all_2022.recent_team].recent_1
all_2022["recent_2"] = own_team_22.loc[all_2022.recent_team].recent_2
all_2022["recent_3"] = own_team_22.loc[all_2022.recent_team].recent_3
all_2022["recent_4"] = own_team_22.loc[all_2022.recent_team].recent_4
all_2022["recent_5"] = own_team_22.loc[all_2022.recent_team].recent_5
opp_team_22 = team_encoder.transform(all_2022[["opponent_team"]])
own_team_23 = team_encoder.transform(all_2023[["recent_team"]])
opp_team_23 = team_encoder.transform(all_2023[["opponent_team"]])
encoded_22 = own_team_22.join(opp_team_22)
encoded_23 = own_team_23.join(opp_team_23)
all_2022 = all_2022.join(encoded_22)
all_2023 = all_2023.join(encoded_23)
"""
