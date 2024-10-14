import nfl_data_py as nfl
import numpy as np
import pandas as pd
import re
from nameparser import HumanName

# import weekly stats for 2022, 2023 seasons, remove cases of fantasy-irrelevant positions
# includes standard box score data as well as some analytics-specific data
regular_22 = nfl.import_weekly_data([2022])
regular_23 = nfl.import_weekly_data([2023])
relevant_positions = ["QB", "WR", "TE", "RB", "FB"]
trimmed_22 = regular_22[regular_22["position"].isin(relevant_positions)]
trimmed_23 = regular_23[regular_23["position"].isin(relevant_positions)]
trimmed_22 = trimmed_22.reset_index(drop=True) # reindex subset to avoid skipped indices
trimmed_23 = trimmed_23.reset_index(drop=True)


# drop redundant columns
# Note: we are using columns player_display_name and fantasy_points_ppr
drop_columns = ["headshot_url", "player_id", "player_name", "fantasy_points"]
trimmed_22 = trimmed_22.drop(drop_columns, axis=1) 
trimmed_23 = trimmed_23.drop(drop_columns, axis=1)

# renames columns of Pro Football Reference stats dataframes to match columns of regular weekly stats
# drops uninformational id columns, any empty columns
pfr_drop_columns = ["game_id", "pfr_game_id", "pfr_player_id"]
def refine_pfr_columns(data, to_drop=pfr_drop_columns):
    data = data.rename(columns = {"game_type": "season_type", 
                                  "team": "recent_team", 
                                  "pfr_player_name": "player_display_name",
                                  "opponent": "opponent_team",
                                  })
    data = data.drop(to_drop, axis=1)
    data.dropna(how="all", axis=1, inplace=True)
    return data

# import weekly Pro Football Reference (PFR) passing, receiving,
# rushing stats for 2022, 2023 seasons from API, fix columns
pass_22 = refine_pfr_columns(nfl.import_weekly_pfr(s_type="pass", years=[2022]))
rec_22 = refine_pfr_columns(nfl.import_weekly_pfr(s_type="rec", years=[2022]))
rush_22 = refine_pfr_columns(nfl.import_weekly_pfr(s_type="rush", years=[2022]))
rush_22 = rush_22.drop(["carries"], axis=1)
pass_23 = refine_pfr_columns(nfl.import_weekly_pfr(s_type="pass", years=[2023]))
rec_23 = refine_pfr_columns(nfl.import_weekly_pfr(s_type="rec", years=[2023]))
rush_23 = refine_pfr_columns(nfl.import_weekly_pfr(s_type="rush", years=[2023]))
rush_23 = rush_23.drop(["carries"], axis=1)

# returns common columns between two DataFrames to be used when merging
def get_common_columns(df1, df2):
    common_columns = [col for col in df2.columns if col in df1.columns]
    return common_columns

# merges two pfr datasets, ensures that stats of "duplicate" players 
# appearing in both datasets for a particular week are kept after merging
# ex. we want the combined pfr dataset to show a QB's rushing/receiving stats
# if they ran the ball or caught a pass in a particular week, not just their passing stats
def merge_pfr(df1, df2):
    duplicate_names = [name for name in pd.unique(df2["player_display_name"])
                            if name in pd.unique(df1["player_display_name"])] 
    duplicates_df1 = df1[df1["player_display_name"].isin(duplicate_names)]
    duplicates_df2 = df2[df2["player_display_name"].isin(duplicate_names)] 
    columns = get_common_columns(duplicates_df1, duplicates_df2)
    duplicates_full = duplicates_df1.merge(duplicates_df2, how="inner", on=columns)
    columns2 = get_common_columns(duplicates_full, df1)
    df1 = df1.merge(duplicates_full, how="left", on=columns2) # update duplicate player rows in df1 w/ stats from df2
    indices = []
    for i in range(duplicates_full.shape[0]):
        index = df2[(df2.week==duplicates_full.iloc[i].week) &
                    (df2.player_display_name==duplicates_full.iloc[i].player_display_name)]\
                    .index.astype(int)[0]
        indices.append(index)
    df2 = df2.drop(indices) # drops rows corresponding to duplicate players from df2
    joined = pd.concat([df1, df2], ignore_index=True) # join rest of df2 (players not appearing in df1) to df1
    return joined

# normalize names so pfr and regular stat DataFrames have same naming conventions
name_discrepancies = {"Chosen Anderson": "Robbie Chosen", "Jeff Wilson": 
                      "Jeffery Wilson", "Jody Fortson": "Joe Fortson",
                      "Gabriel Davis": "Gabe Davis", "Michael Woods": 
                      "Mike Woods", "KaVontae Turpin": "Kavontae Turpin",
                      "Rodney Williams": "Rod Williams", "Christopher Brooks": "Chris Brooks",
                      "De'Von Achane": "Devon Achane"}
def normalize_names(df1):
    for i in range(df1.shape[0]):
        name = HumanName(df1.iloc[i]["player_display_name"])
        if(re.search(".\\..\\.", name.first) != None):
            name.first = "".join(re.split("\\.", name.first)) # changes first names like D.K. to DK
        if name.suffix != "":
            name.suffix = ""
        if name.full_name in list(name_discrepancies.keys()):
            name = HumanName(name_discrepancies[name.full_name]) # uses above dictionary to correct other discrepancies
        df1.at[i, "player_display_name"] = name.full_name
    return df1

# combine pfr passing, receiving, rushing stats into one DataFrame
pfr_all_22 = merge_pfr(pass_22, rec_22)
pfr_all_22 = merge_pfr(pfr_all_22, rush_22)
pfr_all_23 = merge_pfr(pass_23, rec_23)
pfr_all_23 = merge_pfr(pfr_all_23, rush_23)

# standardize naming conventions
pfr_all_22 = normalize_names(pfr_all_22)
trimmed_22 = normalize_names(trimmed_22)
pfr_all_23 = normalize_names(pfr_all_23)
trimmed_23 = normalize_names(trimmed_23)

# obtain complete stats for 2022 and 2023
common_columns = get_common_columns(trimmed_22, pfr_all_22)
all_22 = trimmed_22.merge(pfr_all_22, how="left", on=common_columns)
all_22.to_csv("all_22.csv", header=True, index=True)
all_23 = trimmed_23.merge(pfr_all_23, how="left", on=common_columns)
all_23.to_csv("all_23.csv", header=True, index=True)




