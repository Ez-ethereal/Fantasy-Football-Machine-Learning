import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from nflprocessing import averaged_feat_22, rolling_average, get_rolling_averages

# read files
all_22 = pd.read_csv("full_2022_processed.csv", index_col=0)
all_23 = pd.read_csv("full_2023_processed.csv", index_col=0)

# returns mutual information scores for features 
#   - score indicating the extent to which information about a feature can inform us about variation in the target variable
def make_mi_scores(X, y):
    X = X.copy()
    mi_scores = mutual_info_regression(X, y, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# creates a csv listing mutual information scores for features for a given position
def make_pos_mi(df, pos):
    X = df.copy()
    y = X.pop("fantasy_points_ppr")
    mi = make_mi_scores(X, y)
    mi.to_csv("{}_mi_23.csv".format(pos), header=True, index=True)

# calculates the weight of a stat towards scoring fantasy points among a particular position
def calculate_weight(df, position, stat1, stat2="fantasy_points_ppr"):
    aggregated = df.loc[df.position==position][stat1].sum()
    points_aggregated = df.loc[df.position==position][stat2].sum()
    weight = points_aggregated / aggregated
    return weight

# uses weights for running back carries and targets to create a volume/workload metric
# quantifies the opportunity a running back is given to score fantasy points
# mimics the weighted opportunity rating stat that nfl_data provides for receivers
def calculate_rb_opportunity(df):
    carry_weight = calculate_weight(df, "RB", "carries")
    target_weight = calculate_weight(df, "RB", "targets")
    df.loc[df.position=="RB", "rb_wopr"] = (carry_weight*df.loc[df.position=="RB"].carries)+(target_weight*df.loc[df.position=="RB"].targets)
    df = get_rolling_averages(df, ["rb_wopr"])[0]
    return df

# quantifies the opportunity a quarterback is given to score fantasy points
def calculate_qb_opportunity(df):
    pass_weight = calculate_weight(df, "QB", "attempts")
    carry_weight = calculate_weight(df, "QB", "carries")
    df.loc[df.position=="QB", "qb_wopr"] = (carry_weight*df.loc[df.position=="QB"].carries)+(pass_weight*df.loc[df.position=="QB"].attempts)
    df = get_rolling_averages(df, ["qb_wopr"])[0]
    return df

# based on mutual information scores, combine stats that seem to correlate with fantasy success using ratios
# applying domain knowledge - a QB that doesn't get hit much should score more, as should a QB with a high completion percentage
def qb_ratios(df):
    ratio_features = []
    QB = df.loc[df.position=="QB"]
    for time in ["season", "recent", "prev"]:
        df.loc[df.position=="QB", "{}_hitrate".format(time)] = (QB[time+"_times_sacked"] + QB[time+"_times_hit"]) / (QB[time+"_times_blitzed"] + QB[time+"_times_pressured"])
        df.loc[df.position=="QB", "{}_comp_pct".format(time)] = (QB[time+"_completions"]) / (QB[time+"_attempts"])
        ratio_features += ["{}_hitrate".format(time), "{}_comp_pct".format(time)]
    return df, ratio_features

# ratios for important receiving stats
# receivers excelling in converting first downs and running after the catch should be important in an offense and score more
def wr_te_ratios(df):
    ratio_features = []
    WR = df.loc[df.position=="WR"]
    TE = df.loc[df.position=="TE"]
    for time in ["season", "recent", "prev"]:
        df.loc[df.position=="WR", "{}_fd_rate".format(time)] = WR[time+"_receiving_first_downs"] / WR[time+"_targets"]
        df.loc[df.position=="TE", "{}_fd_rate".format(time)] = TE[time+"_receiving_first_downs"] / TE[time+"_targets"]
        df.loc[df.position=="WR", "{}_rac_pct".format(time)] = WR[time+"_receiving_yards_after_catch"] / WR[time+"_receiving_yards"]
        df.loc[df.position=="TE", "{}_rac_pct".format(time)] = WR[time+"_receiving_yards_after_catch"] / WR[time+"_receiving_yards"]
        ratio_features += ["{}_fd_rate".format(time), "{}_rac_pct".format(time)]
    return df, ratio_features

# ratios for important rushing stats
# among RBs with similar volume, those with high efficiency (yards per carry) should score more 
def rb_ratios(df):
    ratio_features = []
    RB = df.loc[df.position=="RB"]
    for time in ["season", "recent", "prev"]:
        df.loc[df.position=="RB", "{}_ypc".format(time)] = RB[time+"_rushing_yards"] / RB[time+"_carries"]
        ratio_features += ["{}_ypc".format(time)]
    return df, ratio_features

# adding ratios and weighted opportunity ratings to feature set
def add_transformed_features(df):
    df = calculate_rb_opportunity(df)
    df = calculate_qb_opportunity(df)
    df, pass_features = qb_ratios(df)
    df, rec_features = wr_te_ratios(df)
    df, back_features = rb_ratios(df)
    pass_features += ["season_qb_wopr", "recent_qb_wopr", "prev_qb_wopr"]
    back_features += ["season_rb_wopr", "recent_rb_wopr", "prev_rb_wopr"]
    return df, pass_features, rec_features, back_features

# modify all_22 and all_23 to include engineered features
all_22, qb_transformed_features, rec_transformed_features, rb_transformed_features = add_transformed_features(all_22)
all_23 = add_transformed_features(all_23)[0]

# get lists of position-specific feature names
numerical_feat = averaged_feat_22
qb_features = numerical_feat + qb_transformed_features
wr_te_features = numerical_feat + rec_transformed_features
rb_features = numerical_feat + rb_transformed_features

all_22.to_csv("final_22.csv", header=True, index=True)
all_23.to_csv("final_23.csv", header=True, index=True)



# experiments w/ using principal component analysis to generate features based on variance of data
# not included in final feature set
"""
pca_features = [col for col in mi_scores.nlargest(20).index]
X_qb_pca = X_qb.loc[:, pca_features]
def apply_pca(X, standardize=True):
    # option to standardize data
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    # naming component columns with PC and numbers, X_pca.shape[1] returns number of columns
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings, which originally has components as rows and original features as columns
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings

def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    plt.show()
    return axs

pca, X_pca, loadings = apply_pca(X_qb_pca)
plot_variance(pca)
loadings.to_csv("qb_loadings_v2.csv", header=True)
"""



