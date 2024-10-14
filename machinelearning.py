import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error
from nfl_feature_engineering import qb_features, wr_te_features, rb_features
import scipy.stats as stats

final_22 = pd.read_csv("final_22.csv", index_col = 0)
final_23 = pd.read_csv("final_23.csv", index_col = 0)

# split into training and testing data
# models will be trained on the 2022 season and first 6 weeks of the 2023 season
# trained models' predictions for the last 12 weeks of the 2023 season will be compared to actual fantasy points
train = pd.concat([final_22, final_23.loc[final_23["week"] <= 6]])
test = final_23.loc[final_23["week"] > 6]
train[train==np.inf] = np.nan
test[test==np.inf] = np.nan
train = train.fillna(0)
test = test.fillna(0)

positions = ["QB", "WR", "TE", "RB"]
positional_features = {"QB": qb_features, "WR": wr_te_features, "TE": wr_te_features, "RB": rb_features}
models = ["Ridge", "Lasso", "ElasticNet", "XGB", "RandomForest"]
types = ["train", "cv", "test"]

# initialize a dataframe of 0s
# will be updated w/ Mean Absolute Error scores for all models across all positions
mae_names = [x + "_" + y for y in types for x in models]
mae_results = pd.DataFrame([[0.0] * len(positions) for j in range(len(mae_names))], 
    index = mae_names, columns = positions)


# fitting models, using cross validation to find optimal parameters
for position in positions:
    df_pos_train = train.loc[train["position"] == position]
    df_pos_test = test.loc[test["position"] == position]

    print ('Learning for Position %s ...' % position)

    for i in range(len(models)):
        current_model = models[i]

        print("Fitting " + current_model + " ...")
        

        if (current_model == "Ridge"):
            model = RidgeCV().fit(df_pos_train[positional_features[position]], df_pos_train["fantasy_points_ppr"])
        
        elif (current_model == "Lasso"):
            model = LassoCV().fit(df_pos_train[positional_features[position]], df_pos_train["fantasy_points_ppr"])
        
        elif (current_model == "ElasticNet"):
            model = ElasticNetCV().fit(df_pos_train[positional_features[position]], df_pos_train["fantasy_points_ppr"])
        
        
        elif (current_model == "XGB"):
            param_dist = {"n_estimators": stats.randint(50, 300),
                          "max_depth": stats.randint(3, 7),
                          "learning_rate": stats.uniform(0.01, 0.1)}
            model = RandomizedSearchCV(XGBRegressor(), param_dist, n_iter=10, n_jobs=-1, cv=5, random_state=1, scoring="neg_mean_absolute_error")
            model.fit(df_pos_train[positional_features[position]], df_pos_train["fantasy_points_ppr"])
        
        elif (current_model == "RandomForest"):
            param_dist = {"n_estimators": stats.randint(50, 300),
                          "max_depth": stats.randint(3, 7)}
            model = RandomizedSearchCV(RandomForestRegressor(), param_dist, n_iter=10, n_jobs=-1, cv=5, random_state=1, scoring="neg_mean_absolute_error")
            model.fit(df_pos_train[positional_features[position]], df_pos_train["fantasy_points_ppr"])
        
        print(current_model + " Fitted!")

        """
        train_rmse = np.sqrt(np.mean((df_pos_train['FD points'] - \
                     model.predict(df_pos_train[positional_features[position]]))**2.0))
        test_rmse = np.sqrt(np.mean((df_pos_test['FD points'] - \
                    model.predict(df_pos_test[positional_features[position]]))**2.0))
        cv_rmse = np.sqrt(np.abs(cross_val_score(model, df_pos_train[positional_features[position]], df_pos_train["fantasy_points_ppr"],\
            cv = 5, scoring = 'neg_mean_squared_error').mean()))
        """

        # update dataframe with model's score on training set, testing set, and k-fold cross validation
        train_mae = mean_absolute_error(df_pos_train["fantasy_points_ppr"], model.predict(df_pos_train[positional_features[position]]))
        test_mae = mean_absolute_error(df_pos_train["fantasy_points_ppr"], model.predict(df_pos_train[positional_features[position]]))
        cv_mae = np.abs(cross_val_score(model, df_pos_train[positional_features[position]], df_pos_train["fantasy_points_ppr"],\
            cv = 5, scoring = "neg_mean_absolute_error").mean())

        for type in types:
            mae_results.loc[current_model + "_" + type, position] = eval(type + "_mae")

mae_results.to_csv("mae_results.csv", header=True, index=True)
