# Fantasy-Football-Machine-Learning

Fantasy football is a game based on real-time NFL games. Players manage a fantasy football team, and set a lineup of NFL athletes each week to compete against other managers. NFL players score fantasy points corresponding to their real-life performances; for example, in leagues with a PPR (points-per-reception) scoring system, a wide receiver typically scores 1 point for every catch they make, 1 point for every 10 receiving yards, and 6 points for a touchdown. 

This project aimed to train machine learning models that can predict a player's fantasy performance based on their performance in prior weeks' games. Ultimately, the goal was to develop a model that performs on par with or better than fantasy football apps, which conduct their own analyses to predict player performance.

Data
----
All data was obtained via the nfl_data_py API (https://pypi.org/project/nfl-data-py/). This API combines statistics from a variety of other sources and makes it easy for data to be imported into Python. Statistics for all players who played in the 2022-2023 and 2023-2024 NFL seasons were imported as pandas DataFrames; types of statistics included box score statistics, such as catches, rushing yards, and passing attempts, as well as advanced statistics, such as rushing yards after contact, bad throw percentage, and receiving target share. Most of these statistics served as features for the machine learning models; the player's fantasy point total, fantasy_points_ppr, was the target variable

The data had to be cleaned and joined; the main task was joining the API's default data with data it sourced from Pro Football Reference, as the two had to be imported with separate methods. In the merging process, redundant columns were dropped, certain columns were renamed, and names were standardized across dataframes. After these processes, all data was condensed into two dataframes, one for players in the 2022-2023 season and one for players in the 2023-2024 season.

Feature Engineering
-------------------
For all numerical statistics, a rolling average was applied so that the features would be comprised of statistics from games prior to the current week, not statistics from the current week (inherently tied to the current week's fantasy point total). The rolling average was applied such that each statistic would be split into three different features: "previous", the statistic from the previous week's game, "recent", the statistic's average over the last 4 games, and "season", the statistic's average over the season up until before the current week. 

Additionally, some qualititative aspects of the game, namely which teams were playing and which teams were home/away, underwent categorical encoding. Each row of the dataframe, corresponding to a particular player during a particular week, had columns corresponding to the player's team and the team he was facing that week. Both these columns were processed with binary encoding such that all 32 NFL teams had 2 unique codes: one representing their offense (when the player in a row is on a team), and one representing their defense (when the player in a row is facing a team). It was hypothesized that this encoding would help the model consider both a player's supporting cast and the opposing team's defense in predicting the player's performance. Each row also had had a home/away column with 0 if the player was playing at another team's stadium or 1 if the player was playing at their own team's stadium. 

Finally, some features were created manually via ratios of other features. A mutual information regression was performed on all features to see which features were most related to the target. Using the regression's highest scoring features and football knowledge, a handful of ratios were created, including running back and quarterback weighted opportunity (a measure of the opportunity players were given to score fantasy points). Principal component analysis was considered in the hopes of capturing the variation of the existing features, but this was ultimately not pursued as the results did not seem to generate any easily interpretable insights.

Machine Learning Models
-----------------------



