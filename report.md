<div align="center">

# NBA Statistical Analysis through a Machine Learning Classifier
### by Milad Chabok
</div>

## Introduction

Data analytics drives our world today, and there are few better examples than within the world of sports. Sports inherently center around data, scores and other numeric information is kept to determine the winners and best performing players. From the 1927 Yankees with Babe Ruth to the  Billy Beane's 2002 Oakland Athletics, sports has increasingly become more data driven as time passes, and the trend continues on to this day. It was, however, specifically these Oakland A's who are largerly credited with sparking the modern data revolution. With Billy Beane, their general manager, given a paltry $41 million budget to run a talent-disadvantaged professional baseball team, he turned to data analytics to play the game purely by the numbers, ignoring any and all age-tested human intuition and general belief. The result of playing such a purely objective, numbers-driven game? They go on to have a record-breaking 20 consecutive wins become the American League champions. Two years later, the Boston Red Sox would go on to adopt this strategy and win the World Series. This period in sports history demonstrated to the world that analytics could translate directly to real-world success. This mantra has since made its way to into practically every sport. Perhaps the most data-centric sport after baseball is that of basketball, and more specifically, basketball in the context of the National Basketball Association. The NBA is America's, and the world's, premier professional basketball organization, and after the success found in baseball, teams and decision makers within the league have been combing through the numbers and attempting to establish what defines winning-ways in their own sport ever since.

Closely related to the concept of data analytics is the more fundamental subject of data science and it's adjecent field machine learning. Scores of professionals working for sports organizations are constantly refining their metrics and predictive prowess. Machine learning, more specifically, has shown to have tremendous predictive applications when built upon the right data. The problem is professionals within the field cannot agree on what the right data is. While it's trivial to agree on the basic statistics that reflect positive sports performance, the complex datas and forumulas, known as advanced stats, are more nuanced. In search of the best advanced statistics, statistician Dean Oliver identified what he called the "Four Factors of Basketball Success", and they are a team's effective field goal percentage, turnover percentage offensive rebound percentage, and free throw rate. It is claimed that these are among the box score derived metrics that most closely correlate with winning basketball games. However, it is still up for debate which advance stats are more indicitive of team success. This white paper demonstrates the potential for machine learning algorithms to find emergent properties in NBA data analytics, by applying classic ML techniques to Oliver's Four Factors to see if these algorithms naturally intuit the same conclusion.

## Data Used

Sports are an inherently difficult thing to predict. There are countless variables involved all intermingling and many intangible features. Basketball, especially, is an inventory sport and therefor has many games over a season with large periods of variance. Any one game is especially hard, with variables like team momentum, player absences and additions, etc. but with large enough sample sizes trends can be found
and indicators of success can be uncovered. To attain these large samples, the data used sourced from the nba_api python library as well as significant contributions from Eoin A Moore and Wyatt Walsh’s Kaggle sets. Team-only statistics were derived from these sets, and together they formed a database which covered every game spanning *41 seasons*, from 1983 - 2024.

There were a total of 49,626 games present in the data set, and there were team box-score statistics including points (PTS), offensive rebounds (OREB), defensive rebounds (DREB), total assists (AST), total steals (STL), total blocks (BLK), etc. In order to attempt to analyze and compare the effectiveness of advanced stats, I computed and imputed a stable of them, of course including Oliver's Four Factors, but also true shooting percentage, assist-to-turnover ratio, team pace, possession count, net rating, home game advantage, outcome of previous game, and back-to-back status.

The Four Factors were computed as such:

**Effective Field Goal Percentage**
eFG% = (FGM + 0.5 * 3PM) / FGA

**Turnover Percentage**
TOV% = TO / (FGA + 0.44 * FTA + TO)

**Offensive Rebound Percentage**
ORB% = ORB / (ORB + Opponent DRB)

**Free Throw Rate**
FTr = FTA / FGA

At first glance, historical analysis of one of the more relevant advanced stats showed an interesting trend. True shooting percentage seemed to dip to an all-time low late 1990s, before steadily rising to an all time high in the current game. However, the statistics are only useful when presented as relative to how other teams we performing at the time, and that is where the model is of use.

<p align="center">
<img height="300" alt="TS avg over time save for later" src="https://github.com/user-attachments/assets/8070fa0e-34a7-40b2-9223-7302de87215b" />
</p>

## Gradient Boosted Trees to Extract Feature Importance

Many machine learning algorithms can be used, not only to classify or predict things, but to also derive important characteristics about the data you are working with. When these characteristics, or features, are implicit or hidden in nature they are reffered to as latent, and these exactly the kinds of information that ML may uncover. Such is the approach used here in our investigation of the most deterministic basketball advanced statistics. A classification model is implemented using the data to learn to classify a team as a winner or loser within individual games. From this process of learning how to successfully classify a team as winner based on their statistics, the model inherently uncovers possibly-relevant patterns and signals. The more interpretable the model, meaning able to be understood and explored, the more you can derive which traits are having the most influence in its decision making. If you have a model which is highly interpretable and is highly accurate in its classifications, then you can use it as a vessel for deriving latent features behind what it takes to win a basketball game! And that is exactly role of machine learning here, not to necessarily predict outcomes of games unknown but to instead indicate which features may be the largest predictors of success.

### The Model

To this purpose, a gradient boosted decision tree was implemented using HistGradientBoostingClassifier from the extremely canonical scikit-learn library. This is the library's equivalence to the popular and notoriously effective XGBoost algorithm. However, while decision trees like these algorithms are often very interpretable, unlike XGBoost, HistGradientBoostingClassifier does not have direct functionality for interpreting feature importance and so a method called permutation importance is performed instead. To build our model a temporal 80-20% test train split was done, providing ample data for the model to learn from and then be evaluated on. The model was trained to classify a game winner based on the game's retrospective data, and then, upon learning successful classification abilities, latent features are derived to investigate which are being used as the model's effective predictors of success. In order to have any trust in the latent features a model discovers, however, you must first trust its effectivness. An evaluation of the model showed of an **accuracy of 82%**, with baseline models outputting around 60%, and an **ROC AUC of 0.9054**. A confusion matrix showed strong numbers and good results and symmetrical errors, which is important because asymmetric errors might imply a bias toward predicting “win” or “loss.”

<p align="center">
<img height="400" alt="Confusion matrix on original model run" src="https://github.com/user-attachments/assets/a80cb53c-b0f4-4640-8344-2b6bc6f19a7e" />
</p>

### Results

As stated above, a method called permutation importance is performed on the HistGradientBoostingClassifier in order to deduce feature importance within the model. This is done by selecting single features at a time, randomly scrambling their values to kill their signal, and then the decrease in a model's score is measured and recorded. By performing this on all the features involved (permutating) and comparing mean accuracy, you can observe which feature had the most impact on model performance relative to the others.

<p align="center">
<img height="375" alt="Permutation feature importance" src="https://github.com/user-attachments/assets/883224e6-48c2-4cc6-9fff-029071c86838" />
</p>

After performing permutation importance on the model, true shooting percentage was shown to have the largest influence by far at a delta of 25.3%. Defensive rebounding came second, at 11.2%, followed by turnovers, offensive rebounds, and steals. The model does view the advanced efficiency stat and rebounding as being most influential, however it does leave turn over percentage off its list of importance in lieu of total turnovers. It is most likely that there is significant overlap in these signals and due to the directional, hierarchical flow of decision trees, once the tree has captured most of that overlapping signal it will view the impact of the overlapping features that follow very minimally. This can best be rectified by ablating these overlapping features and noting they're significance when acting in isolation.

<p align="center">
<img height="375" alt="Decision surface DREB vs TS" src="https://github.com/user-attachments/assets/9e685a03-b1bd-4eb9-b5b4-d3f441a61429" />
</p>

The decision boundary of the model can be observed in this plotting of rebounds against true shooting percentage. A sort of linear boundry with a steep negative slope can been seen, past which the model observes a win. This implementation is certainly crude relative to what can be done with iterating over with the same tools. Overall though, this model operates as a proof of concept of the notion of using machine learning as a method of uncovering latent features within data. Other methods, on top of feature importance from decision trees, include matrix factorization with collaborative filtering and principle component analysis, although these techniques have varying levels of interpretability. While a powerful tool in prediciting outcomes, machine learning is only as good as the quality of its features in which it is trained upon, and in a sort of bootstrap paradoxical way, machine learning can also be used to identify which features those may be.
