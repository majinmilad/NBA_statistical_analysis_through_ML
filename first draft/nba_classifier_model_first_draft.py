
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    auc,
    brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # print all dataframe columns, regardless of how many there are
pd.set_option('display.width', 1000)  # prevent horizontal wrapping in print by setting a very large display width
pd.set_option('display.max_rows', 100)  # set the maximum number of rows to display before truncation (...) starts
pd.options.mode.chained_assignment = None  # done to prevent warnings later on reformatted columns

from utility_functions import retrieve_encoded_team_id_dictionary

# -------------------------------------------------------------------------------------------------------------
# READ IN DATA

nba_games = pd.read_parquet('nba_all_team_games_enriched.parquet', engine='pyarrow')

# -------------------------------------------------------------------------------------------------------------
# ENCODE TEAM AND GAME ID LABELS

"""
Use a LabelEncoder to transform a few of our categorical data columns
into columns with numerical values that can be better understood by our model.
"""

le = LabelEncoder()
nba_games["TEAM_ID"] = le.fit_transform(nba_games["TEAM_ID"])
nba_games["OPP_TEAM_ID"] = le.fit_transform(nba_games["OPP_TEAM_ID"])
nba_games["GAME_ID"] = le.fit_transform(nba_games["GAME_ID"])

# dictionary keying each team name to their new encoded team ID (may be useful later)
ENCODED_TEAM_IDS = retrieve_encoded_team_id_dictionary()

# -------------------------------------------------------------------------------------------------------------
# CHOOSING FEATURES

"""
X will include features of the dataset that will be used to predict y, where y is the binary 'WIN' column.
"""

features = [
    "TEAM_ID",
    "OPP_TEAM_ID",
    "PTS",
    "OREB",
    "DREB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "EFG%",
    "TOV%",
    "FTR",
    "TS%",
    "HGA",
    "LAST_GAME_OUTCOME",
    "BACK_TO_BACK",
]

X = nba_games[features]
y = nba_games["WIN"]

# -------------------------------------------------------------------------------------------------------------
# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # keep class balance similar in train and test
)

# -------------------------------------------------------------------------------------------------------------
# MODEL

"""
Using HistGradientBoostingClassifier from scikit_learn we implement a gradient boosting algorithm for our
classification model. It handles datasets with null values well, and while it may be slightly inferior to
XGBoost it still offers many of the same strengths.
"""

grad_boost_model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    random_state=42,
)

grad_boost_model.fit(X_train, y_train)

# -------------------------------------------------------------------------------------------------------------
# EVALUATION

# hard predictions
y_pred = grad_boost_model.predict(X_test)

# probabilities (needed for ROC)
y_proba = grad_boost_model.predict_proba(X_test)[:, 1]

# Accuracy and classification report
print("\n BASIC METRICS ")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("\n CONFUSION MATRIX ")

cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC curve & AUC
print("\n ROC / AUC ")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="HistGB").plot()
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # random baseline
plt.tight_layout()
plt.show()