
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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None

from utility_functions import retrieve_encoded_team_id_dictionary


def plot_decision_boundary_2d(model, X_ref, X_points, y_points, feature_x, feature_y, grid_size=100):
    features = X_ref.columns
    if feature_x not in features or feature_y not in features:
        raise ValueError("Features for decision boundary must be in X_ref columns")

    ix = features.get_loc(feature_x)
    iy = features.get_loc(feature_y)

    x_min, x_max = X_ref[feature_x].quantile(0.02), X_ref[feature_x].quantile(0.98)
    y_min, y_max = X_ref[feature_y].quantile(0.02), X_ref[feature_y].quantile(0.98)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )

    base = X_ref.median().values
    grid = np.tile(base, (xx.size, 1))

    grid[:, ix] = xx.ravel()
    grid[:, iy] = yy.ravel()

    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.figure()
    contour = plt.contourf(xx, yy, probs, levels=np.linspace(0.0, 1.0, 11), alpha=0.8)
    plt.colorbar(contour, label="Predicted win probability")

    plt.scatter(
        X_points[feature_x],
        X_points[feature_y],
        c=y_points,
        edgecolor="k",
        alpha=0.3,
        s=10,
    )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"Decision surface: {feature_x} vs {feature_y}")
    plt.tight_layout()
    plt.show()
    

nba_games = pd.read_parquet('nba_all_team_games_enriched.parquet', engine='pyarrow')

le = LabelEncoder()
nba_games["TEAM_ID"] = le.fit_transform(nba_games["TEAM_ID"])
nba_games["OPP_TEAM_ID"] = le.fit_transform(nba_games["OPP_TEAM_ID"])
nba_games["GAME_ID"] = le.fit_transform(nba_games["GAME_ID"])

ENCODED_TEAM_IDS = retrieve_encoded_team_id_dictionary()

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

grad_boost_model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    random_state=42,
)

grad_boost_model.fit(X_train, y_train)

y_pred = grad_boost_model.predict(X_test)
y_proba = grad_boost_model.predict_proba(X_test)[:, 1]

print("\n BASIC METRICS ")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification report:\n", classification_report(y_test, y_pred))

print("\n CONFUSION MATRIX ")

cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - HistGradientBoostingClassifier")
plt.tight_layout()
plt.show()

print("\n ROC / AUC ")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="HistGB").plot()
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.tight_layout()
plt.show()

print("\n CALIBRATION ")

brier = brier_score_loss(y_test, y_proba)
print(f"Brier score (lower is better): {brier:.4f}")

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")

plt.plot(prob_pred, prob_true, marker="o", linewidth=1)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("Predicted win probability")
plt.ylabel("Observed win frequency")
plt.title("Calibration curve - HistGradientBoostingClassifier")
plt.tight_layout()
plt.show()

print("\n PERMUTATION FEATURE IMPORTANCE ")

result = permutation_importance(
    grad_boost_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

importances_mean = result.importances_mean
importances_std = result.importances_std

indices = np.argsort(importances_mean)[::-1]

print("Feature importances (permutation-based, higher = more important):\n")
for idx in indices:
    print(f"{features[idx]:<20} {importances_mean[idx]:.4f} \u00b1 {importances_std[idx]:.4f}")

plt.figure()
plt.barh(range(len(indices)), importances_mean[indices], xerr=importances_std[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel("Mean accuracy decrease when shuffled")
plt.title("Permutation feature importance - HistGradientBoostingClassifier")
plt.tight_layout()
plt.show()

print("\n DECISION BOUNDARY MAPS ")

plot_decision_boundary_2d(
    grad_boost_model,
    X_train,
    X_test,
    y_test,
    feature_x="TS%",
    feature_y="DREB",
)

plot_decision_boundary_2d(
    grad_boost_model,
    X_train,
    X_test,
    y_test,
    feature_x="TS%",
    feature_y="TOV",
)