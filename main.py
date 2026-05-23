import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET
df = pd.read_csv("IPL_Match_Level.csv")
# REMOVE MISSING VALUES
df = df.dropna()
# CREATE TEAM FORM FEATURES
team_recent_wins = {}

df['batting_team_form'] = 0
df['bowling_team_form'] = 0

for index, row in df.iterrows():

    batting_team = row['batting_team']
    bowling_team = row['bowling_team']
    winner = row['match_won_by']

    # Initialize teams
    if batting_team not in team_recent_wins:
        team_recent_wins[batting_team] = []

    if bowling_team not in team_recent_wins:
        team_recent_wins[bowling_team] = []

    # Calculate recent form
    batting_form = sum(
        team_recent_wins[batting_team][-5:]
    )

    bowling_form = sum(
        team_recent_wins[bowling_team][-5:]
    )

    # Store form
    df.at[index, 'batting_team_form'] = batting_form

    df.at[index, 'bowling_team_form'] = bowling_form

    # Update win history
    if winner == batting_team:

        team_recent_wins[batting_team].append(1)

        team_recent_wins[bowling_team].append(0)

    elif winner == bowling_team:

        team_recent_wins[batting_team].append(0)

        team_recent_wins[bowling_team].append(1)

# FEATURES
X = df[[
    'batting_team',
    'bowling_team',
    'toss_winner',
    'toss_decision',
    'venue',
    'batting_team_form',
    'bowling_team_form'
]]
# ENCODE TARGET LABELS
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df['match_won_by']
)
# CATEGORICAL FEATURES
categorical_features = [
    'batting_team',
    'bowling_team',
    'toss_winner',
    'toss_decision',
    'venue'
]
# PREPROCESSOR
preprocessor = ColumnTransformer(
    transformers=[

        (
            'cat',

            OneHotEncoder(
                handle_unknown='ignore'
            ),

            categorical_features
        )

    ],

    remainder='passthrough'
)
# MODELS

# ---------------- RANDOM FOREST ----------------

rf_model = RandomForestClassifier(

    n_estimators=500,

    max_depth=10,

    random_state=42
)

# ---------------- CATBOOST ----------------

cat_model = CatBoostClassifier(

    iterations=500,

    depth=8,

    learning_rate=0.05,

    verbose=0
)

# ---------------- LOGISTIC REGRESSION ----------------

lr_model = LogisticRegression(

    max_iter=3000
)

# ENSEMBLE MODEL

ensemble_model = VotingClassifier(

    estimators=[

        ('rf', rf_model),

        ('cat', cat_model),

        ('lr', lr_model)

    ],

    voting='soft'
)

# PIPELINES

rf_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', rf_model)

])

cat_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', cat_model)

])

lr_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', lr_model)

])

ensemble_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', ensemble_model)

])

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)
# TRAIN RANDOM FOREST

rf_pipeline.fit(X_train, y_train)

rf_predictions = rf_pipeline.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

# TRAIN CATBOOST
cat_pipeline.fit(X_train, y_train)

cat_predictions = cat_pipeline.predict(X_test)

cat_accuracy = accuracy_score(
    y_test,
    cat_predictions
)

# TRAIN LOGISTIC REGRESSION
lr_pipeline.fit(X_train, y_train)

lr_predictions = lr_pipeline.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

# TRAIN ENSEMBLE MODEL

ensemble_pipeline.fit(X_train, y_train)

ensemble_predictions = ensemble_pipeline.predict(X_test)

ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_predictions
)

# PRINT ACCURACIES
print("MODEL ACCURACIES")

print(f"\nRandom Forest Accuracy: {rf_accuracy:.2f}")

print(f"CatBoost Accuracy: {cat_accuracy:.2f}")

print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")

print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}")

# BEST MODEL SELECTION

best_model = ensemble_pipeline

best_predictions = ensemble_predictions

# CLASSIFICATION REPORT

print("CLASSIFICATION REPORT")

print(classification_report(
    y_test,
    best_predictions
))


# CONFUSION MATRIX WITH TEAM NAMES


cm = confusion_matrix(
    y_test,
    best_predictions
)

# Get actual team names
team_names = label_encoder.inverse_transform(
    np.unique(y)
)

plt.figure(figsize=(14, 12))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=team_names,

    yticklabels=team_names

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Team")

plt.ylabel("Actual Team")

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()
# FEATURE IMPORTANCE
rf_model.fit(
    preprocessor.fit_transform(X_train),
    y_train
)

encoded_features = preprocessor.get_feature_names_out()

importance = rf_model.feature_importances_

indices = np.argsort(importance)[::-1]

plt.figure(figsize=(14, 8))

plt.title("Feature Importance")

plt.bar(
    range(len(importance)),
    importance[indices]
)

plt.xticks(
    range(len(importance)),
    encoded_features[indices],
    rotation=90
)

plt.tight_layout()

plt.show()

# ACCURACY COMPARISON GRAPH


models = [
    'Random Forest',
    'CatBoost',
    'Logistic Regression',
    'Ensemble'
]

accuracies = [
    rf_accuracy,
    cat_accuracy,
    lr_accuracy,
    ensemble_accuracy
]

plt.figure(figsize=(10, 5))

plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy")

plt.title("Model Accuracy Comparison")

plt.ylim(0, 1)

plt.show()

# USER INPUT

print("\n===== IPL MATCH PREDICTOR =====")

batting_team = input(
    "Enter Batting Team: "
)

bowling_team = input(
    "Enter Bowling Team: "
)

toss_winner = input(
    "Enter Toss Winner: "
)

toss_decision = input(
    "Enter Toss Decision (bat/field): "
)

venue = input(
    "Enter Venue: "
)

# TEAM FORM INPUT


batting_team_form = int(
    input(
        "Enter Batting Team Last 5 Match Wins (0-5): "
    )
)

bowling_team_form = int(
    input(
        "Enter Bowling Team Last 5 Match Wins (0-5): "
    )
)


# CREATE SAMPLE INPUT

sample = pd.DataFrame([{

    'batting_team': batting_team,

    'bowling_team': bowling_team,

    'toss_winner': toss_winner,

    'toss_decision': toss_decision,

    'venue': venue,

    'batting_team_form': batting_team_form,

    'bowling_team_form': bowling_team_form

}])


# PREDICT WINNER


prediction = best_model.predict(sample)

winner = label_encoder.inverse_transform(
    prediction.astype(int)
)

print("\n🏆 Predicted Winner:", winner[0])


# PROBABILITY FOR ONLY PLAYING TEAMS


probabilities = best_model.predict_proba(sample)[0]

classes = label_encoder.inverse_transform(
    np.arange(len(probabilities))
)

# Create dictionary
team_probabilities = dict(zip(classes, probabilities))

# Get only playing teams
team1_prob = team_probabilities.get(
    batting_team,
    0
)

team2_prob = team_probabilities.get(
    bowling_team,
    0
)

teams_to_plot = [
    batting_team,
    bowling_team
]

probs_to_plot = [
    team1_prob,
    team2_prob
]


# NORMALIZE TO 100%


total = team1_prob + team2_prob

team1_prob = team1_prob / total
team2_prob = team2_prob / total

probs_to_plot = [
    team1_prob,
    team2_prob
]

# PLOT GRAPH
plt.figure(figsize=(8, 5))

plt.bar(
    teams_to_plot,
    probs_to_plot
)

plt.ylabel("Winning Probability")

plt.title("Match Winning Prediction")

# Add percentage labels
for i, prob in enumerate(probs_to_plot):

    plt.text(
        i,
        prob + 0.01,
        f"{prob*100:.1f}%",
        ha='center',
        fontsize=12
    )

plt.ylim(0, 1)

plt.show()