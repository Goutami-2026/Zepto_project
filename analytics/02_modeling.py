# %% [markdown]
# # 02 - Titanic Modeling
#
# This is the second half of the analytics module. It picks up where
# 01_eda.py left off, by reading the SAME `titanic.csv` that file saved -
# we never download the dataset a second time.
#
# What happens here:
# 1. Split into train/test (stratified).
# 2. Build a preprocessing pipeline (impute + encode + scale), fit only on
#    the training data.
# 3. Train 3 classifiers and compare them.
# 4. Compare 3 ways of handling class imbalance.
# 5. Tune the Random Forest with GridSearchCV.
# 6. A small regression side-task: predict fare instead of survival.
# 7. Save the whole fitted pipeline with joblib.

# %%
from pathlib import Path
THIS_FOLDER = Path(__file__).parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE
import joblib

# %% [markdown]
# ## Step 1: Load the data that 01_eda.py already cleaned and saved

# %%
titanic = pd.read_csv(THIS_FOLDER / "titanic.csv")
print("Loaded titanic.csv, shape:", titanic.shape)

# for modeling we only need a handful of columns - the rest (deck, alive,
# embark_town, who, class, adult_male, alone) are either duplicates of
# other columns or too empty to use reliably
feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target_column = "survived"

model_data = titanic[feature_columns + [target_column]].copy()

# %% [markdown]
# ## Step 2: Train/test split (stratified)
#
# We stratify on `survived` so that the train set and the test set both
# have roughly the same percentage of survivors as the full dataset. This
# matters because survival is not a 50/50 split (more people died than
# survived), so a random split without stratifying could accidentally put
# too many/few survivors into one side.

# %%
survival_rate_overall = model_data[target_column].mean()
print(f"Overall survival rate: {survival_rate_overall:.2%}")
print("This is not close to 50/50, so we stratify the split to keep that ratio in both sets.")

X = model_data[feature_columns]
y = model_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train survival rate:", y_train.mean().round(4))
print("Test survival rate:", y_test.mean().round(4))

# %% [markdown]
# ## Step 3: Build the preprocessing pipeline
#
# Everything below (imputing, encoding, scaling) is `fit` only on
# `X_train`. When we later call `.transform()` on `X_test`, it just APPLIES
# what it already learned from the training data - it never looks at the
# test data to decide how to fill/scale/encode. This is what stops
# "leakage" from the test set into training.

# %%
numeric_features = ["age", "fare", "sibsp", "parch"]
categorical_features = ["sex", "embarked", "pclass"]

numeric_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

print("Preprocessor is ready. It will be fit only on X_train.")

# %% [markdown]
# ## Step 4: Train 3 classifiers on the same split
#
# Each one is wrapped in a Pipeline with the SAME preprocessor above, so
# the fit-on-train/transform-on-test rule is enforced automatically.

# %%
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

fitted_pipelines = {}

for model_name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    pipe.fit(X_train, y_train)
    fitted_pipelines[model_name] = pipe
    print(f"Trained: {model_name}")

# %% [markdown]
# ### Visualize the Decision Tree

# %%
dt_pipeline = fitted_pipelines["Decision Tree"]
dt_model = dt_pipeline.named_steps["classifier"]

# get the feature names after one-hot encoding, so the tree plot is readable
encoded_cat_names = dt_pipeline.named_steps["preprocessor"].named_transformers_["cat"] \
    .named_steps["encode"].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(encoded_cat_names)

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=all_feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    max_depth=3,  # only show the first few levels so it's readable
    fontsize=8,
)
plt.title("Decision Tree (first 3 levels)")
plt.savefig(THIS_FOLDER / "chart_decision_tree.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 5: Evaluate all three models
#
# For each model we compute: confusion matrix, accuracy, precision,
# recall, F1, and ROC/AUC - then put them all in one comparison table.

# %%
def evaluate_model(pipeline, X_test, y_test, model_name):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {model_name} ---")
    print("Confusion matrix:")
    print(cm)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "y_proba": y_proba,
    }


classifier_results = []
for model_name, pipeline in fitted_pipelines.items():
    result = evaluate_model(pipeline, X_test, y_test, model_name)
    classifier_results.append(result)

# %%
# put the metrics side by side in one table
comparison_table = pd.DataFrame(classifier_results)[
    ["model", "accuracy", "precision", "recall", "f1", "auc"]
]
print("\nModel comparison table:")
print(comparison_table.round(4).to_string(index=False))

# %% [markdown]
# ### ROC curves for all 3 models on one plot

# %%
plt.figure(figsize=(7, 6))
for result in classifier_results:
    fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
    plt.plot(fpr, tpr, label=f"{result['model']} (AUC={result['auc']:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All 3 Classifiers")
plt.legend()
plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_roc_curves.png")
plt.show()

# %% [markdown]
# ## Step 6: Imbalance handling comparison
#
# We compare 3 ways of dealing with class imbalance, using Random Forest
# as the one model for this sub-task. SMOTE is applied only to the
# training fold (never the test fold), so we don't leak synthetic test
# information.

# %%
print("Class balance in the training set:")
print(y_train.value_counts())
print("(0 = did not survive, 1 = survived)")

# %%
# (a) baseline - no special handling
baseline_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])
baseline_pipe.fit(X_train, y_train)
baseline_pred = baseline_pipe.predict(X_test)

# %%
# (b) class_weight='balanced'
balanced_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42)),
])
balanced_pipe.fit(X_train, y_train)
balanced_pred = balanced_pipe.predict(X_test)

# %%
# (c) SMOTE - oversample the training fold only, then train a plain model
# we can't put SMOTE inside the same sklearn Pipeline easily without
# imblearn's own Pipeline class, so we do it as two explicit steps instead

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_transformed, y_train)

print("Training set size before SMOTE:", X_train_transformed.shape[0])
print("Training set size after SMOTE:", X_train_smote.shape[0])

smote_model = RandomForestClassifier(random_state=42)
smote_model.fit(X_train_smote, y_train_smote)
smote_pred = smote_model.predict(X_test_transformed)

# %%
imbalance_results = []
for label, predictions in [
    ("(a) Baseline", baseline_pred),
    ("(b) class_weight='balanced'", balanced_pred),
    ("(c) SMOTE", smote_pred),
]:
    imbalance_results.append({
        "strategy": label,
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    })

imbalance_table = pd.DataFrame(imbalance_results)
print("\nImbalance handling comparison:")
print(imbalance_table.round(4).to_string(index=False))

# %% [markdown]
# **Conclusion:** Looking at the F1 scores above, whichever strategy has
# the highest F1 is doing the best job of balancing precision and recall
# together (rather than just chasing accuracy, which can be misleading on
# an imbalanced dataset like this one). In most runs, `class_weight` or
# SMOTE improves recall for the minority class compared to the baseline,
# at some small cost to precision - check the printed numbers above for
# this specific run to see which one actually won.

# %% [markdown]
# ## Step 7: Hyperparameter tuning with GridSearchCV
#
# We search over `n_estimators`, `max_depth`, and `max_features` for the
# Random Forest. We build it with `oob_score=True` so we can report the
# out-of-bag score, which is only available when that flag is set.

# %%
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__max_features": ["sqrt", "log2"],
}

tuning_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True)),
])

grid_search = GridSearchCV(
    tuning_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)

best_rf_model = grid_search.best_estimator_.named_steps["classifier"]
print(f"Out-of-bag (OOB) score of the best model: {best_rf_model.oob_score_:.4f}")

# %% [markdown]
# ## Step 8: Regression side-task - predicting fare
#
# Same dataset, but now we predict `fare` (a number) instead of `survived`
# (a category), using multivariate linear regression.

# %%
regression_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
X_reg = model_data[regression_features]
y_reg = model_data["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regression_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
     ["age", "sibsp", "parch"]),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]),
     ["sex", "embarked", "pclass"]),
])

regression_pipeline = Pipeline(steps=[
    ("preprocessor", regression_preprocessor),
    ("regressor", LinearRegression()),
])
regression_pipeline.fit(X_reg_train, y_reg_train)

fare_predictions = regression_pipeline.predict(X_reg_test)

# %%
mae = mean_absolute_error(y_reg_test, fare_predictions)
mse = mean_squared_error(y_reg_test, fare_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_test, fare_predictions)

n_rows = len(y_reg_test)
n_predictors = X_reg_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n_rows - 1) / (n_rows - n_predictors - 1)

print("Regression metrics (predicting fare):")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")

# %% [markdown]
# ### Residual plot

# %%
residuals = y_reg_test - fare_predictions

plt.figure(figsize=(7, 5))
plt.scatter(fare_predictions, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot - Fare Regression")
plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_residuals.png")
plt.show()

# %% [markdown]
# **Interpretation:** The residuals fan out wider as the predicted fare
# increases, instead of staying in an even band around zero. This uneven
# spread is a sign of **heteroscedasticity** - the model is much less
# precise for expensive tickets than for cheap ones. This makes sense
# since a handful of very expensive first-class fares are hard to predict
# exactly from just a few passenger features.

# %% [markdown]
# ## Step 9: Final model comparison table + recommendation

# %%
print("=" * 60)
print("CLASSIFICATION MODELS")
print("=" * 60)
print(comparison_table.round(4).to_string(index=False))

print("\n" + "=" * 60)
print("REGRESSION MODEL (fare prediction)")
print("=" * 60)
regression_summary = pd.DataFrame([{
    "model": "Linear Regression",
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2": round(r2, 4),
    "Adjusted_R2": round(adjusted_r2, 4),
}])
print(regression_summary.to_string(index=False))

# %% [markdown]
# **Final recommendation:** Based on the comparison table above, the model
# with the best combination of F1 score and AUC is the one to deploy,
# since F1 balances precision/recall (important on an imbalanced target
# like survival) and AUC measures how well the model ranks survivors above
# non-survivors overall. Random Forest usually edges out the other two
# because it can capture non-linear interactions (like the sex+class
# pattern we saw in the EDA chart), while still avoiding the overfitting a
# single Decision Tree is prone to. (Confirm this against the exact
# numbers printed above for this run, since results shift slightly with
# random splits.)

# %% [markdown]
# ## Step 10: Save the best pipeline with joblib
#
# We save the FULL pipeline (preprocessing + model together), not just the
# bare classifier, so it can be reloaded and used directly on raw,
# unprocessed new data.

# %%
best_full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", best_rf_model),
])
best_full_pipeline.fit(X_train, y_train)

joblib.dump(best_full_pipeline, THIS_FOLDER / "titanic_pipeline.joblib")
print("Saved full pipeline to titanic_pipeline.joblib")

# %%
# reload it and confirm it still works on raw input
reloaded_pipeline = joblib.load(THIS_FOLDER / "titanic_pipeline.joblib")

sample_raw_passenger = pd.DataFrame([{
    "pclass": 1, "sex": "female", "age": 29, "sibsp": 0,
    "parch": 0, "fare": 100.0, "embarked": "S",
}])

sample_prediction = reloaded_pipeline.predict(sample_raw_passenger)
print("Reloaded pipeline prediction on a sample raw passenger:", sample_prediction)
print("(1 = predicted survived, 0 = predicted did not survive)")

print("\nModeling complete.")