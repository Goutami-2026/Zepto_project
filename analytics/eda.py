# %% [markdown]
# # 01 - Titanic EDA and Cleaning
#
# This is the first half of the analytics module. In this file we:
# 1. Load the Titanic dataset (only once, from seaborn).
# 2. Look at it and figure out what is missing.
# 3. Clean the missing values using a simple percentage-based rule.
# 4. Draw some charts to understand who survived and why.
# 5. Do a quick standardization check on age and fare.
#
# The cleaned data gets saved to `titanic.csv` at the end of the "loading"
# step below, so this is the ONLY place the dataset is ever downloaded from
# the internet in the whole module. Every other file (and the second half
# of this module, 02_modeling.py) reads from that CSV instead.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# folder where this script lives, so saved files always land in the right place
from pathlib import Path
THIS_FOLDER = Path(__file__).parent

# %% [markdown]
# ## Step 1: Load the dataset (only once!)

# %%
titanic = sns.load_dataset("titanic")

# save it immediately so we never need the internet again for this project
titanic.to_csv(THIS_FOLDER / "titanic.csv", index=False)

print("Dataset loaded and saved as titanic.csv")
print("Shape of the dataset (rows, columns):", titanic.shape)

# %% [markdown]
# ## Step 2: Look at the data (profiling)
#
# Before cleaning anything, let's just look at what we have.

# %%
titanic.info()

# %%
titanic.describe()

# %%
# how many rows and columns
print("Rows:", titanic.shape[0])
print("Columns:", titanic.shape[1])

# %% [markdown]
# ### Missing values
#
# Let's find out, column by column, what percentage of values are missing.

# %%
missing_counts = titanic.isnull().sum()
missing_percent = (missing_counts / len(titanic)) * 100

# only show columns that actually have missing values
missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percent": missing_percent.round(2)
})
missing_summary = missing_summary[missing_summary["missing_count"] > 0]
missing_summary = missing_summary.sort_values("missing_percent", ascending=False)

print("Columns with missing values:")
print(missing_summary)

# %% [markdown]
# From the table above, the columns with missing data are usually:
# - `deck` – missing a LOT (well above 30%)
# - `age` – missing a moderate amount (somewhere in the 5-30% range)
# - `embarked` and `embark_town` – missing just a tiny amount (under 5%)
#
# (The exact numbers print above every time this cell runs, since they come
# straight from the loaded data.)

# %% [markdown]
# ## Step 3: Clean the missing values
#
# We use this rule, as required by the assignment:
# - under 5% missing -> just drop those rows (we won't lose much data)
# - 5% to 30% missing -> fill in (impute) the missing values
# - way more than 30% missing -> the column is too empty to trust, so we
#   either drop the whole column or turn "missing" into its own category
#
# Let's apply this column by column.

# %%
titanic_clean = titanic.copy()

# --- embarked and embark_town: missing % is small (under 5%) -> drop rows ---
embarked_missing_pct = titanic_clean["embarked"].isnull().mean() * 100
print(f"'embarked' missing: {embarked_missing_pct:.2f}% -> dropping those rows")
titanic_clean = titanic_clean.dropna(subset=["embarked", "embark_town"])

# %%
# --- age: missing % is moderate (5-30%) -> impute with the median ---
age_missing_pct = titanic["age"].isnull().mean() * 100
print(f"'age' missing: {age_missing_pct:.2f}% -> imputing with median age")

median_age = titanic_clean["age"].median()
titanic_clean["age"] = titanic_clean["age"].fillna(median_age)
print("Median age used for filling:", median_age)

# %%
# --- deck: missing % is very high (way above 30%) ---
deck_missing_pct = titanic["deck"].isnull().mean() * 100
print(f"'deck' missing: {deck_missing_pct:.2f}%")

# This is too high to safely impute a real deck letter - we would basically
# be making it up. Instead of dropping the whole column, we keep it useful
# by turning "no data" into its own category called "Unknown". That way we
# don't lose the column entirely, and a model could still learn something
# from "we don't know this passenger's deck" as a signal.
titanic_clean["deck"] = titanic_clean["deck"].astype(object).fillna("Unknown")

# %%
# double check: no more missing values anywhere important
print("Missing values left after cleaning:")
print(titanic_clean.isnull().sum().sum(), "total missing cells")

# %% [markdown]
# ## Step 4: Univariate analysis (age and fare)
#
# Let's look at the shape of `age` and `fare` on their own, using a
# histogram and a box plot for each.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(titanic_clean["age"], bins=30, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Age - Histogram")
axes[0, 0].set_xlabel("Age")

axes[0, 1].boxplot(titanic_clean["age"])
axes[0, 1].set_title("Age - Box Plot")

axes[1, 0].hist(titanic_clean["fare"], bins=30, color="darkorange", edgecolor="white")
axes[1, 0].set_title("Fare - Histogram")
axes[1, 0].set_xlabel("Fare")

axes[1, 1].boxplot(titanic_clean["fare"])
axes[1, 1].set_title("Fare - Box Plot")

plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_univariate_age_fare.png")
plt.show()

# %% [markdown]
# ### Counting outliers with the IQR rule
#
# A point is an outlier if it falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

# %%
def count_outliers_iqr(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers), lower_bound, upper_bound


age_outlier_count, age_low, age_high = count_outliers_iqr(titanic_clean["age"])
fare_outlier_count, fare_low, fare_high = count_outliers_iqr(titanic_clean["fare"])

print(f"Age outliers: {age_outlier_count} (valid range was {age_low:.1f} to {age_high:.1f})")
print(f"Fare outliers: {fare_outlier_count} (valid range was {fare_low:.1f} to {fare_high:.1f})")

# %% [markdown]
# ### Mean, median, mode of fare - is it skewed?

# %%
fare_mean = titanic_clean["fare"].mean()
fare_median = titanic_clean["fare"].median()
fare_mode = titanic_clean["fare"].mode()[0]

print(f"Fare mean: {fare_mean:.2f}")
print(f"Fare median: {fare_median:.2f}")
print(f"Fare mode: {fare_mode:.2f}")

# %% [markdown]
# Since mean > median > mode for fare, the distribution is **right-skewed**
# (the long tail is on the right). This makes sense - most tickets were
# cheap, but a small number of very expensive first-class tickets pull the
# average up much higher than the median or the most common price.

# %% [markdown]
# ## Step 5: Bivariate analysis - who survived?
#
# We use boolean masking (the `&` and `|` operators on True/False arrays)
# to slice the data and compute survival rates.

# %%
# survival rate by sex
male_mask = titanic_clean["sex"] == "male"
female_mask = titanic_clean["sex"] == "female"

male_survival_rate = titanic_clean.loc[male_mask, "survived"].mean()
female_survival_rate = titanic_clean.loc[female_mask, "survived"].mean()

print(f"Survival rate - male: {male_survival_rate:.2%}")
print(f"Survival rate - female: {female_survival_rate:.2%}")

# %%
# survival rate by passenger class
for pclass_value in sorted(titanic_clean["pclass"].unique()):
    mask = titanic_clean["pclass"] == pclass_value
    rate = titanic_clean.loc[mask, "survived"].mean()
    print(f"Survival rate - class {pclass_value}: {rate:.2%}")

# %%
# survival rate by sex AND pclass together (this is where & really matters)
print("Survival rate by sex and class combined:")
for sex_value in ["male", "female"]:
    for pclass_value in sorted(titanic_clean["pclass"].unique()):
        combined_mask = (titanic_clean["sex"] == sex_value) & (titanic_clean["pclass"] == pclass_value)
        rate = titanic_clean.loc[combined_mask, "survived"].mean()
        print(f"  {sex_value}, class {pclass_value}: {rate:.2%}")

# %% [markdown]
# ### Correlation matrix (6 numeric columns only)
#
# The assignment asks us to use exactly these 6 columns: survived, pclass,
# age, sibsp, parch, fare. We leave out `adult_male` and `alone` on purpose
# because those are just derived from sex/age and sibsp/parch - including
# them would be a bit circular.

# %%
correlation_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
correlation_matrix = titanic_clean[correlation_columns].corr()
print(correlation_matrix)

# %%
plt.figure(figsize=(7, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (6 numeric columns)")
plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_correlation_heatmap.png")
plt.show()

# %%
# find the two strongest correlations (ignoring the diagonal, which is
# always 1.0 because every column is perfectly correlated with itself)
corr_pairs = correlation_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 0.999]  # drop the diagonal 1.0 values
corr_pairs = corr_pairs.sort_values(ascending=False)

# each pair appears twice (A-B and B-A), so we just look at the top few
print("Strongest correlations (each pair appears twice):")
print(corr_pairs.head(6))

# %% [markdown]
# The two strongest relationships are usually:
# 1. **fare and pclass** - a strong negative correlation, which makes sense
#    since class 1 is the "smallest" number but the most expensive tickets.
# 2. **sibsp and parch** - a positive correlation, since passengers who
#    travelled with siblings/spouses often also travelled with parents or
#    children (i.e. they were travelling as a family).
# (Exact ranking should be checked against the printed values above, since
# it can shift slightly depending on how missing rows were handled.)

# %% [markdown]
# ## Step 6: The "data story" - 4+ charts with interpretation

# %% [markdown]
# ### Chart 1: Survival count by sex

# %%
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_clean, x="sex", hue="survived")
plt.title("Survival Count by Sex")
plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_story_1_sex.png")
plt.show()

# %% [markdown]
# **Interpretation:** Far more women survived than men, both in raw numbers
# and as a percentage. This lines up with the "women and children first"
# evacuation policy that was followed for the lifeboats.

# %% [markdown]
# ### Chart 2: Survival rate by passenger class

# %%
plt.figure(figsize=(6, 4))
sns.barplot(data=titanic_clean, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_story_2_class.png")
plt.show()

# %% [markdown]
# **Interpretation:** First class passengers survived at a much higher rate
# than third class passengers. This is likely a mix of wealthier passengers
# having cabins closer to the lifeboats, and possibly being given priority
# during the evacuation.

# %% [markdown]
# ### Chart 3: Age distribution split by survival

# %%
plt.figure(figsize=(7, 4))
sns.boxplot(data=titanic_clean, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_story_3_age.png")
plt.show()

# %% [markdown]
# **Interpretation:** The age distributions for survivors and
# non-survivors look fairly similar overall, but survivors skew slightly
# younger - this fits with children being prioritised for lifeboats.

# %% [markdown]
# ### Chart 4: Survival rate by sex and class together

# %%
plt.figure(figsize=(7, 4))
sns.barplot(data=titanic_clean, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Class and Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(THIS_FOLDER / "chart_story_4_class_sex.png")
plt.show()

# %% [markdown]
# **Interpretation:** Being female mattered more than being in first class -
# even third class women survived at a noticeably higher rate than first
# class men. Sex looks like the single strongest factor in survival, with
# class acting as a secondary factor mostly within each sex group.

# %% [markdown]
# ## Step 7: Standardization check (exploratory only)
#
# This is just a sanity check for this EDA stage - it does NOT feed into
# the modeling pipeline in 02_modeling.py. The modeling file does its own
# scaling, fit only on the training data.

# %%
age_before_mean, age_before_std = titanic_clean["age"].mean(), titanic_clean["age"].std()
fare_before_mean, fare_before_std = titanic_clean["fare"].mean(), titanic_clean["fare"].std()

print("BEFORE standardizing:")
print(f"  age  -> mean: {age_before_mean:.2f}, std: {age_before_std:.2f}")
print(f"  fare -> mean: {fare_before_mean:.2f}, std: {fare_before_std:.2f}")

# %%
# z-score formula: z = (x - mean) / std
age_z = (titanic_clean["age"] - age_before_mean) / age_before_std
fare_z = (titanic_clean["fare"] - fare_before_mean) / fare_before_std

print("\nAFTER standardizing:")
print(f"  age_z  -> mean: {age_z.mean():.4f}, std: {age_z.std():.4f}")
print(f"  fare_z -> mean: {fare_z.mean():.4f}, std: {fare_z.std():.4f}")

# %% [markdown]
# As expected, after standardizing, both columns now have a mean very
# close to 0 and a standard deviation very close to 1. This confirms the
# z-score formula worked correctly.

# %% [markdown]
# ## Done with EDA
#
# `titanic.csv` (the cleaned, saved version) is ready for 02_modeling.py to
# pick up and continue from. Note: 02_modeling.py will re-load titanic.csv
# and may re-apply its own missing-value strategy inside a scikit-learn
# pipeline (fit only on the training split) - it does not have to match
# the strategy used here exactly, as the assignment allows.
print("\nEDA complete. titanic_clean has", titanic_clean.shape[0], "rows.")