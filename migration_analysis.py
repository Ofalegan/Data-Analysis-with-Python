# ============================================================
# YOUTH INTERNAL MIGRATION ANALYSIS - NIGERIA 2023
# ============================================================
# This script analyzes factors affecting youth migration in Nigeria
# including unemployment, poverty, opportunity gaps, and demographics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

print("Libraries loaded successfully!")


# ============================================================
# 1. LOAD DATASET
# ============================================================
data_path = "youth_internal_migration_nigeria_2023.csv"
df = pd.read_csv(data_path)

print("\nDataset loaded successfully!")
print(df.head())


# ============================================================
# 2. BASIC INFO & SUMMARY STATISTICS
# ============================================================
print("\n=== DATA INFO ===")
df.info()

print("\n=== SUMMARY STATISTICS ===")
print(df.describe(include="all"))


# ============================================================
# 3. VALUE COUNTS
# ============================================================
print("\n=== MIGRATION (Last 5 Years) ===")
print(df["migrated_5yrs"].value_counts(normalize=True))

print("\n=== UNEMPLOYMENT ===")
print(df["unemployed"].value_counts(normalize=True))

print("\n=== ZONE DISTRIBUTION ===")
print(df["zone"].value_counts(normalize=True))


# ============================================================
# 4. CREATE DERIVED VARIABLES
# ============================================================
df["opp_gap"] = df["opp_destination"] - df["opp_origin"]
df["poor_household"] = (df["household_poverty"] <= 2).astype(int)

print("\n=== DERIVED VARIABLES CREATED ===")
print(f"opp_gap range: {df['opp_gap'].min()} to {df['opp_gap'].max()}")
print(f"poor_household share: {df['poor_household'].mean():.2%}")


# ============================================================
# 5. VISUALIZATIONS
# ============================================================

# 5.1 Migration by Zone
plt.figure(figsize=(8,5))
df.groupby("zone")["migrated_5yrs"].mean().sort_values().plot(kind="bar")
plt.title("Internal Migration Among Nigerian Youth by Zone")
plt.ylabel("Share Migrated (5yrs)")
plt.tight_layout()
plt.show()

# 5.2 Income by Migration Status
plt.figure(figsize=(6,5))
df.boxplot(column="monthly_income", by="migrated_5yrs")
plt.title("Monthly Income by Migration Status")
plt.suptitle("")
plt.xlabel("Migrated (0=No, 1=Yes)")
plt.ylabel("Monthly Income (₦)")
plt.tight_layout()
plt.show()

# 5.3 Correlation Heatmap
numeric_cols = [
    "age","sex","origin_area","current_area","education_level",
    "unemployed","underemployed","monthly_income","household_poverty",
    "opp_origin","opp_destination","security_origin","security_destination",
    "migrated_5yrs","intent_migrate_12m"
]
corr = df[numeric_cols].corr()
plt.figure(figsize=(10,8))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5.4 Age Distribution
plt.figure(figsize=(7,4))
plt.hist(df["age"], bins=12, edgecolor='black', color='steelblue')
plt.title("Age Distribution of Youth (18–35)")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 5.5 Migration by Education Level
plt.figure(figsize=(8,5))
df.groupby("education_level")["migrated_5yrs"].mean().plot(kind="bar", color='coral')
plt.title("Migration Rate by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Share Migrated (5yrs)")
plt.tight_layout()
plt.show()

# 5.6 Unemployment by Zone
plt.figure(figsize=(8,5))
df.groupby("zone")["unemployed"].mean().plot(kind="bar", color='orange')
plt.title("Unemployment Rate by Zone")
plt.ylabel("Unemployment Share")
plt.tight_layout()
plt.show()

# 5.7 Opportunity Gap vs Migration
plt.figure(figsize=(7,5))
plt.scatter(df["opp_gap"], df["migrated_5yrs"], alpha=0.4, color='purple')
plt.title("Opportunity Gap vs Migration")
plt.xlabel("Opportunity Gap (Destination - Origin)")
plt.ylabel("Migrated (0/1)")
plt.tight_layout()
plt.show()

# 5.8 Household Poverty by Migration Status
plt.figure(figsize=(6,5))
df.boxplot(column="household_poverty", by="migrated_5yrs")
plt.title("Household Poverty by Migration Status")
plt.suptitle("")
plt.xlabel("Migrated (0=No, 1=Yes)")
plt.ylabel("Poverty Score (1=Very Poor)")
plt.tight_layout()
plt.show()

# 5.9 Unemployment vs Intent to Migrate
plt.figure(figsize=(7,5))
plt.scatter(df["unemployed"], df["intent_migrate_12m"], alpha=0.3, color='green')
plt.title("Unemployment vs 12-Month Migration Intent")
plt.xlabel("Unemployed (0/1)")
plt.ylabel("Intent to Migrate (0/1)")
plt.tight_layout()
plt.show()

# 5.10 Income Distribution
plt.figure(figsize=(7,4))
plt.hist(df["monthly_income"], bins=20, edgecolor='black', color='teal')
plt.title("Income Distribution of Youth")
plt.xlabel("Monthly Income (₦)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 6. PREPARE DATA FOR LOGISTIC REGRESSION
# ============================================================

# Select variables for regression
X = df[[
    "unemployed","poor_household","opp_gap","origin_area",
    "age","sex","education_level","zone"
]].copy()

y = df["migrated_5yrs"].copy()

# Create dummy variables for zone
X = pd.get_dummies(X, columns=["zone"], drop_first=True, dtype=int)

# Force all columns to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

y = pd.to_numeric(y, errors="coerce")

# Merge and clean
combined = pd.concat([y.rename("migrated_5yrs"), X], axis=1)
combined_clean = combined.dropna()

print(f"\n=== DATA CLEANING SUMMARY ===")
print(f"Original rows: {len(combined)}")
print(f"After removing missing: {len(combined_clean)}")
print(f"Rows dropped: {len(combined) - len(combined_clean)}")

# Separate y and X after cleaning
y_clean = combined_clean["migrated_5yrs"]
X_clean = combined_clean.drop(columns=["migrated_5yrs"])

# Add constant for intercept
X_clean = sm.add_constant(X_clean)

print("\nRegression data prepared successfully!")
print(f"X shape: {X_clean.shape}")
print(f"y shape: {y_clean.shape}")


# ============================================================
# 7. RUN LOGISTIC REGRESSION
# ============================================================
print("\n" + "="*60)
print("RUNNING LOGISTIC REGRESSION...")
print("="*60 + "\n")

model = sm.Logit(y_clean, X_clean)
result = model.fit()

print(result.summary())


# ============================================================
# 8. ODDS RATIOS & INTERPRETATION
# ============================================================
print("\n" + "="*60)
print("ODDS RATIOS (exp(coefficients))")
print("="*60)

odds_ratios = np.exp(result.params)
ci = np.exp(result.conf_int())
ci.columns = ['2.5%', '97.5%']

odds_summary = pd.DataFrame({
    'Odds Ratio': odds_ratios,
    'Lower CI': ci['2.5%'],
    'Upper CI': ci['97.5%'],
    'p-value': result.pvalues
})

print(odds_summary.round(3))

print("\n--- INTERPRETATION ---")
print("Odds Ratio > 1: Variable INCREASES odds of migration")
print("Odds Ratio < 1: Variable DECREASES odds of migration")
print("Odds Ratio = 1: Variable has NO effect on migration")
print("p-value < 0.05: Result is statistically significant")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)