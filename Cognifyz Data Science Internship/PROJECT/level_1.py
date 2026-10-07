"""
Cognifyz Data Science Internship - Level 1 Tasks
Task 1: Data Exploration and Preprocessing
Task 2: Descriptive Analysis
Task 3: Geospatial Analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
os.makedirs('output_plots', exist_ok=True)

print("=" * 70)
print("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 1")
print("=" * 70)

# -----------------------------------------------------------------------------
# TASK 1: DATA EXPLORATION AND PREPROCESSING
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 1: DATA EXPLORATION AND PREPROCESSING")
print("=" * 50)

# 1.1 Load dataset
df = pd.read_csv('Dataset .csv')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("--- Columns and Data Types ---")
print(df.dtypes)

print("\n--- Missing Values Check ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

# 1.2 Handle missing values
if df['Cuisines'].isnull().sum() > 0:
    missing_cuisines_count = df['Cuisines'].isnull().sum()
    print(f"\nHandling {missing_cuisines_count} missing values in 'Cuisines' column by imputing 'Unknown'...")
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')

# Check duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate Rows Count: {duplicates}")

# 1.3 Analyze target variable ('Aggregate rating')
print("\n--- Target Variable ('Aggregate rating') Statistics ---")
print(df['Aggregate rating'].describe())

# Zero rating check
zero_ratings = (df['Aggregate rating'] == 0).sum()
zero_percent = (zero_ratings / len(df)) * 100
print(f"\nRestaurants with 0.0 Rating: {zero_ratings} ({zero_percent:.2f}%)")

# Plot Rating Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Aggregate rating'], bins=20, kde=True, ax=axes[0], color='#2b5c8f')
axes[0].set_title('Aggregate Rating Distribution (Overall)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Aggregate Rating')
axes[0].set_ylabel('Count')

# Rating without 0.0
rated_df = df[df['Aggregate rating'] > 0]
sns.histplot(rated_df['Aggregate rating'], bins=20, kde=True, ax=axes[1], color='#1b9e77')
axes[1].set_title('Aggregate Rating Distribution (Excluding 0.0 / Unrated)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Aggregate Rating')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('output_plots/level1_task1_rating_distribution.png')
plt.close()
print("Saved plot: output_plots/level1_task1_rating_distribution.png")


# -----------------------------------------------------------------------------
# TASK 2: DESCRIPTIVE ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 2: DESCRIPTIVE ANALYSIS")
print("=" * 50)

# 2.1 Statistical measures for numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical Columns Summary Statistics:")
desc_stats = df[num_cols].describe().T
desc_stats['median'] = df[num_cols].median()
desc_stats['skewness'] = df[num_cols].skew()
print(desc_stats[['count', 'mean', 'std', 'min', 'median', 'max', 'skewness']])

# 2.2 Categorical variables distribution
print("\n--- Categorical Variables Distribution ---")
print(f"Total Unique Countries: {df['Country Code'].nunique()}")
print(f"Total Unique Cities: {df['City'].nunique()}")
print(f"Total Unique Cuisines Entries: {df['Cuisines'].nunique()}")

# Top 10 Cities
top_cities = df['City'].value_counts().head(10)
print("\nTop 10 Cities by Restaurant Count:")
print(top_cities)

# Top 10 Individual Cuisines
cuisines_series = df['Cuisines'].str.split(', ').explode()
top_cuisines = cuisines_series.value_counts().head(10)
print("\nTop 10 Cuisines by Restaurant Count:")
print(top_cuisines)

# Visualize Top Cities and Top Cuisines
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=top_cities.values, y=top_cities.index, ax=axes[0], palette='Blues_r', hue=top_cities.index, legend=False)
axes[0].set_title('Top 10 Cities with Highest Number of Restaurants', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Restaurants')
for i, v in enumerate(top_cities.values):
    axes[0].text(v + 10, i, str(v), va='center', fontweight='semibold')

sns.barplot(x=top_cuisines.values, y=top_cuisines.index, ax=axes[1], palette='Greens_r', hue=top_cuisines.index, legend=False)
axes[1].set_title('Top 10 Cuisines Offered Across Restaurants', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Count')
for i, v in enumerate(top_cuisines.values):
    axes[1].text(v + 20, i, str(v), va='center', fontweight='semibold')

plt.tight_layout()
plt.savefig('output_plots/level1_task2_top_cities_cuisines.png')
plt.close()
print("Saved plot: output_plots/level1_task2_top_cities_cuisines.png")


# -----------------------------------------------------------------------------
# TASK 3: GEOSPATIAL ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 3: GEOSPATIAL ANALYSIS")
print("=" * 50)

# Filter valid coordinates (excluding 0, 0 coords if any)
geo_df = df[(df['Latitude'] != 0) | (df['Longitude'] != 0)]
print(f"Total Restaurants with valid coordinates: {len(geo_df)} / {len(df)}")

# Correlation between location and rating
corr_lat = df['Latitude'].corr(df['Aggregate rating'])
corr_long = df['Longitude'].corr(df['Aggregate rating'])
print(f"Correlation between Latitude and Aggregate Rating: {corr_lat:.4f}")
print(f"Correlation between Longitude and Aggregate Rating: {corr_long:.4f}")

# Geospatial Scatter Plot
plt.figure(figsize=(14, 7))
scatter = plt.scatter(
    geo_df['Longitude'], geo_df['Latitude'], 
    c=geo_df['Aggregate rating'], cmap='Spectral_r', 
    alpha=0.6, s=18, edgecolors='none'
)
plt.colorbar(scatter, label='Aggregate Rating')
plt.title('Global Restaurant Geospatial Distribution & Ratings', fontsize=15, fontweight='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output_plots/level1_task3_geospatial_scatter.png')
plt.close()
print("Saved plot: output_plots/level1_task3_geospatial_scatter.png")

# India-specific map (Country code 1 has the largest density)
india_df = df[df['Country Code'] == 1]
if len(india_df) > 0:
    plt.figure(figsize=(10, 8))
    scatter_in = plt.scatter(
        india_df['Longitude'], india_df['Latitude'],
        c=india_df['Aggregate rating'], cmap='viridis',
        alpha=0.6, s=20
    )
    plt.colorbar(scatter_in, label='Aggregate Rating')
    plt.title('Restaurant Distribution in India (Highest Concentration)', fontsize=14, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('output_plots/level1_task3_india_geo.png')
    plt.close()
    print("Saved plot: output_plots/level1_task3_india_geo.png")

print("\nLevel 1 analysis complete! All metrics computed and plots saved successfully.")
