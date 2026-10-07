"""
Cognifyz Data Science Internship - Level 3 Tasks
Task 1: Predictive Modeling (Regression for Aggregate Rating)
Task 2: Customer Preference Analysis
Task 3: Data Visualization & Insights
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
os.makedirs('output_plots', exist_ok=True)

print("=" * 70)
print("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 3")
print("=" * 70)

# Load dataset (or engineered dataset if present)
if os.path.exists('engineered_dataset.csv'):
    df = pd.read_csv('engineered_dataset.csv')
else:
    df = pd.read_csv('Dataset .csv')
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    df['Restaurant_Name_Length'] = df['Restaurant Name'].astype(str).apply(len)
    df['Address_Length'] = df['Address'].astype(str).apply(len)
    df['Cuisines_Count'] = df['Cuisines'].apply(lambda x: len(str(x).split(',')))
    df['Has_Table_Booking_Code'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
    df['Has_Online_Delivery_Code'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

# -----------------------------------------------------------------------------
# TASK 1: PREDICTIVE MODELING
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 1: PREDICTIVE MODELING")
print("=" * 50)

# Feature selection
# Predict 'Aggregate rating' based on available features (excluding rating leakage like 'Rating color', 'Rating text')
features_num = ['Average Cost for two', 'Price range', 'Votes', 'Restaurant_Name_Length', 'Address_Length', 'Cuisines_Count', 'Latitude', 'Longitude']
features_cat = ['Country Code', 'City', 'Has Table booking', 'Has Online delivery']

# Top 20 cities to prevent dimensionality explosion, others grouped into 'Other'
top_20_cities = df['City'].value_counts().head(20).index
df_model = df.copy()
df_model['City_Grouped'] = df_model['City'].apply(lambda x: x if x in top_20_cities else 'Other')

X = df_model[features_num + ['City_Grouped', 'Country Code', 'Has Table booking', 'Has Online delivery']]
y = df_model['Aggregate rating']

# Train / Test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['City_Grouped', 'Country Code', 'Has Table booking', 'Has Online delivery'])
    ]
)

# Models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'R2 Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse
    })

results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
print("\n--- Model Performance Comparison ---")
print(results_df.to_string(index=False))

# Best model evaluation
best_model_name = results_df.iloc[0]['Model']
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', models[best_model_name])
])
best_pipeline.fit(X_train, y_train)
y_pred_best = best_pipeline.predict(X_test)

# Plot Actual vs Predicted & Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x='R2 Score', y='Model', data=results_df, palette='Blues_r', ax=axes[0], hue='Model', legend=False)
axes[0].set_title('Model Performance Comparison (R² Score)', fontsize=13, fontweight='bold')
axes[0].set_xlim(0, 1.0)
for i, r in enumerate(results_df['R2 Score']):
    axes[0].text(r + 0.02, i, f"{r:.4f}", va='center', fontweight='bold')

sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.3, color='#1f77b4', ax=axes[1])
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_title(f'Actual vs Predicted Ratings ({best_model_name})', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Actual Aggregate Rating')
axes[1].set_ylabel('Predicted Aggregate Rating')
axes[1].legend()

plt.tight_layout()
plt.savefig('output_plots/level3_task1_model_evaluation.png')
plt.close()
print("Saved plot: output_plots/level3_task1_model_evaluation.png")


# -----------------------------------------------------------------------------
# TASK 2: CUSTOMER PREFERENCE ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 2: CUSTOMER PREFERENCE ANALYSIS")
print("=" * 50)

# Unpack individual cuisines
exploded_cuisines = df.assign(Cuisine=df['Cuisines'].str.split(', ')).explode('Cuisine')
exploded_cuisines['Cuisine'] = exploded_cuisines['Cuisine'].str.strip()

# 2.1 Cuisines relationship with rating & customer votes
cuisine_stats = exploded_cuisines.groupby('Cuisine').agg(
    Restaurant_Count=('Restaurant ID', 'count'),
    Average_Rating=('Aggregate rating', 'mean'),
    Total_Votes=('Votes', 'sum'),
    Average_Votes=('Votes', 'mean')
).reset_index()

# Filter for cuisines with sufficient sample size (>= 20 restaurants)
frequent_cuisines = cuisine_stats[cuisine_stats['Restaurant_Count'] >= 20]

# Most popular cuisines by total customer votes
top_cuisines_votes = cuisine_stats.sort_values(by='Total_Votes', ascending=False).head(10)
print("\nTop 10 Most Popular Cuisines by Total Customer Votes:")
print(top_cuisines_votes[['Cuisine', 'Total_Votes', 'Average_Rating', 'Restaurant_Count']].to_string(index=False))

# Highest rated cuisines (with min 20 restaurants)
highest_rated_cuisines = frequent_cuisines.sort_values(by='Average_Rating', ascending=False).head(10)
print("\nTop 10 Highest-Rated Cuisines (Min. 20 Restaurants):")
print(highest_rated_cuisines[['Cuisine', 'Average_Rating', 'Restaurant_Count', 'Total_Votes']].to_string(index=False))

# Visualization for Customer Preferences
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x='Total_Votes', y='Cuisine', data=top_cuisines_votes, palette='Purples_r', ax=axes[0], hue='Cuisine', legend=False)
axes[0].set_title('Top 10 Most Popular Cuisines by Total Votes', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Total Votes')

sns.barplot(x='Average_Rating', y='Cuisine', data=highest_rated_cuisines, palette='YlGn_r', ax=axes[1], hue='Cuisine', legend=False)
axes[1].set_title('Top 10 Highest-Rated Cuisines (Min 20 Restaurants)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Average Aggregate Rating')
axes[1].set_xlim(3.0, 5.0)

plt.tight_layout()
plt.savefig('output_plots/level3_task2_customer_preferences.png')
plt.close()
print("Saved plot: output_plots/level3_task2_customer_preferences.png")


# -----------------------------------------------------------------------------
# TASK 3: DATA VISUALIZATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 3: DATA VISUALIZATION")
print("=" * 50)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2)

# Subplot 1: Distribution of Ratings with KDE
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df['Aggregate rating'], kde=True, bins=25, color='#3470a3', ax=ax1)
ax1.axvline(df['Aggregate rating'].mean(), color='red', linestyle='--', label=f"Mean: {df['Aggregate rating'].mean():.2f}")
ax1.axvline(df['Aggregate rating'].median(), color='green', linestyle=':', label=f"Median: {df['Aggregate rating'].median():.2f}")
ax1.set_title('Distribution of Aggregate Ratings with Mean & Median', fontsize=12, fontweight='bold')
ax1.set_xlabel('Aggregate Rating')
ax1.set_ylabel('Frequency')
ax1.legend()

# Subplot 2: Average Ratings of Top 10 Cities
ax2 = fig.add_subplot(gs[0, 1])
top_cities_list = df['City'].value_counts().head(10).index
city_ratings = df[df['City'].isin(top_cities_list)].groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)
sns.barplot(x=city_ratings.values, y=city_ratings.index, palette='crest_r', ax=ax2, hue=city_ratings.index, legend=False)
ax2.set_title('Average Aggregate Rating in Top 10 Cities by Count', fontsize=12, fontweight='bold')
ax2.set_xlabel('Mean Rating')
for i, v in enumerate(city_ratings.values):
    ax2.text(v - 0.35, i, f"{v:.2f}", color='white', fontweight='bold', va='center')

# Subplot 3: Votes vs Aggregate Rating scatter
ax3 = fig.add_subplot(gs[1, 0])
sns.scatterplot(x='Votes', y='Aggregate rating', hue='Price range', palette='viridis', data=df, alpha=0.6, ax=ax3)
ax3.set_title('Customer Votes vs Aggregate Rating by Price Range', fontsize=12, fontweight='bold')
ax3.set_xlabel('Votes')
ax3.set_ylabel('Aggregate Rating')

# Subplot 4: Correlation Heatmap of Key Variables
ax4 = fig.add_subplot(gs[1, 1])
heatmap_cols = ['Aggregate rating', 'Votes', 'Price range', 'Average Cost for two', 'Restaurant_Name_Length', 'Address_Length', 'Cuisines_Count']
corr_sub = df[heatmap_cols].corr()
sns.heatmap(corr_sub, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax4)
ax4.set_title('Correlation Matrix of Key Features with Rating', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output_plots/level3_task3_comprehensive_visualizations.png')
plt.close()
print("Saved plot: output_plots/level3_task3_comprehensive_visualizations.png")

print("\nLevel 3 analysis complete! All models trained, customer preferences analyzed, and plots saved successfully.")
