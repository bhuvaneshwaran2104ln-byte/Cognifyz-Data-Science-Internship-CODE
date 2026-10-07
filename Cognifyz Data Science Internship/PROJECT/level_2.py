"""
Cognifyz Data Science Internship - Level 2 Tasks
Task 1: Table Booking and Online Delivery Analysis
Task 2: Price Range Analysis
Task 3: Feature Engineering
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
os.makedirs('output_plots', exist_ok=True)

print("=" * 70)
print("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 2")
print("=" * 70)

# Load dataset
df = pd.read_csv('Dataset .csv')
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

# -----------------------------------------------------------------------------
# TASK 1: TABLE BOOKING AND ONLINE DELIVERY
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 1: TABLE BOOKING AND ONLINE DELIVERY")
print("=" * 50)

# 1.1 Percentage of restaurants offering table booking and online delivery
total_restaurants = len(df)
table_booking_pct = (df['Has Table booking'].value_counts(normalize=True) * 100).to_dict()
online_delivery_pct = (df['Has Online delivery'].value_counts(normalize=True) * 100).to_dict()

print(f"Percentage of restaurants with Table Booking:\n  Yes: {table_booking_pct.get('Yes', 0):.2f}%\n  No:  {table_booking_pct.get('No', 0):.2f}%")
print(f"\nPercentage of restaurants with Online Delivery:\n  Yes: {online_delivery_pct.get('Yes', 0):.2f}%\n  No:  {online_delivery_pct.get('No', 0):.2f}%")

# 1.2 Compare average ratings of restaurants with vs without table booking
avg_rating_table = df.groupby('Has Table booking')['Aggregate rating'].agg(['count', 'mean', 'median', 'std'])
print("\nAverage Aggregate Rating by Table Booking Availability:")
print(avg_rating_table)

# Statistical Significance (t-test)
group_yes = df[df['Has Table booking'] == 'Yes']['Aggregate rating']
group_no = df[df['Has Table booking'] == 'No']['Aggregate rating']
t_stat, p_val = stats.ttest_ind(group_yes, group_no, equal_var=False)
print(f"\nT-Test between Table Booking Yes vs No: t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")

# 1.3 Online delivery availability across price ranges
delivery_price_ct = pd.crosstab(df['Price range'], df['Has Online delivery'], normalize='index') * 100
print("\nPercentage Availability of Online Delivery across Price Ranges:")
print(delivery_price_ct)

# Visualizations for Task 1
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Donut chart for services availability
services_df = pd.DataFrame({
    'Service': ['Table Booking (Yes)', 'Table Booking (No)', 'Online Delivery (Yes)', 'Online Delivery (No)'],
    'Percentage': [table_booking_pct.get('Yes', 0), table_booking_pct.get('No', 0), online_delivery_pct.get('Yes', 0), online_delivery_pct.get('No', 0)],
    'Category': ['Table Booking', 'Table Booking', 'Online Delivery', 'Online Delivery']
})
colors = ['#2ca02c', '#d62728', '#1f77b4', '#aec7e8']
axes[0].pie([table_booking_pct.get('Yes', 0), table_booking_pct.get('No', 0)], 
            labels=['Yes', 'No'], autopct='%1.1f%%', colors=['#2ca02c', '#d62728'], startangle=90, 
            wedgeprops=dict(width=0.4, edgecolor='w'))
axes[0].set_title('Table Booking Availability', fontsize=12, fontweight='bold')

# Boxplot comparing ratings
sns.boxplot(x='Has Table booking', y='Aggregate rating', data=df, palette='Set2', ax=axes[1], hue='Has Table booking', legend=False)
axes[1].set_title('Aggregate Rating by Table Booking', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Has Table Booking')
axes[1].set_ylabel('Aggregate Rating')

# Stacked bar for online delivery by price range
delivery_price_ct.plot(kind='bar', stacked=True, ax=axes[2], color=['#d62728', '#2ca02c'])
axes[2].set_title('Online Delivery Availability by Price Range', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Price Range (1: Low to 4: High)')
axes[2].set_ylabel('Percentage (%)')
axes[2].legend(title='Online Delivery', loc='upper right')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('output_plots/level2_task1_services_analysis.png')
plt.close()
print("Saved plot: output_plots/level2_task1_services_analysis.png")


# -----------------------------------------------------------------------------
# TASK 2: PRICE RANGE ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 2: PRICE RANGE ANALYSIS")
print("=" * 50)

# 2.1 Most common price range
price_counts = df['Price range'].value_counts().sort_index()
most_common_price = df['Price range'].mode()[0]
print(f"Price Range Counts:\n{price_counts}")
print(f"\nMost Common Price Range: {most_common_price} with {price_counts[most_common_price]} restaurants ({(price_counts[most_common_price]/len(df))*100:.2f}%)")

# 2.2 Average rating for each price range
avg_rating_price = df.groupby('Price range')['Aggregate rating'].agg(['count', 'mean', 'median', 'std'])
print("\nAverage Rating by Price Range:")
print(avg_rating_price)

# 2.3 Color representing highest average rating among price ranges
color_price_rating = df.groupby(['Price range', 'Rating color'])['Aggregate rating'].agg(['count', 'mean']).reset_index()
print("\nRating Color Summary per Price Range:")
print(color_price_rating)

top_color_per_price = color_price_rating.loc[color_price_rating.groupby('Price range')['mean'].idxmax()]
print("\nColor with Highest Average Rating for Each Price Range:")
print(top_color_per_price[['Price range', 'Rating color', 'mean', 'count']])

overall_highest_color = df.groupby('Rating color')['Aggregate rating'].mean().sort_values(ascending=False)
print(f"\nOverall Highest Average Rating by Color:\n{overall_highest_color}")

# Visualization for Task 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Price range frequency and average rating
sns.barplot(x='Price range', y='mean', data=avg_rating_price.reset_index(), palette='viridis', ax=axes[0], hue='Price range', legend=False)
axes[0].set_title('Average Aggregate Rating across Price Ranges', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Price Range')
axes[0].set_ylabel('Mean Aggregate Rating')
for i, row in avg_rating_price.reset_index().iterrows():
    axes[0].text(i, row['mean'] + 0.05, f"{row['mean']:.2f}", ha='center', fontweight='semibold')

# Rating color distribution vs Aggregate rating
color_map = {'Dark Green': 'darkgreen', 'Green': 'green', 'Yellow': 'gold', 'Orange': 'orange', 'White': 'lightgray', 'Red': 'red'}
sns.barplot(x='Rating color', y='Aggregate rating', data=df, order=overall_highest_color.index, palette=color_map, ax=axes[1], hue='Rating color', legend=False)
axes[1].set_title('Average Rating by Rating Color Category', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Rating Color')
axes[1].set_ylabel('Mean Rating')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('output_plots/level2_task2_price_range_analysis.png')
plt.close()
print("Saved plot: output_plots/level2_task2_price_range_analysis.png")


# -----------------------------------------------------------------------------
# TASK 3: FEATURE ENGINEERING
# -----------------------------------------------------------------------------
print("\n" + "=" * 50)
print("TASK 3: FEATURE ENGINEERING")
print("=" * 50)

# 3.1 Extract additional features
df_fe = df.copy()
df_fe['Restaurant_Name_Length'] = df_fe['Restaurant Name'].astype(str).apply(len)
df_fe['Address_Length'] = df_fe['Address'].astype(str).apply(len)
df_fe['Cuisines_Count'] = df_fe['Cuisines'].apply(lambda x: len(str(x).split(',')))

# 3.2 Encode categorical variables to binary features
binary_cols_map = {'Yes': 1, 'No': 0}
df_fe['Has_Table_Booking_Code'] = df_fe['Has Table booking'].map(binary_cols_map)
df_fe['Has_Online_Delivery_Code'] = df_fe['Has Online delivery'].map(binary_cols_map)
df_fe['Is_Delivering_Now_Code'] = df_fe['Is delivering now'].map(binary_cols_map)
df_fe['Switch_To_Order_Menu_Code'] = df_fe['Switch to order menu'].map(binary_cols_map)

print("Newly Engineered Features Preview:")
engineered_cols = [
    'Restaurant Name', 'Restaurant_Name_Length', 
    'Address_Length', 'Cuisines_Count', 
    'Has_Table_Booking_Code', 'Has_Online_Delivery_Code', 
    'Price range', 'Aggregate rating'
]
print(df_fe[engineered_cols].head())

# Correlation with target variable
correlations = df_fe[[
    'Restaurant_Name_Length', 'Address_Length', 'Cuisines_Count',
    'Has_Table_Booking_Code', 'Has_Online_Delivery_Code', 'Is_Delivering_Now_Code',
    'Price range', 'Votes', 'Average Cost for two', 'Aggregate rating'
]].corr()['Aggregate rating'].sort_values(ascending=False)

print("\nCorrelations of Engineered & Existing Features with Aggregate Rating:")
print(correlations)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df_fe[[
    'Restaurant_Name_Length', 'Address_Length', 'Cuisines_Count',
    'Has_Table_Booking_Code', 'Has_Online_Delivery_Code',
    'Price range', 'Votes', 'Aggregate rating'
]].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
plt.title('Correlation Matrix of Engineered Features with Aggregate Rating', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output_plots/level2_task3_feature_correlations.png')
plt.close()
print("Saved plot: output_plots/level2_task3_feature_correlations.png")

# Save engineered dataset for level 3
df_fe.to_csv('engineered_dataset.csv', index=False)
print("\nEngineered dataset saved to 'engineered_dataset.csv'.")
print("Level 2 analysis complete! All metrics computed and plots saved successfully.")
