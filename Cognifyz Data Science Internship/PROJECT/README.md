# Cognifyz Technologies - Data Science Internship Project

## Project Overview
This repository contains the complete implementation of the **Cognifyz Data Science Internship Program**. 

The project explores, cleans, visualizes, engineers features, and builds predictive machine learning models on a global restaurant dataset containing over 9,500 records across 15 countries.

---

## Deliverables Summary

| Level | File Names | Description | Status |
| :--- | :--- | :--- | :--- |
| **Level 1** | [`Level_1.ipynb`](./Level_1.ipynb) & [`level_1.py`](./level_1.py) | Data Exploration & Preprocessing, Descriptive Analysis, Geospatial Analysis | **Completed** (100%) |
| **Level 2** | [`Level_2.ipynb`](./Level_2.ipynb) & [`level_2.py`](./level_2.py) | Table Booking & Delivery Analysis, Price Range Analysis, Feature Engineering | **Completed** (100%) |
| **Level 3** | [`Level_3.ipynb`](./Level_3.ipynb) & [`level_3.py`](./level_3.py) | Predictive Regression Modeling ($R^2 \approx 0.96$), Customer Preference Analysis, Visual Dashboards | **Completed** (100%) |

All generated visual charts are preserved in the [`output_plots/`](./output_plots/) directory.

---

## Detailed Task Breakdown & Key Insights

### Level 1: Data Exploration, Descriptive & Geospatial Analysis
- **Task 1: Data Exploration and Preprocessing**
  - **Dataset Size:** 9,551 rows and 21 columns.
  - **Missing Values:** 9 missing entries identified and imputed in the `Cuisines` column.
  - **Target Distribution:** 2,148 restaurants (22.49%) have an unrated score (0.0). Active ratings range from 1.8 to 4.9 with a median of 3.2.
- **Task 2: Descriptive Analysis**
  - Identified top 10 cities with New Delhi (5,473), Gurgaon (1,118), and Noida (1,080) having the highest density.
  - Top cuisines offered: North Indian (3,960), Chinese (2,735), and Fast Food (1,986).
- **Task 3: Geospatial Analysis**
  - Global latitude and longitude mappings show extreme concentration in India (Country Code 1) alongside international clusters in the US, UAE, UK, Australia, and the Philippines.
  - Weak direct linear correlation between raw coordinates and ratings ($r \approx -0.11$), indicating local factors drive restaurant quality rather than geographic longitude.

---

### Level 2: Table Booking, Price Range & Feature Engineering
- **Task 1: Table Booking and Online Delivery**
  - **Table Booking:** Offered by **12.12%** of restaurants; **87.88%** do not offer table booking.
  - **Online Delivery:** Offered by **25.66%** of restaurants; **74.34%** do not offer online delivery.
  - **Impact on Rating:** Restaurants with table booking have a significantly higher average rating (**3.44 vs. 2.56**, Welch's $t = 28.51, p < 10^{-150}$).
- **Task 2: Price Range Analysis**
  - **Most Common Price Range:** Tier 1 (budget-friendly) represents **46.53%** (4,444 restaurants).
  - Higher price tiers correlate with higher ratings: Tier 1 (2.00) $\rightarrow$ Tier 2 (2.94) $\rightarrow$ Tier 3 (3.68) $\rightarrow$ Tier 4 (3.82).
  - **Dark Green** represents the highest average rating ($> 4.60$) consistently across all price ranges.
- **Task 3: Feature Engineering**
  - Extracted new features: `Restaurant_Name_Length`, `Address_Length`, and `Cuisines_Count`.
  - Engineered binary flags: `Has_Table_Booking_Code`, `Has_Online_Delivery_Code`, `Is_Delivering_Now_Code`.
  - Saved enriched dataset to [`engineered_dataset.csv`](./engineered_dataset.csv).

---

### Level 3: Predictive Modeling, Customer Preference & Visualization
- **Task 1: Predictive Modeling (Regression for Aggregate Rating)**
  - Benchmarked multiple machine learning algorithms with an 80/20 train/test split:
    - **Gradient Boosting Regressor:** **$R^2 = 0.9597$**, $\text{RMSE} = 0.3030$, $\text{MAE} = 0.2019$ (Best Model)
    - **Random Forest Regressor:** **$R^2 = 0.9592$**, $\text{RMSE} = 0.3049$, $\text{MAE} = 0.2006$
    - **Decision Tree Regressor:** **$R^2 = 0.9542$**, $\text{RMSE} = 0.3227$, $\text{MAE} = 0.2136$
    - **Ridge Regression:** **$R^2 = 0.3553$**, $\text{RMSE} = 1.2114$, $\text{MAE} = 0.9963$
    - **Linear Regression:** **$R^2 = 0.3551$**, $\text{RMSE} = 1.2116$, $\text{MAE} = 0.9970$
- **Task 2: Customer Preference Analysis**
  - **Highest Volume (Votes):** North Indian (595,981 votes), Chinese (364,351 votes), Italian (329,265 votes).
  - **Highest Average Ratings:** International (4.25), Southern (4.13), Vegetarian (4.07), Sandwich (4.07), Grill (4.06).
- **Task 3: Comprehensive Visualizations**
  - Generated multi-panel visual dashboard displaying rating density, city rating benchmarks, customer vote correlations, and feature covariance.

---

## How to Run the Project

### 1. Requirements
Ensure standard Data Science libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 2. Run Scripts
You can run each level independently:
```bash
# Run Level 1
python level_1.py

# Run Level 2
python level_2.py

# Run Level 3
python level_3.py
```

### 3. Open in Jupyter Notebook
Open the interactive notebooks:
```bash
jupyter notebook Level_1.ipynb
jupyter notebook Level_2.ipynb
jupyter notebook Level_3.ipynb
```
