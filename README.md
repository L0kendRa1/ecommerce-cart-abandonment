# 🛒 E-Commerce Cart Abandonment & Prediction Engine

## 📌 The Business Problem
In e-commerce operations, such as managing a live storefront like Urbanora, shopping cart abandonment is a primary driver of lost revenue. This project analyzes a dataset of 500,000+ raw user event logs to identify exactly where users drop off in the purchasing funnel and builds a predictive Machine Learning model to identify high-risk sessions before the customer leaves the site.

## 🛠️ Tech Stack
* **Data Engineering & Cleaning:** Python (Pandas)
* **Funnel Analysis:** SQL (DuckDB)
* **Predictive Modeling:** Python (XGBoost, Scikit-Learn)
* **Data Visualization:** Power BI

## 🚀 Methodology
1. **ETL Pipeline:** Extracted raw event logs, parsed text-based timestamps, dropped invalid sessions, and engineered time-based features (e.g., session duration, hour of day).
2. **SQL Funnel Analysis:** Utilized Common Table Expressions (CTEs) and conditional aggregation in DuckDB to calculate absolute drop-off rates at every stage of the checkout process.
3. **Machine Learning:** Filtered data to active cart sessions and trained an XGBoost classification model to predict abandonment. Extracted Feature Importance to identify behavioral drivers.

## 📊 Key Executive Insights
* **The Bottleneck:** The primary leak in the funnel occurs at the 'Add to Cart' stage, with a significant drop-off before users complete their purchase.
* **Predictive Drivers:** Machine Learning analysis revealed that **user engagement volume (Total Events)** is the primary indicator of purchase intent (contributing 55.18% to the model).
* **Secondary Factors:** Session Duration accounts for 24.03% of abandonment risk, while macro-temporal factors (like weekends vs. weekdays) showed 0.00% impact, proving that abandonment is driven by in-session friction, not the day of the week.

## 📂 Repository Structure
* `df.py`: The Python/SQL script used to clean timestamps and calculate the conversion funnel.
* `ml_model.py`: The XGBoost classification model and feature importance extractor.
* `Dashboard.pdf`: The final Power BI executive summary.
