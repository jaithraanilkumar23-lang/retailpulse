🎯 Project Overview
RetailPulse is an AI-powered retail analytics platform that helps businesses make data-driven decisions by analyzing customer behavior, predicting demand, identifying churn risk, and optimizing inventory management.
Built for: Zidio Development — Data Science & Analytics Domain
Dataset: 50,000 retail customers with behavioral features
Author: Jaithra Anil Kumar

📈 Key Results
ModelMetricResultRequirementDemand ForecastingMAPE2.7%≤ 12% ✅Churn PredictionAUC-ROC0.898≥ 0.88 ✅Customer SegmentationClusters4 segments6-8 ✅Inventory OptimizationProducts10 categories✅

🔧 Features
1. 👥 Customer Segmentation

RFM-based behavioral analysis
K-Means clustering (k=4)
4 segments: Champions, Regular, At Risk, Lost
Interactive scatter plots and pie charts

2. 📈 Demand Forecasting

Facebook Prophet time-series model
104 weeks of training data
12-week ahead forecasting
What-if analysis with growth rate slider

3. ⚠️ Churn Prediction

Gradient Boosting classifier
AUC-ROC score: 0.898
Feature importance analysis
Customer risk level classification (High/Medium/Low)

4. 📦 Inventory Optimization

Automated reorder point calculation
Safety stock recommendations
10 product categories analyzed
Critical/Low/OK/Overstock status alerts


🛠️ Technology Stack
CategoryTechnologyLanguagePython 3.11DashboardStreamlitData ProcessingPandas, NumPyMachine LearningScikit-learnForecastingProphetVisualizationMatplotlib, Seaborn, PlotlyEDAJupyter-style Python scripts

📁 Project Structure
retailpulse/
│
├── app.py                                    # Streamlit dashboard (5 pages)
├── cleaned_retail_customer_segmentation.csv  # Dataset (50,000 customers)
├── requirements.txt                          # Python dependencies
│
└── reports/
    ├── segmentation.py      # K-Means customer segmentation
    ├── forecasting.py       # Prophet demand forecasting
    ├── churn.py             # Gradient Boosting churn prediction
    ├── inventory.py         # Inventory optimization logic
    └── eda_.py              # EDA charts 

⚙️ How to Run Locally
1. Clone the repository:
bashgit clone https://github.com/jaithraanilkumar23-lang/retailpulse.git
cd retailpulse
2. Install dependencies:
bashpip install -r requirements.txt
3. Run the dashboard:
bashstreamlit run app.py
4. Open in browser:
http://localhost:8501

📊 Dashboard Pages
PageDescription🏠 OverviewKPI cards, segment summary, project status👥 Customer SegmentsPie chart, scatter plot, segment table, CSV export📈 Demand ForecastForecast chart, what-if slider, 12-week table⚠️ Churn PredictionROC curve, feature importance, risk customer list📦 InventoryStock levels, reorder alerts, recommendations

📉 EDA Highlights

50,000 customers with 14 features
Zero missing values — 100% clean dataset
Key correlation: avg_monthly_spend ↔ avg_order_value (0.64)
Age distribution: Uniform across 18–70 years
Income distribution: Right-skewed with high-value outliers


🔍 Business Insights

34.9% of customers are At Risk — need retention campaigns
15.6% are Champions — reward with loyalty programs
Books & Stationery needs immediate restocking (CRITICAL)
Clothing is overstocked — pause ordering
Top churn predictor: Purchase Frequency (63% importance)
