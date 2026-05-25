# ============================================================
# CUSTOMER SEGMENTATION — RetailPulse Project
# Just run this file. It does everything automatically.
# ============================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings('ignore')
 
# Create folders for output
os.makedirs('charts', exist_ok=True)
os.makedirs('data',   exist_ok=True)
 
# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('cleaned_retail_customer_segmentation.csv')
print(f"Loaded {len(df)} customers")
 
# ============================================================
# 2. SELECT COLUMNS FOR CLUSTERING
# ============================================================
X = df[['avg_monthly_spend', 'purchase_frequency', 
        'avg_order_value',   'months_active']].copy()
 
# ============================================================
# 3. SCALE THE DATA
# ============================================================
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling done")
 
# ============================================================
# 4. TRAIN K-MEANS WITH 4 CLUSTERS
# ============================================================
kmeans      = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("Clustering done")
 
# ============================================================
# 5. AUTO-LABEL CLUSTERS BY SPEND
#    Highest spend = Champions, Lowest = Lost
# ============================================================
spend_rank = (
    df.groupby('Cluster')['avg_monthly_spend']
    .mean()
    .rank(ascending=True)
    .astype(int)
)
label_map = {
    cluster: name
    for cluster, rank in spend_rank.items()
    for name in [['Lost', 'At Risk', 'Regular', 'Champions'][rank - 1]]
}
df['Segment'] = df['Cluster'].map(label_map)
 
print("\nCustomers per segment:")
print(df['Segment'].value_counts())
print("\nAverage spend per segment:")
print(df.groupby('Segment')['avg_monthly_spend'].mean().round(0))
 
# ============================================================
# 6. CHART 1 — PIE CHART
# ============================================================
counts = df['Segment'].value_counts()
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
 
plt.figure(figsize=(8, 8))
plt.pie(
    counts.values,
    labels=counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 13}
)
plt.title('Customer Segments\n(50,000 Customers)', fontsize=16, fontweight='bold')
plt.savefig('charts/segment_pie.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: charts/segment_pie.png")
 
# ============================================================
# 7. CHART 2 — BAR CHART (Spend per Segment)
# ============================================================
seg_avg = df.groupby('Segment')[
    ['avg_monthly_spend', 'purchase_frequency', 'avg_order_value']
].mean().round(1)
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Average Behaviour by Segment', fontsize=14, fontweight='bold')
 
seg_avg['avg_monthly_spend'].sort_values(ascending=False).plot(
    kind='bar', ax=axes[0], color='#3498db', edgecolor='white')
axes[0].set_title('Avg Monthly Spend')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=30)
 
seg_avg['purchase_frequency'].sort_values(ascending=False).plot(
    kind='bar', ax=axes[1], color='#2ecc71', edgecolor='white')
axes[1].set_title('Avg Purchase Frequency')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=30)
 
seg_avg['avg_order_value'].sort_values(ascending=False).plot(
    kind='bar', ax=axes[2], color='#e74c3c', edgecolor='white')
axes[2].set_title('Avg Order Value')
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=30)
 
plt.tight_layout()
plt.savefig('charts/segment_behaviour.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: charts/segment_behaviour.png")
 
# ============================================================
# 8. CHART 3 — SCATTER PLOT (Spend vs Frequency)
# ============================================================
color_map = {
    'Champions': '#2ecc71',
    'Regular':   '#3498db',
    'At Risk':   '#f39c12',
    'Lost':      '#e74c3c'
}
 
plt.figure(figsize=(10, 6))
for segment, color in color_map.items():
    mask = df['Segment'] == segment
    plt.scatter(
        df[mask]['purchase_frequency'],
        df[mask]['avg_monthly_spend'],
        c=color, label=segment, alpha=0.4, s=8
    )
plt.xlabel('Purchase Frequency',    fontsize=12)
plt.ylabel('Avg Monthly Spend',     fontsize=12)
plt.title('Customer Segments — Spend vs Frequency',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('charts/segment_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: charts/segment_scatter.png")
 
# ============================================================
# 9. SAVE RESULTS
# ============================================================
df.to_csv('data/customers_segmented.csv', index=False)
print("\nSaved: data/customers_segmented.csv")
 
print("""
============================================================
PHASE 1 COMPLETE!
Charts saved in charts/ folder:
  - segment_pie.png
  - segment_behaviour.png
  - segment_scatter.png
Data saved in data/ folder:
  - customers_segmented.csv
============================================================
""")