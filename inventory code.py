# ============================================================
# PHASE 4 — INVENTORY OPTIMIZATION
# RetailPulse Project — Zidio Development
# ============================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
 
os.makedirs('charts', exist_ok=True)
os.makedirs('data',   exist_ok=True)
 
# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('cleaned_retail_customer_segmentation.csv')
print(f"Loaded {len(df)} customers")
 
# ============================================================
# 2. CREATE PRODUCT LIST
# ============================================================
np.random.seed(42)
 
products = [
    'Electronics', 'Clothing', 'Groceries',
    'Home & Kitchen', 'Beauty & Care', 'Sports & Fitness',
    'Books & Stationery', 'Toys & Games', 'Footwear', 'Accessories'
]
 
total_weekly_spend  = df['avg_monthly_spend'].sum() / 4.33
avg_order           = df['avg_order_value'].mean()
base_units_per_week = total_weekly_spend / avg_order
 
demand_share = np.array([0.20, 0.18, 0.15, 0.12, 0.10,
                         0.08, 0.07, 0.05, 0.03, 0.02])
 
weekly_demand = (base_units_per_week * demand_share).astype(int)
 
# ============================================================
# 3. BUILD INVENTORY TABLE
# ============================================================
inventory = pd.DataFrame({
    'Product'            : products,
    'Weekly_Demand_Units': weekly_demand,
    'Unit_Cost_Rs'       : [8500, 1200, 450, 2300, 650,
                            3200,  380, 1800, 2100,  950],
    'Lead_Time_Days'     : [7, 3, 2, 5, 3, 4, 2, 5, 4, 3],
})
 
# ============================================================
# 4. CALCULATE SAFETY STOCK & REORDER POINT
# ============================================================
inventory['Safety_Stock']  = (inventory['Weekly_Demand_Units'] * 1.5).astype(int)
inventory['Reorder_Point'] = (
    (inventory['Weekly_Demand_Units'] / 7) * inventory['Lead_Time_Days']
    + inventory['Safety_Stock']
).astype(int)
inventory['Reorder_Qty']   = (inventory['Weekly_Demand_Units'] * 4).astype(int)
 
# ============================================================
# 5. SET CURRENT STOCK — mix of low and high stock
# Some products below reorder point (need ordering)
# Some products above (healthy stock)
# ============================================================
inventory['Current_Stock'] = (
    inventory['Reorder_Point'] * np.random.uniform(0.4, 2.2, 10)
).astype(int)
 
inventory['Max_Stock'] = inventory['Current_Stock'] + inventory['Reorder_Qty']
 
# ============================================================
# 6. FLAG STATUS
# ============================================================
def stock_status(row):
    if row['Current_Stock'] <= row['Reorder_Point'] * 0.6:
        return 'CRITICAL'
    elif row['Current_Stock'] <= row['Reorder_Point']:
        return 'LOW'
    elif row['Current_Stock'] > row['Reorder_Point'] * 2.0:
        return 'OVERSTOCK'
    else:
        return 'OK'
 
inventory['Status'] = inventory.apply(stock_status, axis=1)
 
print("\nInventory Summary:")
print(inventory[['Product','Current_Stock','Reorder_Point','Status']].to_string(index=False))
print("\nStatus counts:")
print(inventory['Status'].value_counts().to_string())
 
# ============================================================
# 7. CHART 1 — CURRENT STOCK vs REORDER POINT
# ============================================================
status_colors = {
    'CRITICAL' : '#e74c3c',
    'LOW'      : '#f39c12',
    'OK'       : '#2ecc71',
    'OVERSTOCK': '#9b59b6'
}
 
bar_colors = [status_colors[s] for s in inventory['Status']]
 
fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(products))
width = 0.35
 
bars1 = ax.bar(x - width/2, inventory['Current_Stock'],
               width, color=bar_colors, edgecolor='white', label='Current Stock')
bars2 = ax.bar(x + width/2, inventory['Reorder_Point'],
               width, color='#2c3e50', edgecolor='white', alpha=0.5,
               label='Reorder Point')
 
ax.set_xticks(x)
ax.set_xticklabels(products, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Units', fontsize=12)
ax.set_title('Current Stock vs Reorder Point by Product\n'
             '🔴 Critical  🟡 Low  🟢 OK  🟣 Overstock',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('charts/inventory_stock_levels.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: charts/inventory_stock_levels.png")
 
# ============================================================
# 8. CHART 2 — STATUS PIE CHART
# ============================================================
status_counts = inventory['Status'].value_counts()
colors = [status_colors[s] for s in status_counts.index]
 
plt.figure(figsize=(7, 7))
plt.pie(status_counts.values,
        labels=status_counts.index,
        autopct='%1.0f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 13})
plt.title('Inventory Health — 10 Products',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/inventory_status_pie.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: charts/inventory_status_pie.png")
 
# ============================================================
# 9. CHART 3 — WEEKLY DEMAND RANKING
# ============================================================
sorted_inv = inventory.sort_values('Weekly_Demand_Units', ascending=True)
bar_c = [status_colors[s] for s in sorted_inv['Status']]
 
plt.figure(figsize=(10, 6))
plt.barh(sorted_inv['Product'],
         sorted_inv['Weekly_Demand_Units'],
         color=bar_c, edgecolor='white')
plt.xlabel('Weekly Demand (Units)', fontsize=12)
plt.title('Weekly Demand by Product\n'
          '🔴 Critical  🟡 Low  🟢 OK  🟣 Overstock',
          fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('charts/inventory_weekly_demand.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: charts/inventory_weekly_demand.png")
 
# ============================================================
# 10. PRINT RECOMMENDATIONS
# ============================================================
print("\n" + "="*60)
print("BUSINESS RECOMMENDATIONS")
print("="*60)
 
for status, emoji in [('CRITICAL','🔴'), ('LOW','🟡'),
                       ('OK','🟢'), ('OVERSTOCK','🟣')]:
    subset = inventory[inventory['Status'] == status]
    if len(subset) == 0:
        continue
    label = {
        'CRITICAL' : 'ORDER IMMEDIATELY',
        'LOW'      : 'ORDER SOON',
        'OK'       : 'HEALTHY — no action',
        'OVERSTOCK': 'PAUSE ORDERING'
    }[status]
    print(f"\n{emoji} {label} ({len(subset)} products):")
    for _, row in subset.iterrows():
        if status in ('CRITICAL', 'LOW'):
            cost = row['Reorder_Qty'] * row['Unit_Cost_Rs']
            print(f"   {row['Product']:<22} Stock: {row['Current_Stock']:>6,}"
                  f"  →  Order {row['Reorder_Qty']:,} units  (Rs.{cost:,.0f})")
        else:
            print(f"   {row['Product']:<22} Stock: {row['Current_Stock']:>6,}")
 
# ============================================================
# 11. SAVE
# ============================================================
inventory.to_csv('data/inventory_report.csv', index=False)
print("\n\nSaved: data/inventory_report.csv")
print("""
============================================================
PHASE 4 COMPLETE!
 
Charts: charts/inventory_stock_levels.png
        charts/inventory_status_pie.png
        charts/inventory_weekly_demand.png
Data  : data/inventory_report.csv
 
Next: Streamlit Dashboard — Phase 5!
============================================================
""")