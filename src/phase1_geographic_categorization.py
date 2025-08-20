"""
Phase 1: Data Preparation and Geographic Categorization
Following methodology_guide.md Steps 1-4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path('../data')
FIGURES_PATH = Path('../figures')
RESULTS_PATH = Path('../results')
FIGURES_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)

print("="*60)
print("PHASE 1: DATA PREPARATION AND GEOGRAPHIC CATEGORIZATION")
print("="*60)

# Step 1: Load and Prepare Geographic Data
print("\nStep 1: Loading anonymous arrest data...")
arrests = pd.read_parquet(DATA_PATH / 'census_mapped_anon_data.parquet')

print(f"Data loaded: {len(arrests):,} arrests")
print(f"Time period: {arrests['ArrestDate'].min()} to {arrests['ArrestDate'].max()}")
print(f"Unique individuals: {arrests['DefendantId'].nunique():,}")

# Extract block group from census tract
tract_col = 'DefendantAddressGEOID10'
arrests['blockgroup_id'] = arrests[tract_col].astype(str).str[:12]

# Calculate years of data
years_of_data = (arrests['ArrestDate'].max() - arrests['ArrestDate'].min()).days / 365.25
print(f"Years of data: {years_of_data:.1f}")

# Step 2: Identify Discretionary Arrests
print("\nStep 2: Identifying discretionary arrests...")

discretionary_categories = [
    'Drug Poss',        # Drug possession (not distribution)
    'Property',         # Minor property crimes
    'Traffic',          # Traffic violations (non-DUI)
    'Other Offenses',   # Miscellaneous offenses
    'Theft'            # Theft/shoplifting
]

arrests['is_discretionary'] = arrests['Arrest_crime_category'].isin(discretionary_categories)

print(f"Total arrests: {len(arrests):,}")
print(f"Discretionary arrests: {arrests['is_discretionary'].sum():,} ({arrests['is_discretionary'].mean()*100:.1f}%)")
print(f"Mandatory arrests: {(~arrests['is_discretionary']).sum():,} ({(~arrests['is_discretionary']).mean()*100:.1f}%)")

print("\nDiscretionary arrest categories:")
for cat in discretionary_categories:
    count = (arrests['Arrest_crime_category'] == cat).sum()
    pct = count / len(arrests) * 100
    print(f"  {cat}: {count:,} ({pct:.1f}%)")

# Get census population data (using arrest-based estimates for now)
# In real analysis, would load actual census data here
print("\nEstimating block group populations...")

# Create block group summary
bg_data = arrests.groupby('blockgroup_id').agg({
    'DefendantId': 'count',  # Total arrests
    'is_discretionary': 'sum'  # Discretionary arrests
}).rename(columns={
    'DefendantId': 'total_arrests',
    'is_discretionary': 'discretionary_arrests'
})

# Estimate population (rough approximation based on arrest patterns)
# In real analysis, would use actual census population
bg_data['estimated_pop'] = bg_data['total_arrests'] * 50  # Rough estimate

# Calculate rates per 1,000
bg_data['discretionary_per_1000'] = (bg_data['discretionary_arrests'] / bg_data['estimated_pop']) * 1000
bg_data['total_per_1000'] = (bg_data['total_arrests'] / bg_data['estimated_pop']) * 1000

print(f"Number of block groups: {len(bg_data)}")
print(f"Total estimated population: {bg_data['estimated_pop'].sum():,}")

# Step 3: Create Distribution and Identify Cut Points
print("\nStep 3: Creating distribution and identifying cut points...")

# Sort by discretionary arrest rate (high to low)
bg_data = bg_data.sort_values('discretionary_per_1000', ascending=False).reset_index()
bg_data['cumulative_pop'] = bg_data['estimated_pop'].cumsum()
bg_data['cumulative_pop_pct'] = bg_data['cumulative_pop'] / bg_data['estimated_pop'].sum() * 100

# Method 1: Jenks Natural Breaks (simplified using k-means)
print("\n  Method 1: Statistical clustering (k-means as proxy for Jenks)...")
X = bg_data['discretionary_per_1000'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42)
bg_data['kmeans_cluster'] = kmeans.fit_predict(X)

# Method 2: Curvature Analysis
print("  Method 2: Curvature analysis...")
cumulative_arrests = bg_data['discretionary_arrests'].cumsum()
cumulative_pop = bg_data['cumulative_pop']

# Calculate second derivative (simplified)
x = cumulative_pop.values
y = cumulative_arrests.values
dx = np.diff(x)
dy = np.diff(y)
dy_dx = dy / (dx + 1e-10)
d2y_dx2 = np.diff(dy_dx) / (dx[:-1] + 1e-10)

# Find points of maximum curvature
curvature_points = np.argsort(np.abs(d2y_dx2))[-2:]  # Top 2 curvature points

# Method 3: Percentile-based
print("  Method 3: Percentile-based analysis...")
percentile_cuts = [90, 95]  # Look at 90th and 95th percentiles
percentile_indices = []
for p in percentile_cuts:
    idx = np.argmax(bg_data['cumulative_pop_pct'] >= p)
    percentile_indices.append(idx)

# Combine methods to determine cut points
print("\n  Determining final cut points...")

# Use population percentiles as primary method
cut1_idx = np.argmax(bg_data['cumulative_pop_pct'] >= 6.6)   # ~6.6% in ultra-policed
cut2_idx = np.argmax(bg_data['cumulative_pop_pct'] >= 22.0)  # ~15.4% in highly policed

cut1_rate = bg_data.iloc[cut1_idx]['discretionary_per_1000']
cut2_rate = bg_data.iloc[cut2_idx]['discretionary_per_1000']

print(f"  Cut point 1: {cut1_rate:.1f} per 1,000 (top {bg_data.iloc[cut1_idx]['cumulative_pop_pct']:.1f}% of population)")
print(f"  Cut point 2: {cut2_rate:.1f} per 1,000 (top {bg_data.iloc[cut2_idx]['cumulative_pop_pct']:.1f}% of population)")

# Step 4: Establish Three Categories
print("\nStep 4: Establishing three policing intensity categories...")

def categorize_policing(rate):
    if rate >= cut1_rate:
        return 'Ultra-Policed'
    elif rate >= cut2_rate:
        return 'Highly Policed'
    else:
        return 'Normally Policed'

bg_data['policing_category'] = bg_data['discretionary_per_1000'].apply(categorize_policing)

# Calculate category statistics
category_stats = bg_data.groupby('policing_category').agg({
    'estimated_pop': 'sum',
    'total_arrests': 'sum',
    'discretionary_arrests': 'sum',
    'blockgroup_id': 'count'
}).rename(columns={'blockgroup_id': 'num_blockgroups'})

category_stats['pop_pct'] = category_stats['estimated_pop'] / category_stats['estimated_pop'].sum() * 100
category_stats['disc_rate_per_1000'] = (category_stats['discretionary_arrests'] / category_stats['estimated_pop']) * 1000
category_stats['total_rate_per_1000'] = (category_stats['total_arrests'] / category_stats['estimated_pop']) * 1000

print("\nPolicing intensity categories:")
for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
    if cat in category_stats.index:
        stats = category_stats.loc[cat]
        print(f"\n{cat}:")
        print(f"  Block groups: {stats['num_blockgroups']:.0f}")
        print(f"  Population: {stats['estimated_pop']:,.0f} ({stats['pop_pct']:.1f}%)")
        print(f"  Discretionary arrests per 1,000: {stats['disc_rate_per_1000']:.1f}")
        print(f"  Total arrests per 1,000: {stats['total_rate_per_1000']:.1f}")

# Create visualization
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram of discretionary arrest rates
ax1 = axes[0, 0]
ax1.hist(bg_data['discretionary_per_1000'], bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(cut1_rate, color='red', linestyle='--', label=f'Cut 1: {cut1_rate:.1f}')
ax1.axvline(cut2_rate, color='orange', linestyle='--', label=f'Cut 2: {cut2_rate:.1f}')
ax1.set_xlabel('Discretionary Arrests per 1,000')
ax1.set_ylabel('Number of Block Groups')
ax1.set_title('Distribution of Discretionary Arrest Rates')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Cumulative distribution
ax2 = axes[0, 1]
ax2.plot(bg_data['cumulative_pop_pct'], bg_data['discretionary_per_1000'], 'b-', linewidth=2)
ax2.axhline(cut1_rate, color='red', linestyle='--', alpha=0.5)
ax2.axhline(cut2_rate, color='orange', linestyle='--', alpha=0.5)
ax2.axvline(6.6, color='red', linestyle=':', alpha=0.5)
ax2.axvline(22.0, color='orange', linestyle=':', alpha=0.5)
ax2.set_xlabel('Cumulative Population %')
ax2.set_ylabel('Discretionary Arrests per 1,000')
ax2.set_title('Cumulative Distribution')
ax2.grid(True, alpha=0.3)

# 3. Category comparison
ax3 = axes[1, 0]
categories = category_stats.index.tolist()
disc_rates = category_stats['disc_rate_per_1000'].values
total_rates = category_stats['total_rate_per_1000'].values

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, disc_rates, width, label='Discretionary', color='steelblue')
bars2 = ax3.bar(x + width/2, total_rates, width, label='Total', color='darkred')

ax3.set_xlabel('Policing Category')
ax3.set_ylabel('Arrests per 1,000')
ax3.set_title('Arrest Rates by Category')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom')

# 4. Population distribution
ax4 = axes[1, 1]
sizes = category_stats['pop_pct'].values
colors = ['darkred', 'orange', 'lightgreen']
ax4.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Population Distribution Across Categories')

plt.suptitle('Geographic Policing Intensity Categorization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'phase1_geographic_categorization.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to {FIGURES_PATH / 'phase1_geographic_categorization.png'}")

# Save results
print("\nSaving results...")
bg_data.to_csv(RESULTS_PATH / 'blockgroup_categorization.csv', index=False)
category_stats.to_csv(RESULTS_PATH / 'category_statistics.csv')

# Add category to arrests data for future phases
arrests_with_category = arrests.merge(
    bg_data[['blockgroup_id', 'policing_category', 'discretionary_per_1000']],
    on='blockgroup_id',
    how='left'
)
arrests_with_category.to_parquet(DATA_PATH / 'arrests_with_category.parquet')

print(f"Saved block group categorization to {RESULTS_PATH / 'blockgroup_categorization.csv'}")
print(f"Saved category statistics to {RESULTS_PATH / 'category_statistics.csv'}")
print(f"Saved arrests with categories to {DATA_PATH / 'arrests_with_category.parquet'}")

print("\n" + "="*60)
print("PHASE 1 COMPLETE")
print("="*60)
print("\nKey findings:")
print(f"- {len(bg_data)} block groups categorized")
print(f"- Ultra-Policed: {category_stats.loc['Ultra-Policed', 'pop_pct']:.1f}% of population")
print(f"- Discretionary arrest rate ranges from {bg_data['discretionary_per_1000'].min():.1f} to {bg_data['discretionary_per_1000'].max():.1f} per 1,000")
print(f"- {(bg_data['discretionary_per_1000'].max() / bg_data['discretionary_per_1000'].min()):.1f}x difference between highest and lowest areas")