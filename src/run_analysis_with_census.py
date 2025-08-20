"""
Complete Geographic Policing Intensity Analysis with Actual Census Data
Following methodology_guide.md - Using real census populations
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

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Set up paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'data'
FIGURES_PATH = BASE_PATH / 'figures'
RESULTS_PATH = BASE_PATH / 'results'
FIGURES_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)

print("="*60)
print("GEOGRAPHIC POLICING INTENSITY ANALYSIS")
print("Using Actual Census Data")
print("="*60)

# Load arrest data
print("\nLoading arrest data...")
arrests = pd.read_parquet(DATA_PATH / 'census_mapped_anon_data.parquet')
print(f"Arrests loaded: {len(arrests):,}")
print(f"Unique individuals: {arrests['DefendantId'].nunique():,}")

# Load census data with actual populations
print("\nLoading census block group data...")
census_bg = pd.read_parquet(DATA_PATH / 'census_blockgroup_data.parquet')
print(f"Block groups with census data: {len(census_bg):,}")
print(f"Total population: {census_bg['total_pop'].sum():,}")

# Calculate years of data
years_of_data = (arrests['ArrestDate'].max() - arrests['ArrestDate'].min()).days / 365.25
print(f"Years of data: {years_of_data:.1f}")

# Identify discretionary arrests
print("\n" + "="*60)
print("PHASE 1: CATEGORIZATION USING DISCRETIONARY ARRESTS")
print("="*60)

discretionary_categories = [
    'Drug Poss',
    'Property', 
    'Traffic',
    'Other Offenses',
    'Theft'
]

arrests['is_discretionary'] = arrests['Arrest_crime_category'].isin(discretionary_categories)
arrests['blockgroup_id'] = arrests['DefendantAddressGEOID10'].astype(str).str[:12]

print(f"\nDiscretionary arrests: {arrests['is_discretionary'].sum():,} ({arrests['is_discretionary'].mean()*100:.1f}%)")

# Aggregate arrests by block group
bg_arrests = arrests.groupby('blockgroup_id').agg({
    'DefendantId': ['count', 'nunique'],
    'is_discretionary': 'sum'
}).reset_index()

bg_arrests.columns = ['blockgroup_id', 'total_arrests', 'unique_individuals', 'discretionary_arrests']

# Merge with census data
print("\nMerging with census populations...")
bg_data = census_bg[['blockgroup_geoid_str', 'total_pop', 'white_pop', 'black_pop', 
                     'hispanic_pop', 'median_household_income', 'poverty_rate']].copy()
bg_data.rename(columns={'blockgroup_geoid_str': 'blockgroup_id'}, inplace=True)

# Merge arrests with census
bg_combined = bg_data.merge(bg_arrests, on='blockgroup_id', how='inner')
print(f"Block groups with both census and arrest data: {len(bg_combined):,}")

# Calculate rates per 1,000 using ACTUAL population
bg_combined['discretionary_per_1000'] = (bg_combined['discretionary_arrests'] / bg_combined['total_pop']) * 1000
bg_combined['total_per_1000'] = (bg_combined['total_arrests'] / bg_combined['total_pop']) * 1000
bg_combined['unique_per_1000'] = (bg_combined['unique_individuals'] / bg_combined['total_pop']) * 1000

# Remove any infinite values (where pop = 0)
bg_combined = bg_combined[bg_combined['total_pop'] > 0]
bg_combined = bg_combined.replace([np.inf, -np.inf], np.nan).dropna(subset=['discretionary_per_1000'])

print(f"Final block groups for analysis: {len(bg_combined):,}")
print(f"Population range: {bg_combined['total_pop'].min():,} to {bg_combined['total_pop'].max():,}")
print(f"Discretionary rate range: {bg_combined['discretionary_per_1000'].min():.1f} to {bg_combined['discretionary_per_1000'].max():.1f} per 1,000")

# Sort by discretionary rate and calculate cumulative population
bg_combined = bg_combined.sort_values('discretionary_per_1000', ascending=False).reset_index(drop=True)
bg_combined['cumulative_pop'] = bg_combined['total_pop'].cumsum()
bg_combined['cumulative_pop_pct'] = bg_combined['cumulative_pop'] / bg_combined['total_pop'].sum() * 100

# Identify cut points using cumulative population
print("\n" + "="*60)
print("IDENTIFYING CUT POINTS")
print("="*60)

# Target: ~6-7% ultra-policed, ~15-16% highly policed
cut1_idx = np.argmax(bg_combined['cumulative_pop_pct'] >= 6.6)
cut2_idx = np.argmax(bg_combined['cumulative_pop_pct'] >= 22.0)

cut1_rate = bg_combined.iloc[cut1_idx]['discretionary_per_1000']
cut2_rate = bg_combined.iloc[cut2_idx]['discretionary_per_1000']

print(f"Cut point 1: {cut1_rate:.1f} per 1,000 (top {bg_combined.iloc[cut1_idx]['cumulative_pop_pct']:.1f}%)")
print(f"Cut point 2: {cut2_rate:.1f} per 1,000 (top {bg_combined.iloc[cut2_idx]['cumulative_pop_pct']:.1f}%)")

# Categorize block groups
def categorize_policing(rate):
    if rate >= cut1_rate:
        return 'Ultra-Policed'
    elif rate >= cut2_rate:
        return 'Highly Policed'
    else:
        return 'Normally Policed'

bg_combined['policing_category'] = bg_combined['discretionary_per_1000'].apply(categorize_policing)

# Calculate category statistics with ACTUAL populations
category_stats = bg_combined.groupby('policing_category').agg({
    'total_pop': 'sum',
    'total_arrests': 'sum',
    'discretionary_arrests': 'sum',
    'unique_individuals': 'sum',
    'blockgroup_id': 'count'
}).rename(columns={'blockgroup_id': 'num_blockgroups'})

category_stats['pop_pct'] = category_stats['total_pop'] / category_stats['total_pop'].sum() * 100
category_stats['disc_per_1000'] = (category_stats['discretionary_arrests'] / category_stats['total_pop']) * 1000
category_stats['total_per_1000'] = (category_stats['total_arrests'] / category_stats['total_pop']) * 1000
category_stats['unique_per_1000'] = (category_stats['unique_individuals'] / category_stats['total_pop']) * 1000

print("\n" + "="*60)
print("POLICING INTENSITY CATEGORIES (WITH CENSUS DATA)")
print("="*60)

for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
    if cat in category_stats.index:
        stats = category_stats.loc[cat]
        print(f"\n{cat}:")
        print(f"  Block groups: {stats['num_blockgroups']:.0f}")
        print(f"  Actual population: {stats['total_pop']:,.0f} ({stats['pop_pct']:.1f}%)")
        print(f"  Unique individuals arrested: {stats['unique_individuals']:,.0f}")
        print(f"  Discretionary per 1,000: {stats['disc_per_1000']:.1f}")
        print(f"  Total arrests per 1,000: {stats['total_per_1000']:.1f}")
        print(f"  Unique individuals per 1,000: {stats['unique_per_1000']:.1f}")

# Merge category back to arrests for individual-level analysis
arrests_with_cat = arrests.merge(
    bg_combined[['blockgroup_id', 'policing_category', 'total_pop']],
    on='blockgroup_id',
    how='inner'
)

print("\n" + "="*60)
print("PHASE 2: ANNUAL ARREST RISKS (USING CENSUS POPULATIONS)")
print("="*60)

# Overall population annual risk
print("\nOverall Population Annual Risk:")
risk_results = []

for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
    if cat in category_stats.index:
        unique_individuals = category_stats.loc[cat, 'unique_individuals']
        population = category_stats.loc[cat, 'total_pop']
        
        annual_unique = unique_individuals / years_of_data
        annual_risk = (annual_unique / population) * 100
        
        print(f"\n{cat}:")
        print(f"  Population: {population:,.0f}")
        print(f"  Unique individuals: {unique_individuals:,.0f}")
        print(f"  Annual risk: {annual_risk:.2f}% (1 in {100/annual_risk:.0f})")
        
        risk_results.append({
            'Category': cat,
            'Population': population,
            'Annual_Risk_Pct': annual_risk
        })

# Young men (18-35)
print("\nYoung Men (18-35) Annual Risk:")
young_men = arrests_with_cat[
    (arrests_with_cat['Age_years'].between(18, 35)) & 
    (arrests_with_cat['Gender'] == 'Male')
]

young_men_risks = []
for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
    if cat in category_stats.index:
        cat_young_men = young_men[young_men['policing_category'] == cat]
        unique_young_men = cat_young_men['DefendantId'].nunique()
        
        # Estimate young male population as ~20% of total
        est_young_male_pop = category_stats.loc[cat, 'total_pop'] * 0.20
        
        annual_unique = unique_young_men / years_of_data
        annual_risk = (annual_unique / est_young_male_pop) * 100
        
        print(f"\n{cat}:")
        print(f"  Est. young male population: {est_young_male_pop:,.0f}")
        print(f"  Unique young men arrested: {unique_young_men:,}")
        print(f"  Annual risk: {annual_risk:.2f}% (1 in {100/annual_risk:.0f})")
        
        young_men_risks.append({
            'Category': cat,
            'Annual_Risk_Pct': annual_risk
        })

# Calculate lifetime risks
print("\nLifetime Arrest Probability (Young Men by Age 35):")
for risk_data in young_men_risks:
    cat = risk_data['Category']
    annual_risk = risk_data['Annual_Risk_Pct'] / 100
    by_35 = 1 - (1 - annual_risk) ** 17
    print(f"  {cat}: {by_35*100:.1f}%")

# Calculate disparities
print("\n" + "="*60)
print("KEY DISPARITIES")
print("="*60)

ultra_overall = risk_results[0]['Annual_Risk_Pct']
normal_overall = risk_results[2]['Annual_Risk_Pct']
overall_ratio = ultra_overall / normal_overall if normal_overall > 0 else 0

ultra_young = young_men_risks[0]['Annual_Risk_Pct']
normal_young = young_men_risks[2]['Annual_Risk_Pct']
young_ratio = ultra_young / normal_young if normal_young > 0 else 0

print(f"\nOverall population:")
print(f"  Ultra-Policed: {ultra_overall:.2f}% annual risk")
print(f"  Normally Policed: {normal_overall:.2f}% annual risk")
print(f"  Disparity: {overall_ratio:.1f}x")

print(f"\nYoung men (18-35):")
print(f"  Ultra-Policed: {ultra_young:.2f}% annual risk")
print(f"  Normally Policed: {normal_young:.2f}% annual risk")
print(f"  Disparity: {young_ratio:.1f}x")

# Drug offense analysis
print("\n" + "="*60)
print("PHASE 5: DRUG OFFENSE ANALYSIS")
print("="*60)

drug_arrests = arrests_with_cat[arrests_with_cat['Arrest_crime_category'].str.contains('Drug', na=False)]
print(f"\nTotal drug arrests: {len(drug_arrests):,}")
print(f"Unique individuals with drug arrests: {drug_arrests['DefendantId'].nunique():,}")

# Drug arrests by category
drug_by_cat = drug_arrests.groupby('policing_category')['DefendantId'].nunique()
print("\nUnique individuals with drug arrests by category:")
for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
    if cat in drug_by_cat.index and cat in category_stats.index:
        unique_drug = drug_by_cat[cat]
        population = category_stats.loc[cat, 'total_pop']
        annual_per_1000 = (unique_drug / years_of_data) / population * 1000
        print(f"  {cat}: {annual_per_1000:.2f} per 1,000 annually")

# Drug repeat patterns
drug_repeats = drug_arrests.groupby('DefendantId').size()
facing_enhancement = (drug_repeats >= 2).sum()
facing_mandatory = (drug_repeats >= 3).sum()
total_drug_people = len(drug_repeats)

print(f"\nDrug offense escalation:")
print(f"  People with drug arrests: {total_drug_people:,}")
print(f"  Facing enhanced penalties (2+): {facing_enhancement:,} ({facing_enhancement/total_drug_people*100:.1f}%)")
print(f"  Facing mandatory minimums (3+): {facing_mandatory:,} ({facing_mandatory/total_drug_people*100:.1f}%)")

# Create visualization
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Distribution of discretionary rates
ax1 = axes[0, 0]
ax1.hist(bg_combined['discretionary_per_1000'], bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(cut1_rate, color='red', linestyle='--', label=f'Cut 1: {cut1_rate:.1f}')
ax1.axvline(cut2_rate, color='orange', linestyle='--', label=f'Cut 2: {cut2_rate:.1f}')
ax1.set_xlabel('Discretionary Arrests per 1,000')
ax1.set_ylabel('Block Groups')
ax1.set_title('Distribution with Census Populations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Arrest rates by category
ax2 = axes[0, 1]
categories = ['Ultra-Policed', 'Highly Policed', 'Normally Policed']
disc_rates = [category_stats.loc[cat, 'disc_per_1000'] for cat in categories]
total_rates = [category_stats.loc[cat, 'total_per_1000'] for cat in categories]

x = np.arange(len(categories))
width = 0.35
ax2.bar(x - width/2, disc_rates, width, label='Discretionary', color='steelblue')
ax2.bar(x + width/2, total_rates, width, label='Total', color='darkred')
ax2.set_xlabel('Category')
ax2.set_ylabel('Per 1,000 Population')
ax2.set_title('Arrest Rates (Census-Based)')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=15)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Annual risk comparison
ax3 = axes[0, 2]
overall_risks = [r['Annual_Risk_Pct'] for r in risk_results]
young_risks = [r['Annual_Risk_Pct'] for r in young_men_risks]

x = np.arange(len(categories))
width = 0.35
bars1 = ax3.bar(x - width/2, overall_risks, width, label='Overall', color='lightblue')
bars2 = ax3.bar(x + width/2, young_risks, width, label='Young Men', color='navy')

ax3.set_xlabel('Category')
ax3.set_ylabel('Annual Risk (%)')
ax3.set_title('Annual Arrest Risk')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, rotation=15)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 4. Population distribution
ax4 = axes[1, 0]
sizes = [category_stats.loc[cat, 'pop_pct'] for cat in categories]
colors = ['darkred', 'orange', 'lightgreen']
ax4.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Population Distribution')

# 5. Disparity visualization
ax5 = axes[1, 1]
disparities = [overall_ratio, young_ratio]
labels = ['Overall\nPopulation', 'Young Men\n(18-35)']
bars = ax5.bar(labels, disparities, color=['steelblue', 'navy'])
ax5.set_ylabel('Disparity Ratio (Ultra vs Normal)')
ax5.set_title('Arrest Risk Disparities')
ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax5.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, disparities):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}x', ha='center', va='bottom', fontweight='bold')

# 6. Summary text
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""KEY FINDINGS (WITH CENSUS DATA)

Population Distribution:
• Ultra-Policed: {sizes[0]:.1f}%
• Highly Policed: {sizes[1]:.1f}%
• Normally Policed: {sizes[2]:.1f}%

Annual Arrest Risk:
• Ultra: {ultra_overall:.2f}%
• Normal: {normal_overall:.2f}%
• Disparity: {overall_ratio:.1f}x

Young Men (18-35):
• Ultra: {ultra_young:.2f}%
• Normal: {normal_young:.2f}%
• Disparity: {young_ratio:.1f}x

Total Population: {bg_combined['total_pop'].sum():,}
Block Groups: {len(bg_combined):,}"""

ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Geographic Policing Analysis with Actual Census Data', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = FIGURES_PATH / 'analysis_with_census.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved visualization to {output_path}")

# Save updated results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

bg_combined.to_csv(RESULTS_PATH / 'blockgroups_with_census.csv', index=False)
category_stats.to_csv(RESULTS_PATH / 'category_stats_census.csv')
pd.DataFrame(risk_results).to_csv(RESULTS_PATH / 'annual_risks_census.csv', index=False)

# Create updated summary report
report = f"""# Geographic Policing Intensity Analysis - Updated with Census Data

## Executive Summary
Analysis using actual census population data reveals significant disparities in policing intensity across neighborhoods.

## Key Findings

### Population Distribution (Census-Based)
- **Ultra-Policed**: {category_stats.loc['Ultra-Policed', 'pop_pct']:.1f}% of population ({category_stats.loc['Ultra-Policed', 'total_pop']:,.0f} people)
- **Highly Policed**: {category_stats.loc['Highly Policed', 'pop_pct']:.1f}% of population ({category_stats.loc['Highly Policed', 'total_pop']:,.0f} people)
- **Normally Policed**: {category_stats.loc['Normally Policed', 'pop_pct']:.1f}% of population ({category_stats.loc['Normally Policed', 'total_pop']:,.0f} people)

### Arrest Rates per 1,000 Population
- **Ultra-Policed**: {category_stats.loc['Ultra-Policed', 'total_per_1000']:.1f} total, {category_stats.loc['Ultra-Policed', 'disc_per_1000']:.1f} discretionary
- **Highly Policed**: {category_stats.loc['Highly Policed', 'total_per_1000']:.1f} total, {category_stats.loc['Highly Policed', 'disc_per_1000']:.1f} discretionary
- **Normally Policed**: {category_stats.loc['Normally Policed', 'total_per_1000']:.1f} total, {category_stats.loc['Normally Policed', 'disc_per_1000']:.1f} discretionary

### Annual Arrest Risk
**Overall Population:**
- Ultra-Policed: {ultra_overall:.2f}% (1 in {100/ultra_overall:.0f})
- Normally Policed: {normal_overall:.2f}% (1 in {100/normal_overall:.0f})
- **Disparity: {overall_ratio:.1f}x**

**Young Men (18-35):**
- Ultra-Policed: {ultra_young:.2f}% (1 in {100/ultra_young:.0f})
- Normally Policed: {normal_young:.2f}% (1 in {100/normal_young:.0f})
- **Disparity: {young_ratio:.1f}x**

### Drug Enforcement
- Total drug arrests: {len(drug_arrests):,}
- Unique individuals: {drug_arrests['DefendantId'].nunique():,}
- Facing enhanced penalties (2+ arrests): {facing_enhancement/total_drug_people*100:.1f}%
- Facing mandatory minimums (3+ arrests): {facing_mandatory/total_drug_people*100:.1f}%

## Methodology
- Used actual census block group populations from American Community Survey
- Identified policing intensity using discretionary arrest rates
- Calculated per capita risks using unique individuals (not total arrests)
- Time period: {years_of_data:.1f} years of data

## Data Sources
- Census data: {len(bg_combined):,} block groups
- Total population: {bg_combined['total_pop'].sum():,}
- Total arrests: {len(arrests):,}
- Unique individuals: {arrests['DefendantId'].nunique():,}

---
*Analysis completed using actual census population data*
"""

with open(RESULTS_PATH / 'summary_report_census.md', 'w') as f:
    f.write(report)

print(f"Updated summary report saved to {RESULTS_PATH / 'summary_report_census.md'}")
print("\nAnalysis complete with actual census data!")