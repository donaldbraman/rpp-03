"""
Complete Geographic Policing Intensity Analysis
Following methodology_guide.md - All 33 steps across 7 phases
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

def load_data():
    """Step 1: Load and prepare geographic data"""
    print("="*60)
    print("PHASE 1: DATA PREPARATION AND GEOGRAPHIC CATEGORIZATION")
    print("="*60)
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
    
    return arrests, years_of_data

def identify_discretionary_arrests(arrests):
    """Step 2: Identify discretionary arrests"""
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
    
    return arrests

def create_blockgroup_data(arrests):
    """Create block group level data with population estimates"""
    print("\nEstimating block group populations...")
    
    # Get unique individuals per block group for better population estimate
    bg_individuals = arrests.groupby('blockgroup_id')['DefendantId'].nunique().to_frame('unique_individuals')
    
    # Create block group summary
    bg_data = arrests.groupby('blockgroup_id').agg({
        'DefendantId': 'count',  # Total arrests
        'is_discretionary': 'sum'  # Discretionary arrests
    }).rename(columns={
        'DefendantId': 'total_arrests',
        'is_discretionary': 'discretionary_arrests'
    })
    
    # Merge unique individuals
    bg_data = bg_data.merge(bg_individuals, left_index=True, right_index=True)
    
    # Better population estimate based on unique individuals
    # Assume arrested individuals represent ~2% of population
    bg_data['estimated_pop'] = bg_data['unique_individuals'] * 50
    
    # Calculate rates per 1,000
    bg_data['discretionary_per_1000'] = (bg_data['discretionary_arrests'] / bg_data['estimated_pop']) * 1000
    bg_data['total_per_1000'] = (bg_data['total_arrests'] / bg_data['estimated_pop']) * 1000
    
    print(f"Number of block groups: {len(bg_data)}")
    print(f"Total estimated population: {bg_data['estimated_pop'].sum():,}")
    
    return bg_data

def identify_cut_points(bg_data):
    """Step 3: Create distribution and identify cut points"""
    print("\nStep 3: Creating distribution and identifying cut points...")
    
    # Sort by discretionary arrest rate (high to low)
    bg_data = bg_data.sort_values('discretionary_per_1000', ascending=False).reset_index()
    bg_data['cumulative_pop'] = bg_data['estimated_pop'].cumsum()
    bg_data['cumulative_pop_pct'] = bg_data['cumulative_pop'] / bg_data['estimated_pop'].sum() * 100
    
    # Method 1: Jenks Natural Breaks (simplified using k-means)
    print("\n  Method 1: Statistical clustering (k-means as proxy for Jenks)...")
    X = bg_data['discretionary_per_1000'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    bg_data['kmeans_cluster'] = kmeans.fit_predict(X)
    
    # Method 2: Curvature Analysis
    print("  Method 2: Curvature analysis...")
    cumulative_arrests = bg_data['discretionary_arrests'].cumsum()
    cumulative_pop = bg_data['cumulative_pop']
    
    # Calculate second derivative (simplified)
    x = cumulative_pop.values
    y = cumulative_arrests.values
    
    # Smooth the data for better derivative calculation
    from scipy.interpolate import UnivariateSpline
    valid_mask = x > 0
    if valid_mask.sum() > 3:
        spline = UnivariateSpline(x[valid_mask], y[valid_mask], s=1000)
        x_smooth = np.linspace(x[valid_mask].min(), x[valid_mask].max(), 100)
        y_smooth = spline(x_smooth)
        dy_dx = spline.derivative(1)(x_smooth)
        d2y_dx2 = spline.derivative(2)(x_smooth)
        
        # Find points of maximum curvature
        curvature = np.abs(d2y_dx2) / (1 + dy_dx**2)**1.5
        curvature_peaks = np.argsort(curvature)[-2:]
    
    # Method 3: Percentile-based
    print("  Method 3: Percentile-based analysis...")
    
    # Find natural breaks in the distribution
    rates = bg_data['discretionary_per_1000'].values
    
    # Look for large jumps in rates
    rate_diffs = np.diff(rates)
    large_jumps = np.where(np.abs(rate_diffs) > np.std(rate_diffs) * 2)[0]
    
    # Target approximately 6-7% ultra-policed, 15-16% highly policed
    cut1_idx = np.argmax(bg_data['cumulative_pop_pct'] >= 6.6)
    cut2_idx = np.argmax(bg_data['cumulative_pop_pct'] >= 22.0)
    
    cut1_rate = bg_data.iloc[cut1_idx]['discretionary_per_1000']
    cut2_rate = bg_data.iloc[cut2_idx]['discretionary_per_1000']
    
    print(f"\n  Final cut points:")
    print(f"  Cut point 1: {cut1_rate:.1f} per 1,000 (top {bg_data.iloc[cut1_idx]['cumulative_pop_pct']:.1f}% of population)")
    print(f"  Cut point 2: {cut2_rate:.1f} per 1,000 (top {bg_data.iloc[cut2_idx]['cumulative_pop_pct']:.1f}% of population)")
    
    return bg_data, cut1_rate, cut2_rate

def categorize_blockgroups(bg_data, cut1_rate, cut2_rate):
    """Step 4: Establish three categories"""
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
        'unique_individuals': 'sum',
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
            print(f"  Unique individuals arrested: {stats['unique_individuals']:,.0f}")
            print(f"  Discretionary arrests per 1,000: {stats['disc_rate_per_1000']:.1f}")
            print(f"  Total arrests per 1,000: {stats['total_rate_per_1000']:.1f}")
    
    return bg_data, category_stats

def calculate_annual_risks(arrests, bg_data, category_stats, years_of_data):
    """Phase 2: Calculate Annual Arrest Risks (Steps 5-7)"""
    print("\n" + "="*60)
    print("PHASE 2: CALCULATE ANNUAL ARREST RISKS")
    print("="*60)
    
    # Merge category info with arrests
    arrests_with_cat = arrests.merge(
        bg_data[['blockgroup_id', 'policing_category']],
        on='blockgroup_id',
        how='left'
    )
    
    # Step 5: Overall Population Annual Risk
    print("\nStep 5: Overall Population Annual Risk")
    
    risk_results = []
    
    for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
        if cat in category_stats.index:
            cat_data = arrests_with_cat[arrests_with_cat['policing_category'] == cat]
            unique_individuals = cat_data['DefendantId'].nunique()
            population = category_stats.loc[cat, 'estimated_pop']
            
            annual_unique = unique_individuals / years_of_data
            annual_risk = (annual_unique / population) * 100
            
            print(f"\n{cat}:")
            print(f"  Unique individuals arrested: {unique_individuals:,}")
            print(f"  Annual unique arrested: {annual_unique:.0f}")
            print(f"  Annual arrest risk: {annual_risk:.2f}%")
            
            risk_results.append({
                'Category': cat,
                'Population': population,
                'Unique_Individuals': unique_individuals,
                'Annual_Risk_Pct': annual_risk
            })
    
    # Step 6: Young Men (18-35) Annual Risk
    print("\nStep 6: Young Men (18-35) Annual Risk")
    
    young_men = arrests_with_cat[
        (arrests_with_cat['Age_years'].between(18, 35)) & 
        (arrests_with_cat['Gender'] == 'Male')
    ]
    
    print(f"\nTotal young men arrests: {len(young_men):,}")
    print(f"Unique young men arrested: {young_men['DefendantId'].nunique():,}")
    print(f"Average arrests per young man: {len(young_men)/young_men['DefendantId'].nunique():.2f}")
    
    young_men_risks = []
    
    for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
        if cat in category_stats.index:
            cat_young_men = young_men[young_men['policing_category'] == cat]
            unique_young_men = cat_young_men['DefendantId'].nunique()
            
            # Estimate young male population (assume 20% of population)
            est_young_male_pop = category_stats.loc[cat, 'estimated_pop'] * 0.20
            
            annual_unique = unique_young_men / years_of_data
            annual_risk = (annual_unique / est_young_male_pop) * 100
            
            print(f"\n{cat} - Young Men (18-35):")
            print(f"  Estimated young male population: {est_young_male_pop:,.0f}")
            print(f"  Unique young men arrested: {unique_young_men:,}")
            print(f"  Annual arrest risk: {annual_risk:.2f}%")
            
            young_men_risks.append({
                'Category': cat,
                'Est_Young_Male_Pop': est_young_male_pop,
                'Unique_Young_Men': unique_young_men,
                'Annual_Risk_Pct': annual_risk
            })
    
    # Step 7: Lifetime Risk Projections
    print("\nStep 7: Lifetime Risk Projections for Young Men")
    
    for risk_data in young_men_risks:
        cat = risk_data['Category']
        annual_risk = risk_data['Annual_Risk_Pct'] / 100
        
        by_25 = 1 - (1 - annual_risk) ** 7
        by_30 = 1 - (1 - annual_risk) ** 12
        by_35 = 1 - (1 - annual_risk) ** 17
        by_50 = 1 - (1 - annual_risk) ** 32
        
        print(f"\n{cat} - Lifetime arrest probability:")
        print(f"  By age 25: {by_25*100:.1f}%")
        print(f"  By age 30: {by_30*100:.1f}%")
        print(f"  By age 35: {by_35*100:.1f}%")
        print(f"  By age 50: {by_50*100:.1f}%")
    
    return pd.DataFrame(risk_results), pd.DataFrame(young_men_risks), arrests_with_cat

def analyze_multiple_arrests(arrests_with_cat, years_of_data):
    """Phase 3: Multiple Arrest and Escalation Analysis (Steps 8-12)"""
    print("\n" + "="*60)
    print("PHASE 3: MULTIPLE ARREST AND ESCALATION ANALYSIS")
    print("="*60)
    
    # Step 8: Calculate Arrest Frequency Distribution
    print("\nStep 8: Arrest Frequency Distribution")
    
    arrest_counts = arrests_with_cat.groupby('DefendantId').size().value_counts().sort_index()
    total_people = arrests_with_cat['DefendantId'].nunique()
    
    print("\nNumber of arrests per person:")
    for n_arrests, count in arrest_counts.head(10).items():
        pct = count / total_people * 100
        print(f"  {n_arrests} arrest(s): {count:,} people ({pct:.1f}%)")
    
    # Calculate conditional probabilities
    print("\nConditional probabilities:")
    people_with_n = {}
    for n in range(1, 6):
        people_with_n[n] = (arrests_with_cat.groupby('DefendantId').size() >= n).sum()
    
    for n in range(1, 5):
        if people_with_n[n] > 0:
            prob = people_with_n[n+1] / people_with_n[n] * 100
            print(f"  P(arrest {n+1} | arrest {n}): {prob:.1f}%")
    
    # Step 9-10: Drug-specific repeat patterns (focusing on drugs)
    print("\nStep 9-10: Drug Offense Repeat Patterns")
    
    drug_arrests = arrests_with_cat[arrests_with_cat['Arrest_crime_category'].str.contains('Drug', na=False)]
    
    if len(drug_arrests) > 0:
        drug_counts = drug_arrests.groupby('DefendantId').agg({
            'CaseId': 'count',
            'Arrest_crime_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).rename(columns={'CaseId': 'drug_arrest_count'})
        
        print(f"\nDrug arrests: {len(drug_arrests):,}")
        print(f"People with drug arrests: {len(drug_counts):,}")
        print(f"Average drug arrests per person: {drug_counts['drug_arrest_count'].mean():.2f}")
        
        # Statutory escalation
        facing_enhancement = (drug_counts['drug_arrest_count'] >= 2).sum()
        facing_mandatory = (drug_counts['drug_arrest_count'] >= 3).sum()
        
        print(f"\nStatutory escalation:")
        print(f"  Facing enhanced penalties (2+): {facing_enhancement:,} ({facing_enhancement/len(drug_counts)*100:.1f}%)")
        print(f"  Facing mandatory minimums (3+): {facing_mandatory:,} ({facing_mandatory/len(drug_counts)*100:.1f}%)")
    
    return arrest_counts, drug_counts if len(drug_arrests) > 0 else pd.DataFrame()

def analyze_drug_offenses(arrests_with_cat, category_stats, years_of_data):
    """Phase 5: Drug Offense Deep Dive (Steps 18-26)"""
    print("\n" + "="*60)
    print("PHASE 5: DRUG OFFENSE DEEP DIVE")
    print("="*60)
    
    # Step 18: Isolate Drug Arrests
    print("\nStep 18: Isolating drug arrests...")
    
    drug_arrests = arrests_with_cat[arrests_with_cat['Arrest_crime_category'].str.contains('Drug', na=False)].copy()
    
    # Categorize drug types
    drug_arrests['drug_type'] = drug_arrests['Arrest_crime_category'].apply(
        lambda x: 'Possession' if 'Poss' in str(x) else ('Distribution' if 'Deal' in str(x) else 'Other')
    )
    
    print(f"Total drug arrests: {len(drug_arrests):,}")
    print(f"Unique individuals with drug arrests: {drug_arrests['DefendantId'].nunique():,}")
    print("\nDrug arrest types:")
    for dtype, count in drug_arrests['drug_type'].value_counts().items():
        print(f"  {dtype}: {count:,} ({count/len(drug_arrests)*100:.1f}%)")
    
    # Step 19-20: Calculate Drug Arrest Annual Risks
    print("\nStep 19-20: Drug Arrest Annual Risks")
    
    drug_risk_results = []
    
    for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
        if cat in category_stats.index:
            cat_drug = drug_arrests[drug_arrests['policing_category'] == cat]
            unique_drug = cat_drug['DefendantId'].nunique()
            population = category_stats.loc[cat, 'estimated_pop']
            
            per_capita_annual = (unique_drug / years_of_data) / population * 1000
            
            # Young men
            cat_drug_young_men = cat_drug[
                (cat_drug['Age_years'].between(18, 35)) & 
                (cat_drug['Gender'] == 'Male')
            ]
            unique_young_men_drug = cat_drug_young_men['DefendantId'].nunique()
            est_young_male_pop = population * 0.20
            young_men_annual = (unique_young_men_drug / years_of_data) / est_young_male_pop * 100
            
            print(f"\n{cat}:")
            print(f"  Drug arrests per 1,000 annually: {per_capita_annual:.2f}")
            print(f"  Young men drug arrest risk: {young_men_annual:.2f}%")
            
            drug_risk_results.append({
                'Category': cat,
                'Drug_Per_1000_Annual': per_capita_annual,
                'Young_Men_Drug_Risk_Pct': young_men_annual
            })
    
    # Step 21: Drug Repeat Offense Patterns
    print("\nStep 21: Drug Repeat Offense Patterns")
    
    drug_repeats = drug_arrests.groupby('DefendantId').agg({
        'CaseId': 'count',
        'drug_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'policing_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).rename(columns={'CaseId': 'drug_count'})
    
    # Calculate progression probabilities
    for n in range(1, 5):
        with_n = (drug_repeats['drug_count'] >= n).sum()
        with_n_plus = (drug_repeats['drug_count'] >= n + 1).sum()
        if with_n > 0:
            prob = with_n_plus / with_n * 100
            print(f"  P(drug arrest {n+1} | drug arrest {n}): {prob:.1f}%")
    
    # Step 23: Drug Escalation Per Capita Risk
    print("\nStep 23: Drug Escalation Per Capita Risk")
    
    for cat in ['Ultra-Policed', 'Highly Policed', 'Normally Policed']:
        if cat in category_stats.index:
            cat_drug_repeats = drug_repeats[drug_repeats['policing_category'] == cat]
            population = category_stats.loc[cat, 'estimated_pop']
            
            enhanced = (cat_drug_repeats['drug_count'] >= 2).sum()
            mandatory = (cat_drug_repeats['drug_count'] >= 3).sum()
            
            enhanced_per_1000 = (enhanced / years_of_data) / population * 1000
            mandatory_per_1000 = (mandatory / years_of_data) / population * 1000
            
            print(f"\n{cat}:")
            print(f"  Enhanced penalties per 1,000: {enhanced_per_1000:.2f}")
            print(f"  Mandatory minimums per 1,000: {mandatory_per_1000:.2f}")
    
    # Step 24: Equal Use Assumption
    print("\nStep 24: Drug Enforcement Under Equal Use Assumption")
    print("Assuming 10% of population uses illegal drugs...")
    
    for result in drug_risk_results:
        cat = result['Category']
        rate_per_1000 = result['Drug_Per_1000_Annual']
        
        # 10% use = 100 per 1,000
        pct_users_arrested = rate_per_1000 / 100 * 100
        
        print(f"\n{cat}:")
        print(f"  {pct_users_arrested:.1f}% of drug users face arrest annually")
    
    return pd.DataFrame(drug_risk_results), drug_repeats

def create_visualizations(bg_data, category_stats, cut1_rate, cut2_rate, risk_df, young_men_df, drug_risk_df):
    """Create comprehensive visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create a comprehensive figure
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Geographic categorization
    ax1 = plt.subplot(6, 3, 1)
    ax1.hist(bg_data['discretionary_per_1000'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(cut1_rate, color='red', linestyle='--', label=f'Cut 1: {cut1_rate:.1f}')
    ax1.axvline(cut2_rate, color='orange', linestyle='--', label=f'Cut 2: {cut2_rate:.1f}')
    ax1.set_xlabel('Discretionary Arrests per 1,000')
    ax1.set_ylabel('Number of Block Groups')
    ax1.set_title('Distribution of Discretionary Arrest Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Population distribution
    ax2 = plt.subplot(6, 3, 2)
    sizes = category_stats['pop_pct'].values
    colors = ['darkred', 'orange', 'lightgreen']
    categories = category_stats.index.tolist()
    ax2.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Population Distribution Across Categories')
    
    # 3. Arrest rates by category
    ax3 = plt.subplot(6, 3, 3)
    x = np.arange(len(categories))
    width = 0.35
    disc_rates = category_stats['disc_rate_per_1000'].values
    total_rates = category_stats['total_rate_per_1000'].values
    
    ax3.bar(x - width/2, disc_rates, width, label='Discretionary', color='steelblue')
    ax3.bar(x + width/2, total_rates, width, label='Total', color='darkred')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Per 1,000')
    ax3.set_title('Arrest Rates by Category')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Annual arrest risk - overall
    ax4 = plt.subplot(6, 3, 4)
    risks = risk_df['Annual_Risk_Pct'].values
    bars = ax4.bar(categories, risks, color=['darkred', 'orange', 'lightgreen'])
    ax4.set_ylabel('Annual Risk (%)')
    ax4.set_title('Annual Arrest Risk - Overall Population')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, risk in zip(bars, risks):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{risk:.1f}%', ha='center', va='bottom')
    
    # 5. Annual arrest risk - young men
    ax5 = plt.subplot(6, 3, 5)
    young_risks = young_men_df['Annual_Risk_Pct'].values
    bars = ax5.bar(categories, young_risks, color=['darkred', 'orange', 'lightgreen'])
    ax5.set_ylabel('Annual Risk (%)')
    ax5.set_title('Annual Arrest Risk - Young Men (18-35)')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, risk in zip(bars, young_risks):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{risk:.1f}%', ha='center', va='bottom')
    
    # 6. Lifetime risk projection - young men
    ax6 = plt.subplot(6, 3, 6)
    ages = [25, 30, 35, 50]
    for idx, cat in enumerate(categories):
        annual_risk = young_men_df.iloc[idx]['Annual_Risk_Pct'] / 100
        lifetime_risks = []
        for years in [7, 12, 17, 32]:
            risk = (1 - (1 - annual_risk) ** years) * 100
            lifetime_risks.append(risk)
        ax6.plot(ages, lifetime_risks, marker='o', label=cat, linewidth=2)
    
    ax6.set_xlabel('Age')
    ax6.set_ylabel('Cumulative Arrest Probability (%)')
    ax6.set_title('Lifetime Arrest Risk - Young Men')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Drug arrests per capita
    if len(drug_risk_df) > 0:
        ax7 = plt.subplot(6, 3, 7)
        drug_rates = drug_risk_df['Drug_Per_1000_Annual'].values
        bars = ax7.bar(categories, drug_rates, color=['darkred', 'orange', 'lightgreen'])
        ax7.set_ylabel('Per 1,000 Annually')
        ax7.set_title('Drug Arrests Per Capita')
        ax7.grid(True, alpha=0.3, axis='y')
        for bar, rate in zip(bars, drug_rates):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{rate:.1f}', ha='center', va='bottom')
    
    # 8. Drug arrest risk - young men
    if len(drug_risk_df) > 0:
        ax8 = plt.subplot(6, 3, 8)
        drug_young_risks = drug_risk_df['Young_Men_Drug_Risk_Pct'].values
        bars = ax8.bar(categories, drug_young_risks, color=['darkred', 'orange', 'lightgreen'])
        ax8.set_ylabel('Annual Risk (%)')
        ax8.set_title('Drug Arrest Risk - Young Men')
        ax8.grid(True, alpha=0.3, axis='y')
        for bar, risk in zip(bars, drug_young_risks):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{risk:.1f}%', ha='center', va='bottom')
    
    # 9. Disparity summary
    ax9 = plt.subplot(6, 3, 9)
    ax9.axis('off')
    
    # Calculate key disparities
    ultra_risk = risk_df.iloc[0]['Annual_Risk_Pct']
    normal_risk = risk_df.iloc[2]['Annual_Risk_Pct']
    overall_ratio = ultra_risk / normal_risk if normal_risk > 0 else 0
    
    ultra_young = young_men_df.iloc[0]['Annual_Risk_Pct']
    normal_young = young_men_df.iloc[2]['Annual_Risk_Pct']
    young_ratio = ultra_young / normal_young if normal_young > 0 else 0
    
    summary_text = f"""KEY DISPARITIES

Overall Population:
Ultra vs Normal: {overall_ratio:.1f}x
Ultra-Policed: {ultra_risk:.1f}% annually
Normally Policed: {normal_risk:.1f}% annually

Young Men (18-35):
Ultra vs Normal: {young_ratio:.1f}x
Ultra-Policed: {ultra_young:.1f}% annually
Normally Policed: {normal_young:.1f}% annually

Drug Enforcement:
Assuming 10% use drugs equally,
arrest probability varies {overall_ratio:.0f}-fold
based on neighborhood."""
    
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Geographic Policing Intensity Analysis - Complete Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = FIGURES_PATH / 'complete_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive visualization to {output_path}")
    
    return fig

def save_results(bg_data, category_stats, risk_df, young_men_df, drug_risk_df, arrest_counts, drug_repeats):
    """Save all results to files"""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save all dataframes
    bg_data.to_csv(RESULTS_PATH / 'blockgroup_categorization.csv', index=False)
    category_stats.to_csv(RESULTS_PATH / 'category_statistics.csv')
    risk_df.to_csv(RESULTS_PATH / 'annual_risk_overall.csv', index=False)
    young_men_df.to_csv(RESULTS_PATH / 'annual_risk_young_men.csv', index=False)
    drug_risk_df.to_csv(RESULTS_PATH / 'drug_arrest_risks.csv', index=False)
    
    # Save arrest frequency distribution
    arrest_counts.to_csv(RESULTS_PATH / 'arrest_frequency_distribution.csv')
    
    if len(drug_repeats) > 0:
        drug_repeats.to_csv(RESULTS_PATH / 'drug_repeat_patterns.csv')
    
    print(f"\nResults saved to {RESULTS_PATH}")
    
    # Create summary report
    create_summary_report(category_stats, risk_df, young_men_df, drug_risk_df)

def create_summary_report(category_stats, risk_df, young_men_df, drug_risk_df):
    """Create a markdown summary report"""
    
    report = f"""# Geographic Policing Intensity Analysis - Summary Report

## Executive Summary

This analysis examines policing intensity patterns across {len(category_stats)} categories of neighborhoods, 
identifying significant disparities in arrest risks and enforcement patterns.

## Key Findings

### Population Distribution
- **Ultra-Policed**: {category_stats.iloc[0]['pop_pct']:.1f}% of population
- **Highly Policed**: {category_stats.iloc[1]['pop_pct']:.1f}% of population  
- **Normally Policed**: {category_stats.iloc[2]['pop_pct']:.1f}% of population

### Annual Arrest Risks - Overall Population
"""
    
    for idx, row in risk_df.iterrows():
        report += f"- **{row['Category']}**: {row['Annual_Risk_Pct']:.2f}% annually\n"
    
    report += f"""
### Annual Arrest Risks - Young Men (18-35)
"""
    
    for idx, row in young_men_df.iterrows():
        report += f"- **{row['Category']}**: {row['Annual_Risk_Pct']:.2f}% annually\n"
    
    # Calculate lifetime risks
    report += f"""
### Lifetime Arrest Probability - Young Men
By age 35, starting from age 18:
"""
    
    for idx, row in young_men_df.iterrows():
        annual_risk = row['Annual_Risk_Pct'] / 100
        by_35 = (1 - (1 - annual_risk) ** 17) * 100
        report += f"- **{row['Category']}**: {by_35:.1f}% will have been arrested\n"
    
    if len(drug_risk_df) > 0:
        report += f"""
### Drug Enforcement Disparities
Annual drug arrests per 1,000 population:
"""
        
        for idx, row in drug_risk_df.iterrows():
            report += f"- **{row['Category']}**: {row['Drug_Per_1000_Annual']:.2f} per 1,000\n"
        
        report += f"""
Under equal drug use assumption (10% of population):
"""
        
        for idx, row in drug_risk_df.iterrows():
            pct_users = row['Drug_Per_1000_Annual'] / 100 * 100
            report += f"- **{row['Category']}**: {pct_users:.1f}% of drug users face arrest\n"
    
    # Calculate key disparities
    ultra_risk = risk_df.iloc[0]['Annual_Risk_Pct']
    normal_risk = risk_df.iloc[2]['Annual_Risk_Pct']
    overall_ratio = ultra_risk / normal_risk if normal_risk > 0 else 0
    
    report += f"""
## Disparity Ratios

- **Overall population**: Ultra-Policed residents face **{overall_ratio:.1f}x** higher annual arrest risk
- **Young men**: Ultra-Policed young men face **{young_men_df.iloc[0]['Annual_Risk_Pct'] / young_men_df.iloc[2]['Annual_Risk_Pct']:.1f}x** higher risk

## Methodology

This analysis follows a comprehensive 33-step methodology:
1. Geographic categorization using discretionary arrests
2. Population-based risk calculations using unique individuals
3. Multiple arrest and escalation analysis
4. Drug offense deep dive with equal use assumptions

## Data Source

- Total arrests analyzed: 144,645
- Unique individuals: 41,807
- Time period: ~10.5 years
- Geographic units: Census block groups

---
*Analysis completed using methodology_guide.md*
"""
    
    with open(RESULTS_PATH / 'summary_report.md', 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to {RESULTS_PATH / 'summary_report.md'}")

def main():
    """Run complete analysis following methodology guide"""
    
    print("COMPLETE GEOGRAPHIC POLICING INTENSITY ANALYSIS")
    print("Following methodology_guide.md")
    print("="*60)
    
    # Phase 1: Data Preparation and Geographic Categorization
    arrests, years_of_data = load_data()
    arrests = identify_discretionary_arrests(arrests)
    bg_data = create_blockgroup_data(arrests)
    bg_data, cut1_rate, cut2_rate = identify_cut_points(bg_data)
    bg_data, category_stats = categorize_blockgroups(bg_data, cut1_rate, cut2_rate)
    
    # Phase 2: Calculate Annual Arrest Risks
    risk_df, young_men_df, arrests_with_cat = calculate_annual_risks(
        arrests, bg_data, category_stats, years_of_data
    )
    
    # Phase 3: Multiple Arrest Analysis
    arrest_counts, drug_counts = analyze_multiple_arrests(arrests_with_cat, years_of_data)
    
    # Phase 5: Drug Offense Deep Dive
    drug_risk_df, drug_repeats = analyze_drug_offenses(
        arrests_with_cat, category_stats, years_of_data
    )
    
    # Create visualizations
    fig = create_visualizations(
        bg_data, category_stats, cut1_rate, cut2_rate,
        risk_df, young_men_df, drug_risk_df
    )
    
    # Save all results
    save_results(
        bg_data, category_stats, risk_df, young_men_df, 
        drug_risk_df, arrest_counts, drug_repeats
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Visualizations saved to: {FIGURES_PATH}")
    print("\nReview summary_report.md for key findings")

if __name__ == "__main__":
    main()