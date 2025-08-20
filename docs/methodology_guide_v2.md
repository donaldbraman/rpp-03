# Step-by-Step Guide for Geographic Policing Intensity Analysis (Version 2)
## Updated with Actual Census Data Requirements

## Phase 1: Data Preparation and Geographic Categorization

### Step 1: Load and Prepare Geographic Data
- **Input**: Anonymous arrest data with census tract/block group identifiers
- **Key fields needed**: 
  - DefendantId (unique person identifier)
  - DefendantAddressGEOID10 (census tract)
  - Arrest_crime_category (offense type)
  - ArrestDate
  - Age_years, Gender, Race (demographics)
  - Outcome (disposition/conviction)
  - Incarceration/exposure (sentence length)
- **Geographic unit**: Census block groups (12-digit GEOID)
- **Time period**: Define analysis window (e.g., 5 years)

### Step 1A: Obtain Census Block Group Population Data (CRITICAL)
- **Data source**: American Community Survey (ACS) 5-year estimates
- **Geographic level**: Block group (12-digit GEOID)
- **Required tables**:
  - B01001: Sex by Age (total population counts)
  - B02001: Race 
  - B03002: Hispanic or Latino Origin by Race
  - B25077: Median Home Value (optional, for context)
  - B19013: Median Household Income (optional, for context)
  - B17001: Poverty Status (optional, for context)

- **Download methods**:
  1. **Census API**: 
     - Register for API key at https://api.census.gov/data/key_signup.html
     - Use census Python package or direct API calls
     - Example endpoint: `https://api.census.gov/data/2019/acs/acs5?get=B01001_001E&for=block%20group:*&in=state:45&in=county:*`
  
  2. **IPUMS NHGIS**:
     - https://www.nhgis.org/
     - Select geographic level: Block Group
     - Select tables needed
     - Download as CSV with geographic identifiers
  
  3. **Census Bureau Data Portal**:
     - https://data.census.gov/
     - Advanced search by geography
     - Export selected tables

- **Key fields to extract**:
  - GEOID: 12-digit block group identifier
  - total_pop: Total population (B01001_001E)
  - male_pop: Male population (B01001_002E)
  - female_pop: Female population (B01001_026E)
  - white_pop: White alone (B02001_002E)
  - black_pop: Black alone (B02001_003E)
  - hispanic_pop: Hispanic or Latino (B03002_012E)
  - median_income: Median household income (B19013_001E)
  - poverty_count: Below poverty level (B17001_002E)

### Step 1B: Merge Census Data with Geographic Units
- **Match block groups**: Join census data to arrest data using GEOID
- **Handle missing matches**: Some block groups may have no arrests
- **Validation**: 
  - Ensure population totals match county/city totals
  - Check for zero or null populations
  - Verify GEOID format consistency (12 digits)

### Step 2: Identify Discretionary Arrests
- **Definition**: Arrests where officers have discretion vs mandatory arrests
- **Discretionary categories**:
  - Drug Possession (not distribution)
  - Property crimes (minor)
  - Traffic violations (non-DUI)
  - Other Offenses (miscellaneous)
  - Theft (shoplifting, minor)
- **Calculate**: Discretionary arrest rate per 1,000 population for each block group
- **CRITICAL**: Use actual census population as denominator, not estimates

### Step 3: Create Distribution and Identify Cut Points
- **Sort**: Order block groups by discretionary arrest rate (high to low)
- **Use actual populations**: Weight by real census populations, not estimates
- **Calculate cumulative population**: Running sum of actual population
- **Create histogram**: Distribution of discretionary arrest rates across block groups
- **Identify natural breaks using three methods**:
  
  a) **Statistical Method (Jenks Natural Breaks)**:
     - Minimize within-group variance
     - Maximize between-group variance
     - Use 3 classes for clear categories
  
  b) **Curvature Analysis**:
     - Plot cumulative arrests vs cumulative population
     - Calculate second derivative to find inflection points
     - Points of maximum curvature indicate natural boundaries
  
  c) **Percentile-Based**:
     - Examine arrest rates at population percentiles
     - Look for sharp changes in rate of increase
     - Common breaks: 90th, 95th percentiles

### Step 4: Establish Three Categories
- **Final cut points**: Choose based on convergence of methods
- **Typical result with real census data**:
  - Ultra-Policed: Top 5-10% of population (~6.6%)
  - Highly Policed: Next 10-20% of population (~15.4%)
  - Normally Policed: Remaining 70-80% (~77.9%)
- **Validation**: 
  - Ensure each category has sufficient actual population
  - Verify arrest rate differences are substantial (>2x between categories)

## Phase 2: Calculate Annual Arrest Risks

### Step 5: Overall Population Annual Risk
- **Formula**: Annual Risk = (Unique individuals arrested per year / **Actual census population**) × 100
- **Key correction**: Use unique DefendantId count, NOT total arrests
- **CRITICAL**: Use census total_pop field, not estimates
- **By category**:
  - Count unique individuals arrested in each policing category
  - Divide by years in dataset (e.g., 5)
  - Divide by actual census population for category
  - Multiply by 100 for percentage
- **Expected results with census data**:
  - Ultra-Policed: ~1.5-3% annually
  - Highly Policed: ~0.7-1.5% annually
  - Normally Policed: ~0.3-0.7% annually

### Step 6: Young Men (18-35) Annual Risk
- **Filter data**: Age 18-35, Gender = Male
- **Calculate repeat factor**: Total arrests / Unique individuals
- **Estimate young male population from census**:
  - Sum relevant age brackets (portions of B01001 table)
  - Males 18-19 (part of B01001_007E)
  - Males 20-24 (B01001_008E + B01001_009E + B01001_010E)
  - Males 25-29 (B01001_011E)
  - Males 30-34 (B01001_012E)
  - Males 35 (part of B01001_013E)
- **Annual risk formula**:
  - Unique young men arrested / Years of data = Annual unique arrested
  - Annual unique arrested / Census young male population = Annual risk
- **Expected disparities**: 4-8x between Ultra and Normal

### Step 7: Lifetime Risk Projections
- **Formula**: P(arrest by age X) = 1 - (1 - annual_risk)^years
- **Standard age points**:
  - By 25: 7 years from 18
  - By 30: 12 years from 18
  - By 35: 17 years from 18
  - By 50: 32 years from 18
- **With real census data, expect**:
  - Ultra-Policed young men: 40-60% by age 35
  - Normally Policed young men: 10-15% by age 35

## Phase 3: Multiple Arrest and Escalation Analysis

### Step 8: Calculate Arrest Frequency Distribution
- **Group by DefendantId**: Count total arrests per person
- **Create frequency table**:
  - 1 arrest only: X% of people
  - 2 arrests: Y% of people
  - 3 arrests: Z% of people
  - 4+ arrests: W% of people
- **Calculate conditional probabilities**:
  - P(2nd arrest | 1st arrest)
  - P(3rd arrest | 2nd arrest)
  - P(4th+ arrest | 3rd arrest)

### Step 9: Identify Repeat Offense Patterns
- **For specific offense types** (especially drugs):
  - Track arrest sequence by DefendantId and date
  - Number each arrest chronologically (1st, 2nd, 3rd, etc.)
  - Identify offense type progression (possession → distribution)
- **Calculate repeat rates**:
  - Average arrests per person by offense type
  - Time between arrests (recidivism velocity)
  - Progression patterns between offense types

### Step 10: Map Statutory Escalation Triggers
- **Identify enhancement thresholds**:
  - 2nd offense: Enhanced penalties begin
  - 3rd offense: Mandatory minimums trigger
  - 4th+ offense: Severe escalation
- **Document penalty increases**:
  - Average sentence by offense number
  - Conviction rate by offense number
  - Incarceration rate by offense number
- **Calculate escalation impact**:
  - % facing enhanced penalties (2nd+)
  - % facing mandatory minimums (3rd+)
  - Average sentence multiplication factor

### Step 11: Calculate Per Capita Escalation Risk
- **Annual risk of facing enhancement**:
  - People with 2+ arrests in category / Years of data = Annual enhanced
  - Annual enhanced / **Census population** = Per capita enhancement risk
  - Express per 1,000 population
- **Annual risk of mandatory minimums**:
  - People with 3+ arrests / Years of data = Annual mandatory
  - Annual mandatory / **Census population** = Per capita mandatory risk
  - Express per 1,000 population
- **CRITICAL**: Always use actual census populations

### Step 12: Model Cumulative Escalation Risk
- **Multi-year projection**:
  - Year 1: X% arrested, Y% of those get 2nd arrest same year
  - Year 2: New arrests + repeat arrests from Year 1
  - Track accumulation of enhanced penalties over time
- **Cascade calculation**:
  - Initial arrest probability × P(repeat) × P(enhancement)
  - Compound over multiple years
  - Show how small differences in initial risk amplify

## Phase 4: Demographic Analysis

### Step 13: Use Census Demographics (Already Obtained)
- **From Step 1A census data**:
  - Use race fields (white_pop, black_pop, etc.)
  - Use age/sex fields from B01001 table
  - Cross-tabulations may require proportional estimation

### Step 14: Calculate Young Male Population by Race
- **Challenge**: Census doesn't provide race × age × gender at block group level
- **Solution**: Proportional allocation
  - Get total young males from B01001 (Step 6)
  - Get racial proportions from B02001
  - Multiply: Young males × racial proportion
  - Example: If 30% Black population and 1,000 young males = 300 Black young males
- **Validation**: Sum should equal total young male population

### Step 15: Calculate Race × Age × Gender Specific Risks
- **For each demographic group**:
  - Filter arrests: Race, Age 18-35, Gender
  - Count unique individuals
  - Divide by census-estimated population for that group
  - Calculate annual risk
- **Expected findings**:
  - Significant disparities both within and across neighborhoods
  - Compound effects of race and geography

### Step 16: Calculate Race-Specific Risks
- **For each race × neighborhood combination**:
  - Numerator: Unique individuals of that race arrested
  - Denominator: Census population of that race
  - Annual risk: (Numerator / Years) / Denominator × 100
- **Disparity ratios**: 
  - Within neighborhood: Black risk / White risk
  - Across neighborhoods: Black ultra-policed / White normal
  - Document all disparities found

### Step 17: Calculate Race-Specific Escalation Risks
- **Per capita enhancement risk by race**:
  - Black with 2+ arrests / Census Black population × 1,000
  - White with 2+ arrests / Census White population × 1,000
  - Calculate ratio between groups
- **Compound disparities**:
  - Initial arrest disparity × Escalation probability
  - Show multiplicative effect
  - Document how "race-neutral" escalation amplifies disparities

## Phase 5: Drug Offense Deep Dive

### Step 18: Isolate Drug Arrests
- **Filter criteria**:
  - Arrest_crime_category contains "Drug"
  - Separate into: Drug Possession, Drug Dealing/Distribution, Drug Other
- **Count unique individuals**: DefendantId with any drug arrest
- **Calculate drug-specific metrics**:
  - Total drug arrests per neighborhood category
  - Unique individuals with drug arrests
  - Drug arrests per unique person (repeat factor)

### Step 19: Calculate Drug Arrest Annual Risks - General Population
- **Per capita drug arrest risk**:
  - Unique people with drug arrests / **Census population** × 1,000
  - Annual risk = Above / Years of data
- **By neighborhood category**:
  - Ultra-Policed: X per 1,000 annually
  - Highly Policed: Y per 1,000 annually
  - Normally Policed: Z per 1,000 annually
- **Calculate disparities**:
  - Ultra vs Normal ratio (expect 4-8x)

### Step 20: Calculate Drug Arrest Annual Risks - Young Men
- **Filter**: Age 18-35, Gender = Male, Drug arrests only
- **Metrics**:
  - Unique young men with drug arrests
  - Total drug arrests among young men
  - Average drug arrests per person
- **Annual risk calculation**:
  - Annual unique arrested for drugs / **Census young male population** × 100
- **By neighborhood**:
  - Compare across three categories
  - Calculate disparity ratios

### Step 21: Analyze Drug Repeat Offense Patterns
- **Create drug arrest sequence**:
  - Order drug arrests by DefendantId and date
  - Number sequentially (1st drug, 2nd drug, etc.)
- **Calculate progression probabilities**:
  - P(2nd drug arrest | 1st drug arrest)
  - P(3rd drug arrest | 2nd drug arrest)
  - Time between drug arrests
- **Identify escalation triggers**:
  - 2nd drug offense = Enhanced penalties
  - 3rd drug offense = Mandatory minimums
  - Document % facing each level

### Step 22: Drug Offense Type Progression
- **Track offense evolution**:
  - Simple possession → Possession with intent
  - Possession → Distribution
  - Distribution → Trafficking
- **Calculate progression rates**:
  - % starting with possession who progress to distribution
  - % with multiple possession charges
  - % with mixed possession/distribution
- **By neighborhood category**:
  - Compare progression patterns
  - Test if progression rates differ by policing intensity

### Step 23: Calculate Drug Escalation Per Capita Risk
- **Enhanced drug penalties per capita**:
  - People with 2+ drug arrests / **Census population** × 1,000
  - Break down by neighborhood category
- **Mandatory minimum risk**:
  - People with 3+ drug arrests / **Census population** × 1,000
  - Compare across categories
- **Young men specific**:
  - Same calculations for census-derived 18-35 male population
  - Show amplified disparities

### Step 24: Model Drug Enforcement Under Equal Use Assumption
- **Research baseline**: ~10% of population uses illegal drugs
- **Calculate enforcement probability**:
  - Drug arrests per capita / Assumed use rate
  - Shows % of users who face arrest
- **By neighborhood**:
  - Ultra-Policed: X% of drug users arrested
  - Normally Policed: Y% of drug users arrested
  - Ratio shows enforcement disparity despite equal use

### Step 25: Project Lifetime Drug Enforcement Risk
- **For young men starting at 18**:
  - Annual drug arrest risk by neighborhood
  - P(drug arrest by 25) = 1 - (1 - annual_risk)^7
  - P(drug arrest by 35) = 1 - (1 - annual_risk)^17
  - P(drug arrest by 50) = 1 - (1 - annual_risk)^32
- **Enhanced penalty accumulation**:
  - P(facing enhancement by 35)
  - P(facing mandatory minimum by 35)
  - Show cascade effect over lifetime

### Step 26: Calculate Drug Sentencing Disparities
- **Average sentence by drug offense number**:
  - 1st drug offense: X days average
  - 2nd drug offense: Y days (% increase)
  - 3rd drug offense: Z days (% increase)
  - 4th+ drug offense: W days (% increase)
- **Total incarceration burden**:
  - Sum of drug sentences per 1,000 **census population**
  - By neighborhood category
  - Show multiplicative effect of repeat arrests + escalation

## Phase 6: Population Impact Analysis

### Step 27: Calculate Community-Wide Burden
- **Total arrests per day**: By neighborhood type
- **Families affected annually**:
  - Use census household data (average household size)
  - P(family member arrested) = 1 - (1 - individual_risk)^household_size
- **Economic impact**:
  - Average cost per arrest (bail, fines, legal fees)
  - Lost wages from incarceration
  - Multiply by per capita arrest rate
  - Compare to census median income

### Step 28: Project Long-Term Population Effects
- **Incarceration exposure**:
  - Total jail/prison days per 1,000 **census population**
  - By neighborhood and demographics
  - Account for sentence escalation
- **Criminal record accumulation**:
  - % with any record by age 35
  - % with felony convictions
  - % facing employment barriers
- **Use census economic data for context**:
  - Compare burden to median household income
  - Assess impact on poverty rates

### Step 29: Model Alternative Scenarios
- **Equal enforcement scenario**:
  - Apply normally-policed arrest rates to all areas
  - Calculate reduction in disparities
  - Estimate decreased incarceration
- **First-arrest diversion scenario**:
  - Model 50% diversion rate for first arrests
  - Calculate cascade prevention
  - Project reduction in enhanced penalties

## Phase 7: Quality Checks and Validation

### Step 30: Validate Census Data Integration
- **Coverage check**: 
  - What % of arrests matched to census block groups?
  - What % of census block groups have arrest data?
- **Population validation**:
  - Sum block group populations = County/city total?
  - Age/sex distributions reasonable?
  - Racial composition matches known demographics?

### Step 31: Validate Arrest Rate Calculations
- **Denominator checks**:
  - All rates use census populations, not estimates
  - Zero-population block groups excluded
  - Rates per 1,000 are reasonable (<1000)
- **Numerator checks**:
  - Using unique individuals, not arrest counts
  - Time period adjustment applied correctly

### Step 32: Validate Escalation Patterns
- **Internal consistency**:
  - People with 3 arrests ⊂ People with 2 arrests
  - Enhancement % should increase with arrest count
- **External validation**:
  - Compare conviction rates to court statistics
  - Check sentence lengths against guidelines
  - Verify repeat rates against recidivism studies

### Step 33: Calculate Confidence Bounds
- **Small area problem**: Some block groups have small populations
- **Approach**: Use binomial confidence intervals for arrest rates
- **Rule of thumb**: Require minimum 30 arrests for reliable estimates
- **Aggregate if needed**: Combine similar small block groups

### Step 34: Sensitivity Analysis
- **Test different cut points**: ±1-2 percentage points
- **Test demographic assumptions**: ±5% on age/race estimates  
- **Test escalation assumptions**: ±10% on repeat rates
- **Document stability**: How much do results change?
- **Key finding preservation**: Ensure main disparities robust to assumptions

## Key Methodological Notes

### Critical Requirements for Valid Analysis
1. **MUST use actual census population data**
   - Never use arrest-based population estimates
   - Obtain from ACS 5-year estimates for stability
   - Match geographic units precisely (12-digit GEOID)

2. **Use unique individuals via DefendantId, not arrest counts**
   - Prevents ~3.5x overestimation of risk
   
3. **Base categories on discretionary arrests only**
   - Isolates police decision-making from crime reports
   
4. **Track individual arrest sequences**
   - Enables escalation and repeat offense analysis
   
5. **Linear scales in visualizations**
   - Avoid log scales that obscure disparities

### Expected Results with Real Census Data
- **Overall disparities**: 4-8x between Ultra and Normally Policed
- **Young men disparities**: 5-10x between categories
- **Drug enforcement**: 4-8x despite equal use assumptions
- **Lifetime risks**: 40-60% for Ultra-Policed young men by age 35

### Common Pitfalls to Avoid
1. **Using arrest counts as population proxy** - Creates circular logic
2. **Missing census block groups** - Can bias results
3. **Ignoring zero-population areas** - Causes infinite rates
4. **Not adjusting for time period** - Misses annual rate calculation
5. **Double-counting repeat arrests** - Overestimates affected individuals

### Documentation Requirements
- **Census data source**: ACS year and version
- **Geographic matching rate**: % of arrests matched to census
- **Population coverage**: % of county/city population included
- **Time period**: Start and end dates of arrest data
- **Cut point methodology**: Which method(s) determined categories
- **Assumptions made**: All demographic estimations

This methodology produces robust, replicable analysis of policing intensity patterns and their demographic impacts while maintaining individual privacy through geographic aggregation. The use of actual census data is critical for accurate per capita calculations and revealing true disparities.