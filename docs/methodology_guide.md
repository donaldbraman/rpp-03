# Step-by-Step Guide for Geographic Policing Intensity Analysis

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

### Step 2: Identify Discretionary Arrests
- **Definition**: Arrests where officers have discretion vs mandatory arrests
- **Discretionary categories**:
  - Drug Possession (not distribution)
  - Property crimes (minor)
  - Traffic violations (non-DUI)
  - Other Offenses (miscellaneous)
  - Theft (shoplifting, minor)
- **Calculate**: Discretionary arrest rate per 1,000 population for each block group

### Step 3: Create Distribution and Identify Cut Points
- **Sort**: Order block groups by discretionary arrest rate (high to low)
- **Calculate cumulative population**: Running sum of population as you go down the list
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
- **Typical result**:
  - Ultra-Policed: Top 5-10% of population (~6.6%)
  - Highly Policed: Next 10-20% of population (~15.4%)
  - Normally Policed: Remaining 70-80% (~77.9%)
- **Validation**: Ensure each category has sufficient population for analysis

## Phase 2: Calculate Annual Arrest Risks

### Step 5: Overall Population Annual Risk
- **Formula**: Annual Risk = (Unique individuals arrested per year / Total population) × 100
- **Key correction**: Use unique DefendantId count, NOT total arrests
- **By category**:
  - Count unique individuals arrested in each policing category
  - Divide by years in dataset (e.g., 5)
  - Divide by category population
  - Multiply by 100 for percentage

### Step 6: Young Men (18-35) Annual Risk
- **Filter data**: Age 18-35, Gender = Male
- **Calculate repeat factor**: Total arrests / Unique individuals
  - Important: People average multiple arrests (typically 3-4 over 5 years)
- **Annual risk formula**:
  - Unique young men arrested / Years of data = Annual unique arrested
  - Annual unique arrested / Young male population = Annual risk
- **By category**: Repeat for each policing intensity group

### Step 7: Lifetime Risk Projections
- **Formula**: P(arrest by age X) = 1 - (1 - annual_risk)^years
- **Standard age points**:
  - By 25: 7 years from 18
  - By 30: 12 years from 18
  - By 35: 17 years from 18
  - By 50: 32 years from 18
- **Assumption**: Constant annual risk (acknowledge this limitation)

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
  - Annual enhanced / Population = Per capita enhancement risk
  - Express per 1,000 population
- **Annual risk of mandatory minimums**:
  - People with 3+ arrests / Years of data = Annual mandatory
  - Annual mandatory / Population = Per capita mandatory risk
  - Express per 1,000 population
- **By neighborhood and demographics**:
  - Calculate for each policing category
  - Break down by race when demographic data available

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

### Step 13: Obtain Census Demographics
- **Data source**: American Community Survey (ACS) 5-year estimates
- **Geographic level**: Block group (matches arrest data)
- **Key tables needed**:
  - B01001: Sex by Age (total population by age/gender)
  - B02001: Race (racial composition)
  - B03002: Hispanic or Latino Origin by Race
- **Download**: Use Census API or IPUMS NHGIS

### Step 14: Estimate Young Male Population
- **Age brackets from census**: Usually 15-19, 20-24, 25-29, 30-34
- **Interpolation needed**: 
  - 18-19 portion of 15-19 bracket (assume 40%)
  - Full 20-24, 25-29, 30-34 brackets
  - 35 portion of 35-39 bracket (assume 20%)
- **By block group**: Sum male population ages 18-35
- **Aggregate**: Sum by policing category

### Step 15: Estimate Racial Demographics
- **Challenge**: Census race data not cross-tabulated with age/gender at block group level
- **Two approaches**:

  a) **Proportional Method**:
     - Get overall racial percentages for block group
     - Apply to young male population estimate
     - Assumption: Racial distribution consistent across age groups
  
  b) **Arrest Proxy Method** (if census incomplete):
     - Use racial distribution of arrests as proxy
     - Calculate by policing category
     - Acknowledge circular reasoning limitation

### Step 16: Calculate Race-Specific Risks
- **For each race × neighborhood combination**:
  - Numerator: Unique individuals of that race arrested
  - Denominator: Estimated population of that race
  - Annual risk: (Numerator / Years) / Denominator × 100
- **Disparity ratios**: 
  - Within neighborhood: Black risk / White risk
  - Across neighborhoods: Black ultra-policed / White normal
  - Document all disparities found

### Step 17: Calculate Race-Specific Escalation Risks
- **Per capita enhancement risk by race**:
  - Black with 2+ arrests / Black population × 1,000
  - White with 2+ arrests / White population × 1,000
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
  - Unique people with drug arrests / Population × 1,000
  - Annual risk = Above / Years of data
- **By neighborhood category**:
  - Ultra-Policed: X per 1,000 annually
  - Highly Policed: Y per 1,000 annually
  - Normally Policed: Z per 1,000 annually
- **Calculate disparities**:
  - Ultra vs Normal ratio
  - Document fold differences

### Step 20: Calculate Drug Arrest Annual Risks - Young Men
- **Filter**: Age 18-35, Gender = Male, Drug arrests only
- **Metrics**:
  - Unique young men with drug arrests
  - Total drug arrests among young men
  - Average drug arrests per person
- **Annual risk calculation**:
  - Annual unique arrested for drugs / Young male population × 100
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
  - People with 2+ drug arrests / Population × 1,000
  - Break down by neighborhood category
- **Mandatory minimum risk**:
  - People with 3+ drug arrests / Population × 1,000
  - Compare across categories
- **Young men specific**:
  - Same calculations for 18-35 male population
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
  - Sum of drug sentences per 1,000 population
  - By neighborhood category
  - Show multiplicative effect of repeat arrests + escalation

## Phase 6: Population Impact Analysis

### Step 27: Calculate Community-Wide Burden
- **Total arrests per day**: By neighborhood type
- **Families affected annually**:
  - Assume average family size (2.5 adults)
  - P(family member arrested) = 1 - (1 - individual_risk)^family_size
- **Economic impact**:
  - Average cost per arrest (bail, fines, legal fees)
  - Lost wages from incarceration
  - Multiply by per capita arrest rate

### Step 28: Project Long-Term Population Effects
- **Incarceration exposure**:
  - Total jail/prison days per 1,000 population
  - By neighborhood and demographics
  - Account for sentence escalation
- **Criminal record accumulation**:
  - % with any record by age 35
  - % with felony convictions
  - % facing employment barriers

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

### Step 30: Validate Population Estimates
- **Sum check**: Total estimated population should match census total
- **Proportion check**: Young men typically 15-25% of population
- **Racial composition**: Should align with known community demographics
- **Document uncertainties**: Note where estimates required assumptions

### Step 31: Validate Escalation Patterns
- **Internal consistency**:
  - People with 3 arrests ⊂ People with 2 arrests
  - Enhancement % should increase with arrest count
- **External validation**:
  - Compare conviction rates to court statistics
  - Check sentence lengths against guidelines
  - Verify repeat rates against recidivism studies

### Step 32: Calculate Confidence Bounds
- **Small area problem**: Some block groups have small populations
- **Approach**: Use binomial confidence intervals for arrest rates
- **Rule of thumb**: Require minimum 30 arrests for reliable estimates
- **Aggregate if needed**: Combine similar small block groups

### Step 33: Sensitivity Analysis
- **Test different cut points**: ±1-2 percentage points
- **Test demographic assumptions**: ±5% on age/race estimates  
- **Test escalation assumptions**: ±10% on repeat rates
- **Document stability**: How much do results change?
- **Key finding preservation**: Ensure main disparities robust to assumptions

## Key Methodological Notes

### Critical Corrections
1. **Use unique individuals via DefendantId, not arrest counts**
   - Prevents ~3.5x overestimation of risk
2. **Base categories on discretionary arrests only**
   - Isolates police decision-making from crime reports
3. **Track individual arrest sequences**
   - Enables escalation and repeat offense analysis
4. **Linear scales in visualizations**
   - Avoid log scales that obscure disparities

### Important Metrics for Multiple Arrests
- **Repeat arrest factor**: Total arrests / Unique individuals
- **Escalation rate**: % facing 2nd, 3rd, 4th+ offense penalties
- **Conditional probability**: P(next arrest | current arrests)
- **Sentence multiplication**: Average sentence by offense number
- **Per capita burden**: Enhanced penalties per 1,000 population

### Drug-Specific Metrics
- **Drug arrest per capita**: Per 1,000 population annually
- **Drug enhancement risk**: % facing 2nd+ offense penalties
- **Drug mandatory risk**: % facing 3rd+ offense penalties
- **Possession vs distribution**: Progression patterns
- **Enforcement probability**: % of assumed users arrested
- **Sentence escalation**: Multiplication factor by offense number

### Documentation Requirements
- **Data vintage**: Year of census data and arrest data
- **Geographic coverage**: Counties/cities included
- **Time period**: Start and end dates of arrest data
- **Population definitions**: Age ranges, geographic boundaries
- **Statutory framework**: Document penalty enhancement laws
- **Drug law specifics**: Enhancement triggers and mandatory minimums
- **Assumptions made**: List all estimation methods used

This methodology produces robust, replicable analysis of policing intensity patterns, their demographic impacts, the cascading effects of repeat arrests and statutory escalation, with special attention to drug enforcement disparities, while maintaining individual privacy through geographic aggregation.