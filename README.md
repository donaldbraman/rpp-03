# RPP-03: Geographic Policing Intensity Analysis

This repository contains a systematic analysis of policing intensity patterns using anonymized criminal justice data from Charleston and Berkeley Counties, SC.

## Repository Contents

- `docs/methodology_guide.md` - Comprehensive step-by-step methodology for the analysis
- `data/census_mapped_anon_data.parquet` - Anonymized arrest data with census geography
- `src/` - Analysis scripts following the methodology guide
- `figures/` - Generated visualizations
- `results/` - Analysis outputs and reports

## Analysis Overview

This analysis follows a 33-step methodology across 7 phases:

1. **Phase 1**: Data Preparation and Geographic Categorization
2. **Phase 2**: Calculate Annual Arrest Risks
3. **Phase 3**: Multiple Arrest and Escalation Analysis
4. **Phase 4**: Demographic Analysis
5. **Phase 5**: Drug Offense Deep Dive
6. **Phase 6**: Population Impact Analysis
7. **Phase 7**: Quality Checks and Validation

## Key Features

- Uses discretionary arrests to identify policing intensity patterns
- Calculates per capita risks accounting for unique individuals (not arrest counts)
- Analyzes statutory escalation and repeat offense patterns
- Examines drug enforcement disparities under equal use assumptions
- Projects lifetime risks and community impacts

## Data

The anonymized dataset contains:
- 144,645 arrests
- 41,807 unique individuals
- 376 census block groups
- 5-year time period

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- geopandas (for geographic analysis)
- scikit-learn (for clustering)

## Usage

Follow the methodology guide in `docs/methodology_guide.md` step by step to reproduce the analysis.