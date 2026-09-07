# Health-data-stat

Applied statistics on hospital administrative data, in R.

Two independent pieces of work. Both use synthetic data. Neither reports findings about real patients.

## Contents

| Folder | What it is |
|--------|------------|
| `hospital_data_gemini_prep/` | An end to end cohort analysis pipeline on synthetic GEMINI-structured hospital data: cohort construction with an attrition table, data quality checks, ICD-10-CA comorbidity flags, CIHI discharge disposition coding, early lab extraction, logistic regression with full diagnostics. |
| `GLM_Model_Practice/` | A self contained logistic regression reference script on simulated data, plus written notes on diagnostics, odds ratio interpretation and threshold selection. Practice scale. |

## Data

All data in this repository is synthetic. No real patient records are present. See `hospital_data_gemini_prep/README.md` for the source, licence and attribution.

## Requirements

R 4.x with `tidyverse`, `lubridate`, `table1`, `broom`, `janitor`, `knitr`, `kableExtra`, `pROC`, `car`, `scales`, `pheatmap`, `RColorBrewer`. Install with `pacman::p_load()` as each document does at the top.

## Author

Nasim Mortazavi. github.com/Nasim-Mortazavi
