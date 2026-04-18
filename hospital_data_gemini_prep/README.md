# GEMINI Assessment Prep — GLM & Clinical Data Pipeline Practice

Practice code for the GEMINI Research Data Scientist technical assessment.  
Covers the full analytical pipeline expected in a timed EHR data analysis: cohort construction, data quality validation, descriptive statistics, logistic regression, and visualization.

---

## Repository Structure

```
├── data/
│   └── hospital_data.csv          # Synthetic GEMINI-style dataset (500 encounters)
├── scripts/
│   ├── St01_setup.Rmd             # Package loading, data ingestion, column standardization
│   ├── St02_data_schema.Rmd       # Type coercion, schema review, flag column design
│   ├── St03_data_quality.Rmd      # Validation pipeline: flagging, corrections, attrition table
│   ├── St04_table1.Rmd            # Cohort description (Table 1), stratified by hospital
│   ├── St05_regression.Rmd        # Logistic GLM: model fitting, OR table, diagnostics
│   └── St06_visualization.Rmd     # Forest plot, ROC curve, predicted probabilities, emmeans
├── reference/
│   ├── glm_reference.R            # Self-contained GLM reference script (~100 annotated lines)
│   └── GLM_Logistic_Regression_Notes.md   # Comprehensive concept notes (diagnostics, thresholds, interpretation)
└── README.md
```

---

## Dataset

`hospital_data.csv` is a synthetic dataset modelled on GEMINI hospital administrative (EHR) data. It mimics the structure of real GEMINI encounter-level tables.

| Column | Description |
|--------|-------------|
| `patient_id` | Unique encounter identifier |
| `age` | Patient age (years) |
| `sex` | M / F |
| `hospital` | Site identifier (SiteA–D) |
| `admit_date`, `discharge_date` | Admission and discharge dates |
| `los` | Length of stay (days) |
| `diagnosis` | Diagnosis category (6 levels) |
| `sodium`, `creatinine`, `hemoglobin` | Lab values |
| `hypertension`, `diabetes`, `chf` | Binary comorbidity indicators |
| `disposition` | Discharge disposition (Died / Discharged / etc.) |
| `readmitted_30d` | 30-day readmission indicator |

---

## Pipeline Overview

### 1. Data Quality Validation (`St03`)

Implements a row-level audit pipeline with three error categories:

- **Type/format errors** — invalid or out-of-range values (e.g., negative LOS, sex not in {M, F})
- **Domain/range errors** — physiologically implausible lab values
- **Logical/relational errors** — internally inconsistent records (e.g., `readmitted_30d = TRUE` where `disposition = "Died"`)

Key design principles applied:
- Flagging and value replacement in **separate `mutate()` blocks** to preserve audit trail
- Valid value sets derived with `unique()` at load time — not hardcoded
- Explicit attrition table: starting N → exclusions at each step → final analytic N

### 2. Table 1 (`St04`)

Cohort descriptive statistics stratified by hospital site using the `table1` package.  
Continuous variables: median (IQR). Categorical variables: n (%).

### 3. Logistic Regression (`St05`)

Outcome: in-hospital mortality (`died`).

```r
fit <- glm(died ~ age + sex + charlson + los + hospital,
           data = df, family = binomial(link = "logit"))
```

**Validity diagnostics run:**

| Diagnostic | Test / Function | Threshold |
|------------|----------------|-----------|
| Multicollinearity | `car::vif()` | VIF < 5 (concern), < 10 (critical) |
| Linearity of continuous predictors | `car::boxTidwell()` | p > 0.05 → linearity holds |
| Calibration | `ResourceSelection::hoslem.test()` | p > 0.05 → good fit |
| Influential observations | Cook's distance | > 4/n flags for review |

**Discrimination and fit:**

- AUC via `pROC::roc()` — target ≥ 0.70 for acceptable discrimination
- Pseudo-R² (McFadden, Nagelkerke) via `performance::r2()`
- Odds ratios extracted with `broom::tidy(exponentiate = TRUE, conf.int = TRUE)`

### 4. Visualizations (`St06`)

- **Forest plot** — adjusted ORs with 95% CI (`ggplot2`)
- **ROC curve** — model discrimination (`pROC::ggroc()`)
- **Predicted probability plot** — P(death) vs. age by sex
- **Estimated marginal means** — adjusted P(death) per hospital (`emmeans`)
- **Linearity check** — log-odds vs. continuous predictor (Box-Tidwell visual)

---

## Key Packages

```r
library(tidyverse)          # data wrangling + ggplot2
library(janitor)            # clean_names()
library(lubridate)          # date handling
library(table1)             # Table 1 generation
library(broom)              # tidy model outputs
library(car)                # vif(), boxTidwell()
library(ResourceSelection)  # hoslem.test()
library(pROC)               # roc(), auc(), ggroc()
library(emmeans)            # estimated marginal means
library(performance)        # r2(), check_model()
library(knitr)              # kable()
library(kableExtra)         # kable_styling()
library(patchwork)          # side-by-side plots
```

---

## Reproducing the Analysis

1. Clone the repository and open the `.Rproj` file in RStudio.
2. All scripts use **relative paths** — no `setwd()` required.
3. Data is loaded with `readr::read_csv()`.
4. Knit each `.Rmd` individually in order (St01 → St06), or run the reference script `glm_reference.R` as a standalone end-to-end demo.

---

## Context

This repository was developed as targeted preparation for the [GEMINI](https://www.geminimedicine.ca) Research Data Scientist assessment — a timed (~2 hour) applied analysis task using hospital EHR data. The synthetic data and pipeline structure are modelled on GEMINI's relational data architecture and CIHI coding conventions.

The `reference/GLM_Logistic_Regression_Notes.md` file contains detailed concept notes covering logistic regression theory, diagnostic thresholds, interpretation of odds ratios, and visualization rationale — written as a study reference during prep.
