# GLM Logistic Regression — Comprehensive Reference Notes
## From GEMINI Assessment Prep Sessions

---

## 1. WHY LOGISTIC REGRESSION?

The outcome in most hospital-based studies is binary: died vs. survived, readmitted vs. not,
complication vs. none. Logistic regression models the probability of that binary outcome via
the logit link function.

Why it dominates over alternatives in clinical research:
- Produces **odds ratios (ORs)**, the standard effect measure in observational epidemiology
- ORs are interpretable to clinician audiences
- Handles confounding adjustment naturally through covariate inclusion
- Unlike classification trees, you get interpretable, adjusted effect estimates

**The formula:**

    logit(P(Y=1)) = β0 + β1·X1 + β2·X2 + ... + βk·Xk

    where logit(p) = log(p / (1-p))

- glm() estimates β on the **log-odds scale**
- Exponentiate to get odds ratios: exp(β) = OR
- Example: exp(0.04) = 1.04 means each 1-unit increase in X → 4% higher odds of outcome

**R code:**

    fit <- glm(died ~ age + sex + charlson + los + hospital,
               data = df, family = binomial(link = "logit"))


---

## 2. WHAT YOU DO NOT NEED TO CHECK

**Common misconception:** you do NOT need to check normality of predictors before logistic
regression. There is no distributional assumption on independent variables. This is different
from OLS where you check normality of residuals (and even that is mainly about inference
in small samples).


---

## 3. WHAT YOU ACTUALLY CHECK (5 VALIDITY DIAGNOSTICS)

### 3a. MULTICOLLINEARITY — VIF

**What:** Variance Inflation Factor measures how much the variance of a coefficient is
inflated because that predictor is correlated with other predictors.

**Why it matters:** If two predictors are highly correlated (e.g., age and Charlson), the model
struggles to separate their individual effects. Standard errors inflate, p-values become
unreliable, ORs become unstable.

**Thresholds:**
- VIF = 1 → no correlation, no problem
- VIF 1–5 → acceptable
- VIF 5–10 → concerning
- VIF > 10 → serious, must act

**For factors** (like hospital with 5 levels), R reports GVIF^(1/(2*Df)) instead of raw VIF.
Square it to compare against standard thresholds.

**Code:**

    library(car)
    vif(fit)

**Common GEMINI trap:** Including BOTH total Charlson score AND individual comorbidity flags
(diabetes, CHF, COPD, etc.) that compose it. Charlson is literally the sum of those flags —
putting both in the model creates perfect or near-perfect collinearity. VIF explodes.

**Rule:** Pick ONE strategy:
- Charlson score alone → when you just need comorbidity adjustment
- Individual flags alone → when the research question is "which comorbidities matter?"
- NEVER both

**What to do if VIF is high:**
- Check for redundant variables (the Charlson trap above)
- Drop one of the collinear pair
- Combine into a composite

**Write-up template:**
*"Variance inflation factors were below [X] for all predictors, indicating no evidence
of multicollinearity."*


### 3b. LINEARITY OF CONTINUOUS PREDICTORS ON THE LOGIT SCALE

**What:** The most important assumption people skip. Logistic regression assumes each
continuous predictor has a LINEAR relationship with the log-odds of the outcome.

**Why it matters:** If age has a U-shaped relationship with mortality, forcing it into a linear
term gives biased, misleading estimates.

**Visual check — Empirical logit plot:**
1. Bin the continuous variable into deciles (ntile)
2. In each bin, compute the proportion with the outcome
3. Transform to logit: log(p / (1-p))
4. Plot against the bin mean
5. Fit a loess smoother — if it's roughly straight, assumption holds

**Code:**

    check_linearity_logit <- function(data, var, outcome, bins = 10) {
      data %>%
        mutate(bin = ntile({{ var }}, bins)) %>%
        group_by(bin) %>%
        summarise(
          mean_x  = mean({{ var }}),
          prop    = mean({{ outcome }}),
          logit_p = log(prop / (1 - prop)),
          .groups = "drop"
        ) %>%
        ggplot(aes(mean_x, logit_p)) +
        geom_point() + geom_smooth(method = "loess", se = TRUE) +
        labs(title = paste("Linearity check:", deparse(substitute(var))),
             x = deparse(substitute(var)), y = "Empirical logit")
    }

    p_age <- check_linearity_logit(df, age, died)
    p_age  # MUST call to display — assigning to variable does not render

**Key syntax notes:**
- {{ }} is "curly-curly" / tidy evaluation — lets you pass bare column names into custom
  functions that use dplyr verbs
- deparse(substitute(var)) converts bare column name to string for plot title
- Use se = TRUE to show confidence ribbon (not just bare loess line)
- Use library(patchwork) then p_age + p_charlson for side-by-side display

**Formal statistical test — Box-Tidwell:**

Tests whether adding x * log(x) significantly improves the model.
H0: λ = 1 (linear relationship). Non-significant p → linearity holds.

    library(car)
    boxTidwell(died ~ age_pos + charlson_pos, data = df)

**Box-Tidwell gotcha:** Only works on strictly POSITIVE continuous variables. If any variable
has zeros (Charlson often does), add +1 before testing:

    df <- df %>% mutate(charlson_pos = charlson + 1)

The +1 shift doesn't change the linearity conclusion. Use original variables in your glm(),
this is only for the diagnostic.

**What to look for in results:**
- MLE of lambda close to 1 and p > 0.05 → linearity holds
- Joint F-test p > 0.05 → no evidence of non-linearity for any variable

**If linearity is violated:**
- Use restricted cubic splines: rcs(age, 4) from the rms package
- Or categorize the variable (less preferred — loses information)

**For the assessment, do BOTH** visual (loess with se=TRUE) and formal (Box-Tidwell).
Plot shows you checked, p-value gives the reviewer a number.

**Write-up template:**
*"Empirical logit plots and Box-Tidwell tests confirmed approximately linear relationships
between age (p = X.XX) and Charlson score (p = X.XX) on the logit scale, supporting the
use of linear terms in the logistic model."*


### 3c. HOSMER-LEMESHOW GOODNESS-OF-FIT

**What:** Tests whether predicted probabilities from the model match actual observed outcomes
across the range of risk.

**How it works:**
1. Sort all patients by predicted probability of death
2. Split into g = 10 equal groups (deciles of risk)
3. Compare expected vs. actual deaths in each group
4. Chi-squared test on that table

**Key concept:** The null hypothesis is that the model fits well. You WANT a non-significant
result. This is the opposite of most tests.

- p > 0.05 → no evidence of poor fit → model is adequate
- p < 0.05 → predicted and observed don't match → model is miscalibrated

**Code:**

    library(ResourceSelection)
    hl_test <- hoslem.test(fit$y, fitted(fit), g = 10)
    print(hl_test)

**Important caveats:**
- g = 10 is standard. Don't change it unless you have a reason.
- Sensitive to sample size. With large GEMINI datasets (50k+), it almost always rejects
  because even trivial miscalibration becomes significant. If this happens, report it and
  note large sample size as a limitation.
- df = g - 2. So g = 10 → df = 8. Report df because the same χ² means different things
  at different df.

**Why report df:** It tells the reader how many groups you used and makes the result
reproducible. The p-value depends on both χ² and df together.

**Write-up template:**
*"The Hosmer-Lemeshow test indicated adequate model fit (χ² = X.XX, df = 8, p = X.XX)."*


### 3d. INFLUENTIAL OBSERVATIONS — COOK'S DISTANCE

**What:** Identifies individual observations that disproportionately influence the model
coefficients. High Cook's distance means removing that one patient substantially changes
your ORs.

**Code:**

    plot(fit, which = 4)

**Thresholds:**
- Common rule: 4/n (with n=2000, that's 0.002)
- Practical: values below 0.5 are harmless
- Values above 1.0 truly warrant investigation

**What a problem looks like:** One or two points towering far above everything else with
Cook's distance > 0.5. That means a single patient is dragging the model.

**What to do if flagged:**
- Inspect that row — is it a data entry error?
- Refit without it and compare ORs — did they change meaningfully?
- If yes, report sensitivity analysis with and without

**Write-up template:**
*"Cook's distance values were all below [X], well under conventional thresholds,
indicating no influential observations."*


### 3e. SEPARATION / SPARSITY (check BEFORE fitting)

**What:** If a predictor perfectly predicts the outcome (e.g., all patients with diagnosis X
died), maximum likelihood estimation does not converge. Coefficient goes to infinity.

**How to check:** Cross-tabulate predictors with outcome before fitting. Look for cells with
zero counts.

**Fixes:**
- Firth's penalized likelihood regression
- Exact logistic regression
- Collapse rare categories

**Also check:** Events per variable (EPV). Classic rule of thumb: ~10 events per predictor
variable. With large GEMINI datasets this is usually fine, but in subgroup analyses it
can bite.


---

## 4. DISCRIMINATION & PSEUDO-R²

### 4a. AUC / ROC CURVE

**What AUC measures:** Discrimination — the model's ability to rank patients correctly.
If you pick one patient who died and one who survived at random, how often does the model
assign the correct higher probability to the one who died?

**Thresholds:**
- AUC = 0.5 → coin flip, useless
- AUC 0.5–0.7 → poor
- AUC 0.7–0.8 → acceptable
- AUC 0.8–0.9 → good
- AUC > 0.9 → excellent (rare in clinical research)

**Code:**

    library(pROC)
    roc_obj <- roc(df$died, fitted(fit))
    auc_val <- auc(roc_obj)

    ggroc(roc_obj) +
      geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "red") +
      labs(title = paste0("ROC Curve (AUC = ", round(auc_val, 3), ")")) +
      theme_minimal()

**How to read the ROC curve:**
- Diagonal line = AUC 0.5 (coin flip). Always add this reference line.
- The more the curve hugs the TOP-LEFT corner, the better.
- Y-axis (Sensitivity): of all who died, proportion correctly flagged (true positive rate)
- X-axis (Specificity, reversed): of all who survived, proportion correctly identified

**If AUC is low on the assessment:** Don't try to inflate it by overfitting. Report it
honestly, explain why (e.g., "limited to demographic and comorbidity variables without
laboratory or physiological data"), and suggest next steps.


### 4b. PSEUDO-R²

**Key concept:** There is NO true R² for logistic regression. Never call it "R²" without
"pseudo" in front of it. Always label WHICH pseudo-R² you're reporting.

**Common variants:**
- Tjur's R² — difference in mean predicted probability between events and non-events.
  Intuitive. performance::r2() reports this by default for binomial.
- McFadden's R² — 1 - (logLik(model) / logLik(null)). Analogous to proportion of
  log-likelihood explained.

**Code:**

    library(performance)
    r2(fit)  # Tjur's by default

    # Manual McFadden's:
    null_fit <- glm(died ~ 1, data = df, family = binomial)
    mcfadden_r2 <- 1 - (logLik(fit) / logLik(null_fit))

**Always pair AUC with pseudo-R².** AUC measures discrimination (ranking), pseudo-R² measures
explained variation. They tell different stories.

**Write-up template:**
*"The model demonstrated limited discrimination (AUC = 0.69) and low explanatory power
(Tjur's R² = 0.11), likely reflecting the absence of key clinical predictors such as
laboratory values and acuity measures."*


---

## 5. ODDS RATIO TABLE

### Extracting and presenting ORs:

    library(broom)
    or_table <- tidy(fit, conf.int = TRUE, exponentiate = TRUE) %>%
      filter(term != "(Intercept)") %>%
      select(term, estimate, conf.low, conf.high, p.value) %>%
      mutate(across(where(is.numeric), ~ round(.x, 3)))

**Why filter out the intercept:** The exponentiated intercept is not an OR — it's the baseline
odds when all predictors are zero. Clinically meaningless. Remove it before presenting.

### Significance stars:

    or_table <- or_table %>%
      mutate(
        sig = case_when(
          p.value < 0.001 ~ "***",
          p.value < 0.01  ~ "**",
          p.value < 0.05  ~ "*",
          TRUE            ~ ""
        ),
        p.value = ifelse(p.value < 0.001, "< 0.001", round(p.value, 3))
      )

**Never show p = 0.000.** It implies exactly zero, which is impossible. Use "< 0.001".

**Note:** case_match() uses ~ (tilde), not = (equals). recode() is deprecated in newer dplyr.

### Presenting with kable:

    library(knitr)
    library(kableExtra)

    or_table %>%
      kable(col.names = c("Term", "OR", "Lower 95%", "Upper 95%", "p-value", "")) %>%
      kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                    full_width = FALSE) %>%
      footnote(general = "* p<0.05, ** p<0.01, *** p<0.001. Reference: Female, Hospital H1")

**Critical details:**
- Always state the REFERENCE CATEGORY in a footnote. Without it, the reader cannot
  interpret hospital or sex ORs.
- Clean variable labels before presenting: "sexM" → "Male", "hospitalH3" → "Hospital 3"
  using case_match() before piping to kable()
- print() gives raw console output — never use in knitted Rmd. Always use kable().

### Interpreting ORs:

- OR = 1.04 for age → 4% higher odds per 1-year increase
- Per 10-year increase: 1.04^10 = 1.48 (48% higher odds)
- Per 20-year increase: 1.04^20 = 2.19
- Clinical papers typically report per-decade for age since per-year ORs near 1.0 look
  deceptively weak

**Precision vs. clinical significance:** A tiny confidence interval (like age OR 1.03–1.05)
means the estimate is very precise. But precise ≠ important. The per-unit effect is small —
it's the cumulative effect across realistic ranges that matters clinically.

**Why continuous predictors have narrow CIs and categorical have wide:**
- Continuous: every patient contributes a unique value → lots of information per coefficient
- Categorical (e.g., hospital): 2000 patients split into 5 groups (~400 each) → less data
  per estimate → more uncertainty → wider CI


---

## 6. ESTIMATED MARGINAL MEANS (emmeans)

### What it does:

emmeans extracts adjusted predicted probabilities from a fitted model, averaging over
other covariates at their observed distribution. The adjustment comes from the MODEL,
not from the emmeans call.

    fit ← does the adjusting
    emmeans ← extracts interpretable predictions from that adjusted model

### Code:

    library(emmeans)

    # Predicted probability by sex
    em_sex <- emmeans(fit, ~ sex, type = "response")
    print(em_sex)

**Key arguments:**
- type = "response" → gives probabilities, not log-odds. Without it you get the logit
  scale, which is uninterpretable to a clinical audience.
- df = Inf is normal for GLMs — uses asymptotic (z-based) inference.
- "Results are averaged over the levels of: hospital" means probabilities are marginalized
  across all hospitals, not specific to one.

### Pairwise comparisons:

    pairs(em_sex, type = "response")

- For 2 groups (sex): no multiple comparison adjustment needed
- Output gives OR on the response scale
- Direction depends on alphabetical ordering: pairs() computes F/M, while glm() reports
  M/F. They are inverses: 1/0.638 = 1.567. Same result, inverse framing.

### Multiple comparison correction (3+ groups):

    em_hosp <- emmeans(fit, ~ hospital, type = "response")
    pairs(em_hosp, type = "response", adjust = "tukey")

**When to use which adjustment:**
- 2 groups → no adjustment needed (single comparison)
- 3+ groups, all pairwise → adjust = "tukey" (standard default)
- Conservative / non-standard comparisons → adjust = "bonferroni"
- Never leave unadjusted with 3+ groups — with 5 hospitals (10 comparisons), 40% chance
  of at least one false positive at α = 0.05

**Why correction matters:** Each comparison has a 5% false positive rate. With k groups you
have k(k-1)/2 comparisons. Without correction, family-wise error rate inflates rapidly.

**Write-up template (sex):**
*"After adjustment for age, Charlson score, length of stay, and hospital, males had a
higher predicted probability of in-hospital mortality (46.2%, 95% CI: 42.9–49.5%)
compared to females (35.4%, 95% CI: 32.4–38.6%)."*

**Write-up template (hospital):**
*"Pairwise comparisons of adjusted mortality across hospitals, with Tukey correction for
multiple comparisons, revealed no statistically significant differences (all p > 0.43).
Odds ratios ranged from 0.77 (H3 vs H4) to 1.26 (H1 vs H3), indicating relatively
consistent outcomes across sites."*

**Reporting non-significant results:** Non-significant hospital variation IS a meaningful
finding for GEMINI, where hospital-level quality variation is a central research question.
Never ignore non-significant results — state them explicitly.


---

## 7. VISUALIZATIONS — WHAT, WHY, AND HOW

### 7a. FOREST PLOT (OR table as a figure)

**Purpose:** Visual summary of adjusted odds ratios. The standard manuscript figure for
logistic regression results.

**How to read:** Any CI crossing the dashed line at OR = 1 is non-significant. Effects
entirely to the right of 1 increase odds; entirely to the left decrease odds.

**Code:**

    or_plot_data <- or_table %>%
      filter(term != "(Intercept)") %>%
      mutate(term = case_match(term,
        "sexM"       ~ "Male",
        "charlson"   ~ "Charlson Score",
        "age"        ~ "Age (per year)",
        "los"        ~ "Length of Stay",
        "hospitalH2" ~ "Hospital 2",
        "hospitalH3" ~ "Hospital 3",
        "hospitalH4" ~ "Hospital 4",
        "hospitalH5" ~ "Hospital 5"
      ))

    ggplot(or_plot_data, aes(x = estimate, y = reorder(term, estimate))) +
      geom_point(size = 3) +
      geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
      labs(x = "Odds Ratio (95% CI)", y = NULL,
           title = "Adjusted Odds of In-Hospital Mortality") +
      theme_minimal()

**Polish tips:**
- Clean variable labels (case_match, not recode — recode is deprecated)
- Consider separating patient-level and hospital-level terms with a gap or group label
- Reference line at OR = 1 is mandatory


### 7b. ROC CURVE

**Purpose:** Visualize model discrimination — how well the model separates outcomes.

**Must include:** Diagonal reference line (coin flip). Without it, the reader cannot gauge
performance visually.

    ggroc(roc_obj) +
      geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "red") +
      labs(title = paste0("ROC Curve (AUC = ", round(auc_val, 3), ")")) +
      theme_minimal()


### 7c. PREDICTED PROBABILITY CURVE

**Purpose:** Most clinically intuitive plot. Shows how predicted probability of outcome
changes across a continuous predictor, holding everything else constant.

**Critical rules:**
- Derive age range from actual data: seq(min(df$age), max(df$age)) — never hardcode
  arbitrary values. Predicting outside your data range is EXTRAPOLATION.
- Add confidence ribbon — a bare line without uncertainty looks incomplete
- Compute CIs on the LINK (logit) scale, then back-transform with plogis(). This ensures
  they stay bounded between 0 and 1.
- State held-constant values in subtitle, not crammed into the title

**Full code with CI ribbon and sex overlay:**

    newdata <- expand.grid(
      age      = seq(min(df$age), max(df$age), by = 1),
      sex      = c("M", "F"),
      charlson = 2,
      los      = 7,
      hospital = "H1"
    )

    preds <- predict(fit, newdata, type = "link", se.fit = TRUE)
    newdata$pred  <- plogis(preds$fit)
    newdata$lower <- plogis(preds$fit - 1.96 * preds$se.fit)
    newdata$upper <- plogis(preds$fit + 1.96 * preds$se.fit)

    ggplot(newdata, aes(age, pred, color = sex, fill = sex)) +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
      geom_line(linewidth = 1) +
      labs(x = "Age", y = "Predicted P(Death)",
           title = "Predicted Mortality by Age and Sex",
           subtitle = "Charlson = 2, LOS = 7, Hospital 1",
           color = "Sex", fill = "Sex") +
      theme_minimal()

**Why the S-shape:** The logistic function maps the linear predictor (which goes from -∞ to
+∞) to probabilities (bounded 0 to 1). The curve is steepest where predicted probability
is near 0.5 and flattens at the extremes.


### 7d. EMMEANS PLOT

**Purpose:** Compare adjusted predicted probabilities across levels of a categorical variable.

**Better version using geom_pointrange** (cleaner than default emmeans plot):

    em_hosp_df <- as.data.frame(em_hosp)

    ggplot(em_hosp_df, aes(x = prob, y = hospital)) +
      geom_pointrange(aes(xmin = asymp.LCL, xmax = asymp.UCL)) +
      labs(x = "Predicted P(Death)", y = "Hospital",
           title = "Adjusted Mortality by Hospital") +
      theme_minimal()


### 7e. LINEARITY CHECK PLOTS (Empirical Logit)

**Purpose:** Diagnostic — verify linearity assumption before trusting model results.

**How to read:** Loess smoother should be roughly straight. Curves, U-shapes, or plateaus
indicate violated assumption.

**Use se = TRUE** to show confidence ribbon — a bare loess line without uncertainty is
not convincing for a reviewer.

**Side-by-side display:**

    library(patchwork)
    p_age + p_charlson


---

## 8. CLUSTERING — WHEN TO UPGRADE FROM GLM TO GLMER

If patients are nested within hospitals (which they are in GEMINI), a plain glm() ignores
clustering. This underestimates standard errors — your p-values are too small.

**Options:**
- Clustered standard errors (sandwich estimator)
- GEE (generalized estimating equations)
- Mixed-effects logistic regression: glmer(died ~ age + sex + charlson + los + (1|hospital),
  data = df, family = binomial) from lme4

**When to use which:**
- Hospital is a nuisance variable you're adjusting for → fixed effect in glm() is fine
  (what we did)
- Hospital-level variation is of scientific interest → random effect in glmer()
- You want population-average estimates → GEE
- You want cluster-specific estimates → glmer()

The emmeans workflow is identical for glmer — just point it at the fitted glmer object.


---

## 9. COMPLETE ANALYTIC WORKFLOW ORDER

1. Data import & inspection
2. Data quality & missingness assessment (check by hospital — systematic?)
3. Descriptive / cohort characterization (Table 1)
4. Variable derivation & preprocessing
5. Check separation / sparsity (cross-tabs) and EPV
6. Fit the model: glm()
7. Diagnostics: VIF → Linearity → Hosmer-Lemeshow → Cook's distance
8. Discrimination: AUC/ROC + Pseudo-R²
9. Estimated marginal means + pairwise comparisons
10. Visualization: forest plot, ROC, predicted probability, emmeans plot
11. Interpretation & write-up


---

## 10. QUICK SYNTAX REMINDERS

- %>% at end of line → pipes into next function. Last function in chain has no pipe.
- {{ }} → tidy evaluation for passing bare column names into custom functions
- case_match() uses ~ (tilde). recode() is deprecated.
- Assigning ggplot to variable does NOT display it. Must call p_age or print(p_age).
- type = "response" in emmeans and predict → probabilities, not log-odds
- type = "link" in predict → log-odds scale (needed for proper CI computation)
- plogis() → inverse logit, converts log-odds to probability
- kable() + kable_styling() for HTML tables in Rmd. Never use print() in knitted output.
- expand.grid() creates all combinations of specified values for prediction


---

## 11. LIBRARIES USED

    library(tidyverse)          # data manipulation + ggplot2
    library(broom)              # tidy() for model outputs
    library(car)                # vif(), boxTidwell()
    library(ResourceSelection)  # hoslem.test()
    library(pROC)               # roc(), auc(), ggroc()
    library(emmeans)            # estimated marginal means + pairwise comparisons
    library(performance)        # r2(), check_model()
    library(knitr)              # kable()
    library(kableExtra)         # kable_styling(), footnote()
    library(patchwork)          # side-by-side plots with +
