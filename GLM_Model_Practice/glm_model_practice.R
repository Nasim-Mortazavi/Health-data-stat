# =============================================================================
# GLM Logistic Regression — GEMINI-Style Reference
# Outcome: in-hospital mortality (died = 1/0)
# =============================================================================

library(tidyverse)
library(broom)        # tidy model outputs
library(car)          # vif()
library(ResourceSelection) # Hosmer-Lemeshow test
library(pROC)         # AUC/ROC
library(emmeans)      # estimated marginal means (yes, works for glm too)
library(performance)  # check_model(), r2()
library(knitr)
library(kableExtra)
library(patchwork)
library(car)

# --- 1. SIMULATE GEMINI-LIKE DATA ----
set.seed(42)
n <- 2000
df <- tibble(
  patient_id  = 1:n,
  age         = round(rnorm(n, 72, 12)),
  sex         = factor(sample(c("M", "F"), n, replace = TRUE)),
  hospital    = factor(sample(paste0("H", 1:5), n, replace = TRUE)),
  charlson    = rpois(n, lambda = 2),          # comorbidity score
  los         = pmax(1, round(rnorm(n, 7, 4))) # length of stay (days)
)

# Generate outcome with known log-odds structure
logit_p <- -4 + 0.04 * df$age + 0.3 * df$charlson +
  0.5 * (df$sex == "M") - 0.02 * df$los
df$died <- rbinom(n, 1, plogis(logit_p))

# --- 2. THE GLM FORMULA ----
# logit(P(died=1)) = β0 + β1*age + β2*sex + β3*charlson + β4*los + β5*hospital
#
# glm() with family = binomial(link = "logit") estimates β coefficients
# on the LOG-ODDS scale. Exponentiate to get ODDS RATIOS.

fit <- glm(died ~ age + sex + charlson + los + hospital,
           data = df, family = binomial(link = "logit"))

summary(fit)

# --- 3. ODDS RATIOS + 95% CI ----
# This is what you report in a manuscript table
or_table <- tidy(fit, conf.int = TRUE, exponentiate = TRUE) %>%
  select(term, estimate, conf.low, conf.high, p.value) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

or_table %>%
  filter(term != "(Intercept)") %>%
  mutate(
    sig = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE            ~ ""
    ),
    p.value = ifelse(p.value < 0.001, "< 0.001", round(p.value, 3))
  ) %>%
  kable(col.names = c("Term", "OR", "Lower 95%", "Upper 95%", "p-value", "")) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                full_width = FALSE) %>%
  
  #footnote(general = "* p<0.05, ** p<0.01, *** p<0.001. Reference: Female, Hospital H1")
#footnote(general = "Reference: Female, Hospital H1") from kableExtra
# estimate = odds ratio; e.g., OR = 1.04 for age means
# each 1-year increase → 4% higher odds of death, holding others constant

# --- 4. MODEL VALIDITY CHECKS ----

# 4a. MULTICOLLINEARITY — VIF > 5 is concerning, > 10 is serious
# for factors, gives GVIF^(1/(2*Df))
car::vif(fit)

#"VIF values were ≤ 1.01 for all predictors, indicating no multicollinearity.
# 4b. LINEARITY OF CONTINUOUS PREDICTORS ON LOGIT SCALE
# Plot empirical logits — the most important assumption people skip

#ntile({{ var }}, bins) — takes your continuous variable (e.g., age) and chops it into 10 equal-sized groups.
#So bin 1 is the youngest 10% of patients, bin 10 is the oldest 10%

check_linearity_logit <- function(data, var, outcome, bins = 10) {
  data %>%
    mutate(bin = ntile({{ var }}, bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_x  = mean({{ var }}),
      prop    = mean({{ outcome }}),
      logit_p = log(prop / (1 - prop)),  # empirical logit
      .groups = "drop"
    ) %>%
    ggplot(aes(mean_x, logit_p)) +
    geom_point() + geom_smooth(method = "loess", se = TRUE) +
    labs(title = paste("Linearity check:", deparse(substitute(var))),
         x = deparse(substitute(var)), y = "Empirical logit")
}

p_age <- check_linearity_logit(df, age, died)
p_charlson <- check_linearity_logit(df, charlson, died)
p_age + p_charlson
df <- df %>%
  mutate(
    age_pos      = age,
    charlson_pos = charlson + 1
  )

boxTidwell(died ~ age_pos + charlson_pos, data = df)
#"Box-Tidwell tests indicated no significant departure from linearity on the logit scale for age (p = 0.65) 
# or Charlson score (p = 0.59), supporting the use of linear terms in the logistic model."

# If the loess curve bends → consider splines or categorization

# 4c. HOSMER-LEMESHOW GOODNESS-OF-FIT
# H0: model fits well. p < 0.05 → evidence of poor fit.
hl_test <- hoslem.test(fit$y, fitted(fit), g = 10)
print(hl_test)

# 4d. INFLUENTIAL OBSERVATIONS — Cook's distance
plot(fit, which = 4)  # points above 4/n threshold need inspection
# "Cook's distance values were all below 0.003, well under conventional thresholds, indicating no influential observations."

# --- 5. DISCRIMINATION & PSEUDO-R² ----

# 5a. AUC / ROC — discrimination: can the model separate died vs. survived?
roc_obj <- roc(df$died, fitted(fit))
auc_val <- auc(roc_obj)
cat("AUC:", round(auc_val, 3), "\n")  # > 0.7 acceptable, > 0.8 good

# 5b. PSEUDO R² — there is no true R² for logistic regression
# Report McFadden's or Nagelkerke's, but always label which one
r2_vals <- r2(fit)  # from performance package
print(r2_vals)       # reports Tjur's R² by default for binomial

# Manual McFadden's:
null_fit <- glm(died ~ 1, data = df, family = binomial)
mcfadden_r2 <- 1 - (logLik(fit) / logLik(null_fit))
cat("McFadden R²:", round(as.numeric(mcfadden_r2), 3), "\n")

# --- 6. ESTIMATED MARGINAL MEANS (emmeans) ----
# emmeans works with glm — gives predicted probabilities on response scale
# This is analogous to what you do with glmer / mixed models

# Predicted probability of death by sex, averaged over other covariates
em_sex <- emmeans(fit, ~ sex, type = "response")  # type = "response" → probabilities
print(em_sex)

# Pairwise contrasts on the odds-ratio scale
pairs(em_sex, type = "response")
plot(em_sex) +
  labs(x = "Predicted P(Death)", y = "Sex",
       title = "Adjusted Mortality by Sex") +
  theme_minimal()
# Predicted probability by hospital
em_hosp <- emmeans(fit, ~ hospital, type = "response")
print(em_hosp)
pairs(em_hosp, type = "response", adjust = "tukey")

# --- 7. VISUALIZATION ----

# 7a. FOREST PLOT of odds ratios — the standard manuscript figure
or_plot_data <- or_table %>% filter(term != "(Intercept)")
or_plot_data <- or_plot_data %>%
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

p_forest <- ggplot(or_plot_data, aes(x = estimate, y = reorder(term, estimate))) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  labs(x = "Odds Ratio (95% CI)", y = NULL,
       title = "Adjusted Odds of In-Hospital Mortality") +
  theme_minimal()

print(p_forest)

# 7b. ROC CURVE
ggroc(roc_obj) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "red") +
  labs(title = paste0("ROC Curve (AUC = ", round(auc_val, 3), ")")) +
  theme_minimal()
#"The ROC curve (AUC = 0.69) indicated limited discriminative ability, 
#consistent with a model based on demographic and comorbidity variables without laboratory or physiological data."
# 7c. PREDICTED PROBABILITY BY AGE — useful for clinical communication
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
# 7d. emmeans PLOT — hospital-level predicted probabilities
em_hosp_df <- as.data.frame(em_hosp)

ggplot(em_hosp_df, aes(x = prob, y = hospital)) +
  geom_pointrange(aes(xmin = asymp.LCL, xmax = asymp.UCL)) +
  labs(x = "Predicted P(Death)", y = "Hospital",
       title = "Adjusted Mortality by Hospital") +
  theme_minimal()