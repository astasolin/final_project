# Gameplan — Predicting Voter Turnout in Europe

**Course:** Machine Learning and Deep Learning, Spring 2026 (CBS)
**Author:** Maria Bitner
**Dataset:** European Social Survey, Round 11 (ESS11e04_1.csv) — 50,116 respondents, 30 European countries
**Target variable:** `vote` — binary (1 = voted in last national election, 0 = did not vote)

---

## Project files (multi-notebook structure)

```
gameplan.md                  ← this file (master plan)
gameplan.ipynb               ← high-level scaffold (one notebook overview)
01_eda.ipynb                 ← variable-by-variable EDA (DONE — first build)
02_preprocessing.ipynb       ← cleaning + feature engineering (next)
03_modeling.ipynb            ← train all 4 models on Sets A and B
04_results_and_analysis.ipynb ← test metrics + SHAP + per-country errors
ESS11e04_1.csv               ← raw data (50,116 × 25)
ESS11e04_1 codebook.html     ← codebook (open in browser when in doubt)
```

Each notebook has one job, kernels stay snappy, and the oral-exam screen-share is much cleaner.

---

## Two parallel feature sets

Rather than picking 24 vs. 7 raw features arbitrarily, we train every model on **both**:

| Set | What's in it | Purpose |
|---|---|---|
| **A — raw** | 22 cleaned features individually | Maximum predictive power; let permutation importance / SHAP rank features |
| **B — composite** | 15 features. The 4 trust items become `trust_index`, the 3 efficacy items become `efficacy_index`, the 3 satisfaction items become `satisfaction_index`. | Cleaner story, easier oral-exam defence; matches the "7–10 effective predictors" sweet spot when grouped by construct |

Both sets are saved to disk by `02_preprocessing.ipynb` as `feature_set_A.csv` and `feature_set_B.csv`. The Results section reports both side by side and explains the trade-off.

> **Media & news variables dropped.** The original selection included `nwspol` and `netusoft`. A 5-fold CV logistic-regression check showed they add only **0.001** to PR-AUC (well within noise) because `netusoft` is heavily redundant with age/education and `nwspol`'s signal is mediated through `polintr`. Both have been removed from both feature sets. The framework therefore has **6 feature categories** (not 7).

---

## Plotting / reporting transformations

Some variables are kept *raw for the model* but *binned for plots and tables* (binning is easier for the reader, raw preserves variance for the model):

| Variable | Categorical version (for plots) | Used in model as |
|---|---|---|
| `agea` | `age_group` ∈ {18-24, 25-34, 35-49, 50-64, 65+} | raw `agea` |
| `eduyrs` | `edu_level` ∈ {low ≤9, medium 10-14, high ≥15} | raw `eduyrs` |
| `hinctnta` | `income_tertile` ∈ {low 1-3, middle 4-7, high 8-10} | raw 1-10 ordinal |
| `lrscale` | — | raw `lrscale` **plus** `lr_extreme = abs(lrscale - 5)` |

---

## 0. Working title

**"Who shows up at the ballot box? Predicting voter turnout in 30 European countries using machine learning and deep learning on European Social Survey data."**

---

## 1. Introduction

### 1.1 Motivation

Voter turnout is one of the most studied — and most worried-about — outcomes in modern democracies. Across Europe, turnout has been declining unevenly for decades, fuelling concerns about democratic legitimacy, representational bias, and the rise of populist movements that benefit when disengaged citizens stay home. Understanding *who votes* and *why some don't* matters for political parties (mobilisation strategy), governments (designing inclusive electoral institutions), NGOs (turnout campaigns), and academic political science.

Most of the existing literature on turnout has been written from a classical statistical (logistic regression) perspective, with a handful of recent papers applying machine learning. This project asks whether modern ML and a small deep learning model can outperform a logistic baseline — and, just as important, whether they tell us anything *new* about the relative importance of socio-demographic, economic, attitudinal, and behavioural drivers.

### 1.2 Research questions

**RQ1 (predictive):** How accurately can voter turnout be predicted from individual-level survey responses across 30 European countries, and which model family (linear, tree-based ensemble, feed-forward neural network) performs best?

**RQ2 (explanatory):** Which categories of features — socio-demographic, socio-economic, political attitudes, institutional trust, civic/political engagement, or satisfaction with the system — contribute most to predictive performance?

**RQ3 (cross-national):** Does the addition of country-level information meaningfully improve prediction, and which countries are hardest to classify correctly?

### 1.3 Related work

This section will cover three strands (target ~5–8 citations, found via Google Scholar):

1. **Classical political science on turnout determinants** — the Civic Voluntarism Model (Verba, Schlozman & Brady, 1995); Resource Model of political participation; Almond & Verba's *The Civic Culture* on institutional trust.
2. **Recent ML applications to turnout/voting** — search keywords: *"voter turnout machine learning"*, *"election prediction neural network"*, *"ESS classification"*. Aim for 3–4 papers from 2018–2024.
3. **ML-on-survey-data methodology** — handling sentinel-coded missingness, ordinal Likert features, survey weights, and the SMOTE/ADASYN literature for class imbalance ([J01], [J02] from the lecture plan).

---

## 2. Conceptual Framework

### 2.1 Theoretical anchor

The feature design is anchored in the **Civic Voluntarism Model (CVM)** of Verba, Schlozman & Brady, which explains political participation as a function of *resources*, *engagement*, and *recruitment*. The ESS variables map naturally onto an extended version of this framework, which adds *institutional trust* and *system satisfaction* as moderators well-established in subsequent literature.

### 2.2 Feature taxonomy (refined from initial idea)

The original four categories (socio-demographic, socio-economic, political interest, institutional trust) are kept and **extended with two more groups** that are already in the loaded dataset and theoretically distinct. (A seventh "media & information" group was tested and dropped — see note in §0 above.)

| # | Category | ESS variables | Theoretical role |
|---|---|---|---|
| 1 | **Socio-demographic** | `agea`, `gndr`, `cntry` | Baseline controls; turnout has a strong age gradient |
| 2 | **Socio-economic resources** | `eduyrs`, `hinctnta` | CVM "resources" pillar |
| 3 | **Political interest & ideology** | `polintr`, `lrscale`, `clsprty` | CVM "engagement" pillar |
| 4 | **Institutional trust** | `trstprl`, `trstplt`, `trstprt`, `trstlgl` | Institutional confidence; predictor of system support |
| 5 | **Civic & political engagement** | `contplt`, `sgnptit`, `cptppola`, `psppsgva`, `psppipla` | Behavioural participation + internal/external efficacy |
| 6 | **Satisfaction with system** | `stfdem`, `stfgov`, `stfeco` | Government performance evaluation |
| — | *Survey design weights* | `pspwght`, `anweight` | Used for descriptive stats only, **not** as features |

### 2.3 Problem statement

Given a feature vector **x** = [socio-demographic, socio-economic, political interest, institutional trust, civic engagement, satisfaction] for individual *i*, predict the binary outcome y_i ∈ {0, 1} where 1 = voted in last national election. The task is **supervised binary classification** on imbalanced tabular survey data (≈78 / 22 split after cleaning).

### 2.4 Conceptual diagram (to be drawn for the report)

```
[Socio-demo]    [Socio-econ]
     \               /
      \             /
[Political interest] ─────┐
                          │
[Institutional trust] ────┼──▶  P(vote = 1 | x)
                          │
[Civic engagement] ───────┤
                          │
[Satisfaction] ───────────┘
```

---

## 3. Methodology *(special focus — most detailed section)*

### 3.1 Dataset description

- **Source:** European Social Survey, Round 11 (fieldwork 2023–2024).
- **Unit of observation:** individual adult respondent.
- **Raw rows:** 50,116. **Raw columns selected:** 25 (24 features + target).
- **Coverage:** 30 European countries (AT, BE, BG, CH, CY, DE, EE, ES, FI, FR, GB, GR, HR, HU, IE, IL, IS, IT, LT, LV, ME, NL, NO, PL, PT, RS, SE, SI, SK, UA).
- **Codebook reference:** `ESS11e04_1 codebook.html` (in workspace).

### 3.2 Exploratory Data Analysis (EDA)

Planned EDA outputs (each with figure or table):

1. **Sample composition** — country participation map (already done), age pyramid, gender split, education distribution.
2. **Target distribution** — bar + pie for `vote` (already done); turnout by country; turnout by age band; turnout by education tertile.
3. **Feature distributions** — histograms / value-count bars for every Likert-style variable; flag the prevalence of ESS sentinel codes (66, 77, 88, 99 etc.).
4. **Bivariate associations** — turnout rate by each ordinal feature; point-biserial / Cramér's V correlation with the target.
5. **Multicollinearity check** — Spearman correlation heatmap of all numeric features; pay attention to the 4 trust variables (likely highly correlated → candidates for PCA or composite).
6. **Class imbalance summary** — confirm post-cleaning ratio and decide if SMOTE/ADASYN is needed.

### 3.3 Data preprocessing (variable-by-variable)

A pre-processing decision is documented for **every** feature. Sentinel codes in ESS:

- `66`, `77`, `88`, `99` → typically *Not applicable / Refusal / Don't know / No answer*
- `7`, `8`, `9` → same logic for short Likert scales
- `5555`, `7777`, `8888`, `9999` → for time-in-minutes type questions

**Per-variable plan:**

| Variable | Type | Sentinel codes | Cleaning decision | Encoding |
|---|---|---|---|---|
| `vote` (target) | Binary | 3, 7, 8, 9 | Drop ineligible (3) and non-response (7/8/9). Recode 1→1, 2→0. **Already done.** | Binary 0/1 |
| `agea` | Continuous (yrs) | 999 | Replace 999 → NaN, then median-impute by country | Standardised (z-score) for LR/NN; raw for trees |
| `gndr` | Categorical | 9 | Drop 9 (very rare); recode 1=Male, 2=Female | One-hot (drop_first) |
| `cntry` | Categorical (30 lvls) | none | Keep as-is | One-hot for LR/NN; ordinal/target encoding for trees |
| `eduyrs` | Continuous | 77, 88, 99 | Replace → NaN, median-impute; cap at 99th percentile | Standardised |
| `hinctnta` | Ordinal (income decile, 1–10) | 77, 88, 99 | Replace → NaN, mode-impute by country | Treat as numeric (already ordinal) |
| `trstprl`, `trstplt`, `trstprt`, `trstlgl` | Ordinal 0–10 | 77, 88, 99 | Replace → NaN, median-impute. **Decision point:** keep all four OR build a `trust_index` (mean) + drop the originals after correlation check | Numeric |
| `polintr` | Ordinal 1–4 (reverse: 1 = very interested) | 7, 8, 9 | Replace → NaN, median-impute. **Reverse-code** so higher = more interest | Numeric |
| `lrscale` | Ordinal 0–10 | 77, 88, 99 | Replace → NaN, median-impute. Optionally also create `lr_extreme = abs(lrscale - 5)` | Numeric |
| `cptppola`, `psppsgva`, `psppipla` | Ordinal 1–5 (efficacy) | 7, 8, 9 | Replace → NaN, median-impute | Numeric |
| `contplt`, `sgnptit` | Binary 1/2 (last 12 months) | 7, 8, 9 | Replace → NaN, mode-impute. Recode 1→1 (yes), 2→0 (no) | Binary |
| `clsprty` | Binary 1/2 (close to a party) | 7, 8, 9 | Same as above | Binary |
| `stfdem`, `stfgov`, `stfeco` | Ordinal 0–10 | 77, 88, 99 | Replace → NaN, median-impute | Numeric |
| `nwspol` | Continuous (mins/day news) | 7777, 8888, 9999 | Replace → NaN, median-impute, log1p-transform (heavy right tail) | log1p, then standardise |
| `netusoft` | Ordinal 1–5 (internet use freq) | 7, 8, 9 | Replace → NaN, mode-impute | Numeric |
| `pspwght`, `anweight` | Continuous weights | none | **Do not feed to model.** Use only for descriptive statistics. | — |

**Imputation strategy:** Median for continuous/ordinal, mode for categorical/binary, optionally country-stratified. Keep an `_was_missing` flag column for the high-missing variables (`hinctnta`, `lrscale` typically have non-trivial DK rates) so the model can learn from missingness itself.

### 3.4 Filtering

- Drop ineligible voters and non-respondents on the target (already done in the existing notebook → ~6,500 rows lost, ~46,500 remain).
- Drop respondents with missing on `agea` (rare).
- After all imputations, re-check that no NaNs remain.

### 3.5 Feature engineering (planned)

- `trust_index` = mean(`trstprl`, `trstplt`, `trstprt`, `trstlgl`) — composite, justified by expected high correlation.
- `efficacy_index` = mean of efficacy items (`cptppola`, `psppsgva`, `psppipla`) — internal/external political efficacy.
- `satisfaction_index` = mean(`stfdem`, `stfgov`, `stfeco`).
- `age_group` = bins {18–24, 25–34, 35–49, 50–64, 65+} — useful for stratified analysis even if model uses raw age.
- `lr_extreme` = |`lrscale` − 5| — captures ideological extremity (often more predictive than raw left-right).

We will train **two parallel feature sets** and compare:
- **(A)** All raw features.
- **(B)** Composite indices replacing their components.

### 3.6 Train/validation/test split

- Stratified 70 / 15 / 15 split by `vote` to preserve class ratio.
- Use `random_state=42` everywhere for reproducibility.
- For tree models we will additionally run **5-fold stratified cross-validation** on the train+val portion for hyperparameter tuning.

### 3.7 Class imbalance handling

The post-cleaning split is approximately 78 % voted / 22 % did not vote. We will compare three strategies:

1. **Class weights** (`class_weight='balanced'` in scikit-learn / `pos_weight` in Keras) — preferred default.
2. **SMOTE oversampling** of the minority class on the training set only (per Lecture 7 / [J01]).
3. **ADASYN** as alternative ([J02]).

Resampling is applied **only** to the training fold inside CV — never to validation or test.

### 3.8 Models

Per the project guidelines: max 3 primary models + 1 baseline.

| Role | Model | Library | Why this model |
|---|---|---|---|
| **Baseline** | Logistic Regression (L2) | scikit-learn | Standard in turnout literature; gives interpretable coefficients; serves as the reference |
| **Primary 1** | Random Forest | scikit-learn | Handles mixed types, captures non-linearity, gives feature importances out of the box |
| **Primary 2** | Gradient Boosting (XGBoost or sklearn GradientBoostingClassifier) | xgboost / sklearn | Typically the strongest tabular learner; lecture 5 |
| **Primary 3** | Feed-Forward Neural Network | Keras / TensorFlow | Demonstrates the deep-learning portion of the course (lectures 8–9); 2–3 hidden layers, dropout, early stopping |

### 3.9 Hyperparameter tuning

- **GridSearchCV** for Logistic Regression (C ∈ {0.01, 0.1, 1, 10}, penalty L1/L2/ElasticNet).
- **RandomizedSearchCV** for Random Forest (n_estimators, max_depth, min_samples_leaf) and Gradient Boosting (learning_rate, n_estimators, max_depth, subsample).
- **Manual grid** for FFNN: layer widths {64, 128}, depth {2, 3}, dropout {0.2, 0.4}, learning rate {1e-3, 5e-4}, batch size {128, 256}.
- Selection metric: **PR-AUC** (more informative than ROC-AUC under imbalance), with F1 as a tiebreaker.

### 3.10 Evaluation metrics

Reported on the held-out **test set** for the final selected configuration of each model:

- **Confusion matrix** (TP / TN / FP / FN).
- **Accuracy** (with caveat that the 78/22 baseline already gets 78 % by always predicting "voted").
- **Precision, Recall, F1** for both classes (especially the minority "did not vote" class).
- **ROC-AUC** and **PR-AUC** (Lecture 6).
- **Calibration plot** for the probabilistic models.
- **Brier score** for probabilistic accuracy.

### 3.11 Model complexity & runtime

For each model we record (per project guidelines):

- Number of parameters / nodes / depth.
- Wall-clock training time on the full training set.
- Inference time per sample.
- Memory footprint of the trained model.

This enables a "performance vs. cost" trade-off discussion against the logistic baseline.

### 3.12 Feature importance & interpretability

- **Logistic regression:** standardised coefficients with 95 % CIs.
- **Random Forest / GBM:** Gini importance and **permutation importance** on the test set.
- **FFNN:** **SHAP values** (TreeExplainer for trees, KernelExplainer or DeepExplainer for the NN).
- **Category-level importance:** sum permutation importances within each of the 7 feature categories — directly answers RQ2.
- **Country-level error analysis:** per-country accuracy and recall — answers RQ3.

### 3.13 Overfitting / underfitting checks

- Plot **learning curves** (train vs. validation score as N grows).
- Plot **validation curves** (score vs. key hyperparameter).
- For the FFNN: training-loss vs. validation-loss curve over epochs; trigger early stopping on val_loss.

---

## 4. Results *(structure for the eventual report)*

For each model:

- **Key Findings:** test-set metrics, comparison to baseline, statistical significance of the difference where appropriate.
- **Actionable Insights:** which feature categories drive prediction (RQ2); which countries are hardest (RQ3); ranking of models on PR-AUC vs. complexity.
- **Practical Outcomes:** what this means for turnout campaigns — e.g. *if institutional trust is the largest driver, NGO outreach should prioritise trust-building over GOTV mobilisation in low-trust regions.*

Planned tables and figures:

- **Table 1:** test-set metric comparison across all 4 models.
- **Table 2:** runtime / model-complexity comparison.
- **Figure 1:** ROC and PR curves overlaid.
- **Figure 2:** category-level permutation importance.
- **Figure 3:** SHAP summary plot for the best model.
- **Figure 4:** confusion matrices side by side.
- **Figure 5:** per-country recall / precision heatmap.

---

## 5. Ethical Considerations

- **Survey privacy:** ESS data are anonymised at source — no individual identifiers — but turnout prediction at the individual level still raises questions about *micro-targeting* of voters. The report will discuss the dual-use risk.
- **Algorithmic fairness:** check whether model errors are systematically larger for certain age groups, genders, or countries; this would constitute disparate impact. Report per-group recall.
- **Sample bias:** ESS over-/under-sampling and country quotas mean the pooled sample is not a uniform representation of "Europe". Use design weights for descriptive stats; flag this as a limitation.
- **Causal inference disclaimer:** prediction ≠ causation. The report will be careful not to claim that, e.g., low trust *causes* abstention — only that the two co-occur.
- **AI alignment / dual use:** brief note (Lecture 14) on how an accurate turnout model could be misused by political operatives.

---

## 6. Discussion

- **Answers to RQ1, RQ2, RQ3** — directly, with numbers.
- **Implications & learning reflections** — what surprised us, what the lecture concepts looked like in practice (especially bias-variance, regularisation, CV, SMOTE).
- **Limitations:**
  - ESS is cross-sectional → cannot model individual change over time.
  - Self-reported turnout is overstated by ~10–15 pp in surveys vs. official records (well-documented response bias).
  - 30 countries pooled hides national-context heterogeneity.
  - Feature set excludes regional variables (urban/rural) and political-system variables (compulsory voting, electoral system).

---

## 7. Conclusion & Future Work

- One-paragraph recap of the strongest finding.
- **Future work:**
  - Multi-level (hierarchical) model with country random effects.
  - Compare across ESS rounds to detect temporal drift.
  - Add macro-level features (turnout in last election, electoral system, GDP per capita) → cross-level interactions.
  - Try a TabNet or FT-Transformer (post-course, since not allowed as primary here) and benchmark against the FFNN.

---

## 8. References (APA 7)

To be populated. Anchor citations:

- Verba, S., Schlozman, K. L., & Brady, H. E. (1995). *Voice and Equality: Civic Voluntarism in American Politics.* Harvard University Press.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. *IJCNN 2008*, 1322–1328.
- European Social Survey ERIC (ESS ERIC). (2024). ESS Round 11: European Social Survey Round 11 Data (Edition 4.1). Sikt – Norwegian Agency for Shared Services in Education and Research.

---

## 9. Self-improvement & process notes

This section is for me — not the final report.

### What's already done
- Selected 25 ESS variables across 7 conceptual groups.
- Built a country-participation choropleth (in `marysia_data.ipynb`).
- Cleaned the target (`vote`): dropped ineligible + non-response codes, recoded to 0/1.
- Produced two presentation-ready table figures (`table_vote_raw.png`, `table_vote_binary.png`).
- Wrote `gameplan.md` and `gameplan.ipynb` (this scaffold).
- Wrote `01_eda.ipynb` — variable-by-variable EDA with one section per feature, sentinel-code analysis, distribution plot, turnout-rate plot, cleaning decision per variable, and a final decision summary table.

### What I should do next, in order
1. **Run `01_eda.ipynb` end-to-end**, save the key figures (per-country turnout, age gradient, trust correlation heatmap, full correlation matrix) into the workspace folder.
2. **Build `02_preprocessing.ipynb`** — implement the cleaner + the two feature sets (A: raw 24, B: composite ~11), save both as CSV.
3. **Build `03_modeling.ipynb`** — train all four models (LR baseline, RF, GBM, FFNN) on **both** feature sets. Each model must beat the previous PR-AUC by a margin worth the added complexity.
4. **Build `04_results_and_analysis.ipynb`** — test-set metrics table, ROC + PR curves, category-level permutation importance (sums permutation importance within each of the 7 feature groups → directly answers RQ2), SHAP for the best model, per-country recall heatmap (RQ3).
5. **Write up Results, Discussion, Ethics** in the Word doc — references already wired in.

### Pitfalls I want to avoid
- Treating ESS sentinel codes (66, 77, 88, 99, etc.) as real numbers — would silently destroy results.
- Forgetting to apply SMOTE only on the training fold.
- Reporting accuracy as the headline metric (the 78/22 split makes that misleading).
- Letting the FFNN be deeper than necessary — the tabular dataset is small enough that 2–3 layers will plateau.
- Using survey weights as features (they're design weights, not predictors).
- Cross-country pooling without including `cntry` — would let the model learn from country-level confounding without acknowledging it.

### Things to watch at the oral exam
- Be ready to explain bias-variance, why PR-AUC > ROC-AUC under imbalance, and what early stopping does.
- Be ready to justify the FFNN architecture choice and dropout rate.
- Be ready to discuss the SMOTE vs. class-weights trade-off.
- Have one slide for each of: research question, methodology, best model + headline number, top 3 features, ethical caveat.

---

**End of gameplan.**
