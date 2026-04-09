# Exploratory Data Analysis — Carseats Retail Dataset

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Analysis Pipeline](#analysis-pipeline)
- [Statistical Tests](#statistical-tests)
- [Regression Model](#regression-model)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)

---

## Project Overview

This project performs a systematic exploratory data analysis on the **Carseats dataset** — a retail dataset capturing sales figures and operational characteristics for 400 car seat stores across different markets. The goal is to identify the key drivers of product sales using a combination of visual analysis, hypothesis testing, grouped aggregation, and linear regression modelling.

The analysis moves from raw data inspection through to actionable business recommendations, demonstrating how statistical techniques can bridge the gap between raw observations and real-world retail strategy.

---

## Dataset

| Property | Detail |
|---|---|
| Observations | 400 store records (1 missing per key variable → 399–400 usable) |
| Variables | 11 (mix of numerical and categorical) |
| Source | Standard `Carseats` dataset (ISLR) |
| Target variable | `Sales` (unit sales in thousands) |

### Variable Reference

| Variable | Type | Description |
|---|---|---|
| `Sales` | Numerical | Unit sales at each location (thousands) |
| `CompPrice` | Numerical | Competitor price at each location |
| `Income` | Numerical | Community income level (thousands USD) |
| `Advertising` | Numerical | Local advertising budget (thousands USD) |
| `Population` | Numerical | Population size of region |
| `Price` | Numerical | Price charged for the product |
| `ShelveLoc` | Categorical | Shelf location quality: Bad / Medium / Good |
| `Age` | Numerical | Average customer age |
| `Education` | Numerical | Average education level |
| `Urban` | Binary | Whether store is in an urban area (Yes/No) |
| `US` | Binary | Whether store is in the US (Yes/No) |

---

## Analysis Pipeline

### 1. Descriptive Statistics

`rawdata.describe()` was used to inspect distributional properties across all numeric variables.

**Notable observations:**
- `Sales` mean ≈ 7.5 units, std ≈ 2.82 — moderate spread, indicating genuine performance variation across stores
- `Advertising` is heavily right-skewed (mean 6.6, but many stores spend 0) — most stores invest minimally in promotions
- `Price` is relatively stable (IQR: 100–131), suggesting a fairly competitive pricing band

### 2. Missing Value Treatment

```python
rawdata.isnull().sum()
# Sales: 1, CompPrice: 1, Income: 1, Urban: 1
```

Missing values (one per affected variable) were handled using **forward fill**:

```python
Data = rawdata.ffill()
```

Forward fill was chosen over mean imputation because the dataset rows represent sequential store records where the adjacent value is the most contextually appropriate substitute. Post-treatment: zero missing values across all columns.

### 3. Visualisations

#### Distribution of Sales (Histogram + Bell Curve)

Sales follow an approximately normal distribution centred around 6–8 units. A normal curve overlay confirms the near-symmetrical shape. Extreme sales values (very low or very high) are rare — typical store performance dominates.

#### Price vs Sales (Scatter Plot)

A negative trend line confirms the classic inverse demand relationship: higher-priced products are associated with lower sales volumes. Points are colour-coded above/below the median price for visual clarity.

#### Sales by Shelf Location (Boxplot)

| ShelveLoc | Median Sales |
|---|---|
| Good | 10.09 |
| Medium | 7.42 |
| Bad | 5.52 |

Good shelf placement is associated with both higher median sales and greater variability, suggesting that premium visibility amplifies the full range of store performance outcomes.

#### Correlation Heatmap

A full numeric correlation matrix was plotted using `plt.imshow()` with annotated cell values.

**Key correlations with `Sales`:**
- `Advertising`: strongest positive relationship — promotional spend is the top driver
- `Price`: moderate negative relationship — price sensitivity is present but not dominant
- `CompPrice`, `Income`, `Age`, `Education`, `Population`: minimal to negligible correlations — demographics and competitor pricing have limited explanatory power in this dataset

---

## Statistical Tests

### 4. Categorical Frequency Analysis

```python
Data['ShelveLoc'].value_counts()
# Medium: 219 | Bad: 96 | Good: 85

Data['Urban'].value_counts()
# Yes: 281 | No: 119

Data['US'].value_counts()
# Yes: 258 | No: 142
```

Most stores occupy medium-visibility shelf positions. The dataset skews towards urban, US-based stores — findings should be interpreted in that context.

### 5. Chi-Square Test of Independence

**Variables:** `ShelveLoc` × `Urban`

```
H₀: Shelf location quality and urban/rural setting are independent
H₁: A significant association exists between the two variables
```

```
Chi-square statistic: 2.132
p-value: 0.344
Degrees of freedom: 2
```

**Conclusion:** p = 0.344 >> 0.05 → **Fail to reject H₀**. There is no statistically significant association between shelf location quality and whether a store is urban or rural. The distribution of shelf placements is independent of store geography.

### 6. Subgroup Analysis

**Subset 1 — High Income + High Advertising** (`Income > 80k` AND `Advertising > 10k`, n = 47):

| Metric | Value |
|---|---|
| Mean Sales | 9.11 units |
| Mean Income | $103,400 |
| Mean Competitor Price | $119.23 |
| Sales Std Dev | 2.51 |

High-income, high-advertising stores show above-average sales with relatively low variance — suggesting this segment performs more consistently than the general population.

**Subset 2 — US Stores with Good Shelf Placement** (`US == 'Yes'` AND `ShelveLoc == 'Good'`, n = 61):

| Metric | Value |
|---|---|
| Mean Sales | 10.63 units |
| Mean Competitor Price | $125.33 |
| Sales Std Dev | 2.20 |

This is the strongest-performing segment in the dataset — US stores with prime shelf visibility achieve the highest average sales and tightest distribution.

### 7. Independent Samples T-Tests

**Test 1 — Urban vs Non-Urban Sales:**

```
T-statistic: -0.121
p-value: 0.904
```

**Conclusion:** No statistically significant difference in mean sales between urban and non-urban stores. Store geography (urban/rural) is not a meaningful driver of sales performance.

**Test 2 — US vs Non-US Sales:**

```
T-statistic: 3.649
p-value: 0.0003
```

**Conclusion:** Highly significant difference (p << 0.05). US-based stores have statistically higher average sales than non-US stores. This is very unlikely to be due to chance.

### 8. Grouped Aggregation

```python
Data.groupby('ShelveLoc').agg({'Sales': 'mean', 'Price': 'mean', 'Advertising': 'sum'})
```

| ShelveLoc | Mean Sales | Mean Price | Total Advertising |
|---|---|---|---|
| Bad | 5.49 | 114.27 | 597 |
| Medium | 7.31 | 115.65 | 1,432 |
| Good | 10.21 | 117.88 | 625 |

Notably, Medium-shelf stores collectively spend the most on advertising yet still underperform Good-shelf stores on average — confirming that **shelf placement has a stronger effect on sales than promotional spend alone**.

---

## Regression Model

### 9. OLS Multiple Linear Regression

**Predictors:** `Price`, `Advertising`, `Income`  
**Target:** `Sales`

```python
import statsmodels.api as sm
X = sm.add_constant(Data[['Price', 'Advertising', 'Income']])
model = sm.OLS(Data['Sales'], X).fit()
```

**Results:**

| Variable | Coefficient | p-value | Interpretation |
|---|---|---|---|
| Intercept | 12.166 | < 0.001 | Baseline sales |
| `Price` | -0.054 | < 0.001 | Strong negative effect — each £1 price increase reduces sales by ~0.054 units |
| `Advertising` | 0.121 | < 0.001 | Each $1k advertising increase drives ~0.12 additional unit sales |
| `Income` | 0.011 | 0.012 | Weak but significant positive effect |

**Model fit:** R² = 0.292 — the three predictors explain ~29% of sales variance. The remaining 71% is attributable to unmeasured factors (seasonality, store management, competitor behaviour, etc.).

### Residual Diagnostics

All three standard regression assumptions were checked:

**Normality of residuals (Histogram):** Residuals are approximately bell-shaped and centred at zero. Most fall within ±2 units — errors are small and balanced.

**Normality of residuals (Q-Q Plot):** Points lie close to the reference line throughout the central range. Slight tail deviations indicate mild outlier presence but are not severe enough to invalidate the model.

**Actual vs Predicted Plot:** Predictions track the 45° reference line reasonably well. Spread increases at higher sales values — a common pattern indicating mild heteroscedasticity at the upper end.

**Overall diagnostic verdict:** Model assumptions are reasonably met. Results are reliable for identifying directional effects and variable significance, though predictive accuracy is limited by the 29% R².

---

## Key Findings

1. **Price** has a strong, statistically significant negative effect on sales — the primary demand lever
2. **Advertising** is the strongest positive driver — promotional investment consistently translates to higher sales
3. **Shelf placement** is the most powerful categorical driver — Good vs Bad shelf location produces a ~4.7 unit difference in mean sales
4. **Urban vs rural location** has no significant effect on sales (p = 0.90)
5. **US stores** significantly outperform non-US stores (p = 0.0003)
6. **Demographics** (income, age, education, population) show negligible correlation with sales — consumer decisions here are not strongly demographically determined
7. Medium-shelf stores spend the most on advertising but still underperform Good-shelf stores, suggesting shelf placement ROI exceeds advertising ROI in this context

---

## Recommendations

1. **Prioritise shelf placement** — Good shelf positions have the highest impact on sales; this should be a core negotiation objective with retail partners
2. **Invest in targeted advertising** — advertising is the only consistently significant controllable driver; underspending is a missed opportunity
3. **Maintain competitive pricing** — price has a strong negative demand effect; overpricing relative to competitors risks meaningful sales loss
4. **Deprioritise demographic targeting** — income, age, and education show no actionable relationship with sales in this dataset; resources are better allocated to in-store factors
5. **Focus US market strategy** — US stores significantly outperform; understanding what operational factors differentiate US performance could inform international expansion

---

## Repository Structure

```
carseats-eda/
│
├── data/
│   └── Carseats.xlsx                   # Raw dataset
│
├── notebooks/
│   └── carseats_eda.ipynb              # Full analysis notebook
│
├── reports/
│   └── Programming_for_Data_Analytics.pdf   # Full written report
│
└── README.md
```

---

## Dependencies

```txt
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scipy>=1.9
statsmodels>=0.13
openpyxl>=3.0       # for pd.read_excel()
```
