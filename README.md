# HR Analytics End-to-End Implementation (Employees Dataset)

This project provides a reproducible Python implementation for the six core Employees tables:

- `employees`
- `departments`
- `dept_emp`
- `dept_manager`
- `salaries`
- `titles`

It covers:

1. Data quality checks + cleanup/imputation (nulls/placeholders/invalid dates/current sentinel date)
2. Dependency analysis (correlation, partial correlation, hypothesis tests)
3. Similarity analysis (clustering model comparison + cluster interpretation)
4. Forecasting + validation (historical salary trend forecast vs actual with error visualization)
5. Dashboard with multi-chart big-screen layout, dynamics, and interactivity
6. External benchmark linkage (synthetic industry baseline comparison)
7. Optional map visual (synthetic geo data if external geo source is unavailable)

---

## 1) Setup

```bash
cd /home/runner/work/employee/employee
python -m pip install -r requirements.txt
```

## 2) Run dashboard

```bash
python /home/runner/work/employee/employee/run_dashboard.py
```

Open: `http://127.0.0.1:8050/`

> Data loading is automatic.  
> Loader first checks `/home/runner/work/employee/employee/data`, then falls back to:
> `/home/runner/work/employee/employee/Employees（大型企业 HR，百万级数据）/test_db-master.zip`

---

## 3) Data quality logic

Implemented in `src/data_cleaning.py`:

- Detect placeholders: `NULL/null/None/N/A/NA/UNKNOWN/''/' '`
- Convert `9999-01-01` to `NaT` for temporal “current” values
- Type conversion for IDs, salary, and dates
- Invalid range handling:
  - if `from_date > to_date`, swap the pair
  - if `birth_date > hire_date`, backfill birth date with median hire-age rule
- Imputation:
  - numeric columns: median
  - categorical columns: mode
  - non-`to_date` datetime columns: median datetime
  - `to_date` missing values are retained as current-status markers

---

## 4) Analytics and modeling

### Dependency analysis (`src/analysis.py`)
- Spearman correlation matrix
- Partial correlation: salary vs tenure, controlling for age
- Hypothesis tests:
  - ANOVA: salary differences across departments
  - Spearman test: tenure vs salary

### Similarity analysis (`src/analysis.py`)
- KMeans and Agglomerative clustering are both evaluated
- Silhouette score comparison selects a preferred model
- Cluster profile table summarizes salary/tenure/age means per cluster

---

## 5) Forecasting and validation

Implemented in `src/prediction.py`:

- Aggregate monthly mean salary from historical records
- Train on earlier months, validate on later months (time-based split)
- Forecast with linear trend model
- Visualize:
  - actual vs predicted salary trend
  - error/variance bars
- Metrics shown in chart title: MAE, RMSE, MAPE

---

## 6) Dashboard features

Implemented in `src/dashboard.py`:

- Distribution charts:
  - salary histogram + marginal box
- Categorical comparisons:
  - department salary boxplot
- Relationship/correlation visuals:
  - correlation heatmap
  - tenure vs salary cluster scatter
  - tenure vs attrition boxplot
- Cluster interpretation:
  - cluster profile grouped bar chart
- Forecast validation chart:
  - actual/predicted + error bars
- External linkage:
  - internal vs external benchmark comparison chart
- Map support:
  - synthetic NYC geo/weather points map
- Dynamics:
  - animated yearly department salary chart
- Interactivity:
  - department filter dropdown updates key charts
  - hover tooltips enabled across Plotly charts

---

## 7) Chart selection rationale

- **Histogram (+ marginal box):** salary distribution shape + outliers
- **Boxplot (department):** compare central tendency/spread across categories
- **Heatmap:** compact view of pairwise correlation strengths
- **Scatter (cluster-colored):** intuitive cluster separation and local patterns
- **Grouped bar (cluster profile):** easy cluster-level feature interpretation
- **Actual-vs-predicted line:** time-aligned trend validation
- **Error bar overlay:** directly highlights variance and bias over time
- **Map scatter:** geographic pattern context for external linked data
- **Animated bar race:** temporal evolution of department salary structure

---

## 8) Reproducibility and scalability notes

- Fixed random seed is used where sampling/synthetic data appears.
- Aggregated views (monthly/yearly/groupby) are used for large-table rendering efficiency.
- Scatter visuals are sampled for browser-side performance on million-scale records.
