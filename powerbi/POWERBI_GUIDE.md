# Power BI Companion Dashboard — Setup Guide

Power BI Desktop is a GUI tool, so it can't be scripted headlessly — build it
following this guide (~2-3 hours). Connecting live to MySQL (DirectQuery)
rather than importing a CSV is what makes this look like a real BI deployment
instead of a classroom exercise.

## 1. Connect to your data source

1. Deploy your MySQL database first (see main README → Deployment).
2. In Power BI Desktop: **Get Data → More → Database → MySQL database**
3. Enter server + database name from your MySQL host (e.g. Railway/Aiven).
4. Choose **DirectQuery** (not Import) so the dashboard reflects live data
   after each automated refresh — this is the detail that separates a real
   BI deployment from a static-CSV school project.
5. Load these tables: `employees`, `departments`, `job_roles`, `managers`,
   `salary_history`, `performance_reviews`, `engagement_surveys`,
   `attendance`, `attrition_events`, `attrition_predictions`, `sentiment_scores`.

## 2. Build the data model (relationships)

Power BI should auto-detect most FK relationships. Verify:
- `employees[department_id]` → `departments[department_id]`
- `employees[job_role_id]` → `job_roles[job_role_id]`
- `attrition_predictions[employee_id]` → `employees[employee_id]` (1:1)
- `attrition_events[employee_id]` → `employees[employee_id]` (1:1)
- `salary_history` / `performance_reviews` / `engagement_surveys` /
  `attendance` all → `employees[employee_id]` (many:1)

## 3. Key DAX measures

```dax
Total Headcount =
CALCULATE(COUNTROWS(employees), employees[is_active] = TRUE())

Attrition Rate =
DIVIDE(COUNTROWS(attrition_events), COUNTROWS(employees), 0)

Attrition Rate (Rolling 3-Month) =
CALCULATE(
    [Attrition Rate],
    DATESINPERIOD(attrition_events[exit_date], MAX(attrition_events[exit_date]), -3, MONTH)
)

Avg Predicted Risk =
AVERAGE(attrition_predictions[risk_score])

Critical Risk Count =
CALCULATE(
    COUNTROWS(attrition_predictions),
    attrition_predictions[risk_band] = "Critical",
    employees[is_active] = TRUE()
)

Cost of Turnover =
-- Estimated using a common HR industry heuristic: ~50% of annual salary per exit
SUMX(
    attrition_events,
    RELATED(employees[employee_id]) * 0 +
    CALCULATE(MAX(salary_history[monthly_salary])) * 12 * 0.5
)

YoY Attrition % Change =
VAR CurrentYear = [Attrition Rate]
VAR PriorYear =
    CALCULATE([Attrition Rate], SAMEPERIODLASTYEAR(attrition_events[exit_date]))
RETURN DIVIDE(CurrentYear - PriorYear, PriorYear, BLANK())

Avg Tenure (Months) =
AVERAGEX(
    employees,
    DATEDIFF(employees[hire_date], TODAY(), MONTH)
)
```

## 4. Recommended pages

| Page | Visuals |
|---|---|
| **Executive Overview** | KPI cards (Headcount, Attrition Rate, Critical Risk Count, Cost of Turnover), attrition trend line, department bar chart |
| **Driver Analysis** | Table of `top_driver_1/2/3` frequency, decomposition tree on `risk_score` by department/role/cluster |
| **Flight-Risk Watchlist** | Table visual filtered to `risk_band = Critical/High`, sorted by `risk_score`, with drill-through to an individual employee page |
| **Survival Curves** | Line chart of `predicted_tenure_months` distribution by department (use a histogram visual) |
| **Sentiment & Exit Themes** | Bar chart of `sentiment_label` counts, table of `exit_reason` frequency, word cloud visual (import from AppSource) fed by `comment_text` |

## 5. Interactivity to add

- **Bookmarks**: create one bookmark per risk band that filters the watchlist page — wire to buttons for a guided-navigation feel.
- **Drill-through**: right-click an employee row → drill through to a dedicated "Employee 360" page showing their salary history, review history, and SHAP drivers.
- **Slicers**: department, job role, risk band, hire-year.
- **Row-level security (optional, bonus)**: create a role that filters `employees` to `manager_id = USERPRINCIPALNAME()` so each manager only sees their own team — a nice "production-grade" touch to mention in interviews.

## 6. Publish

**File → Publish → Power BI Service** (free account works). Set a scheduled
refresh matching your Python retraining cadence (e.g. daily) under
**Dataset settings → Scheduled refresh**, using the same MySQL credentials.
