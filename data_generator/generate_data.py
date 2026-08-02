"""
TalentPulse — Synthetic Longitudinal HR Data Generator
=========================================================
Generates 24 months of realistic HR history (salary changes, performance
reviews, engagement surveys, attendance, attrition events) for a population
of employees. Writes to:
  - data/talentpulse.db   (SQLite, for local dev / Streamlit demo)
  - data/csv/*.csv        (for importing into real MySQL via LOAD DATA)

Run:  python generate_data.py --n_employees 1500 --months 24
"""
import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)
Faker.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
DATA_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

DEPARTMENTS = ["Sales", "R&D", "Human Resources", "Engineering", "Marketing", "Finance"]
JOB_ROLES = {
    "Sales": ["Sales Executive", "Sales Manager", "Account Manager"],
    "R&D": ["Research Scientist", "Lab Technician", "R&D Manager"],
    "Human Resources": ["HR Executive", "HR Manager", "Recruiter"],
    "Engineering": ["Software Engineer", "Senior Engineer", "Engineering Manager"],
    "Marketing": ["Marketing Analyst", "Marketing Manager", "Content Strategist"],
    "Finance": ["Financial Analyst", "Accountant", "Finance Manager"],
}
EDU_FIELDS = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
EXIT_REASONS = [
    "Better opportunity elsewhere", "Compensation dissatisfaction", "Work-life balance",
    "Relocation", "Career growth stagnation", "Manager conflict", "Retirement", "Health reasons",
]

POSITIVE_COMMENTS = [
    "I feel valued and supported by my team.",
    "Great growth opportunities and a manager who listens.",
    "Happy with the recent recognition for my work.",
    "Good balance between work and personal life lately.",
]
NEGATIVE_COMMENTS = [
    "Overworked and underpaid, feeling exhausted.",
    "No clear path for promotion despite good performance.",
    "Management does not listen to feedback.",
    "Considering other options due to stagnant salary.",
    "Constant overtime is affecting my health.",
]
NEUTRAL_COMMENTS = [
    "Things are okay, nothing major to report.",
    "Workload is manageable this month.",
    "No strong feelings either way currently.",
]


def months_between(start: date, n: int):
    out = []
    y, m = start.year, start.month
    for i in range(n):
        out.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def generate(n_employees=1500, n_months=24, attrition_rate=0.16):
    today = date.today()
    start_month = date(today.year, today.month, 1) - timedelta(days=30 * n_months)
    timeline = months_between(start_month, n_months)

    departments = pd.DataFrame({"department_id": range(1, len(DEPARTMENTS) + 1), "department_name": DEPARTMENTS})

    job_roles_rows, jr_id = [], 1
    for dept_id, dept in zip(departments.department_id, departments.department_name):
        for role in JOB_ROLES[dept]:
            job_roles_rows.append({"job_role_id": jr_id, "job_role_name": role, "department_id": dept_id})
            jr_id += 1
    job_roles = pd.DataFrame(job_roles_rows)

    managers_rows, mgr_id = [], 1
    for dept_id in departments.department_id:
        for _ in range(random.randint(2, 4)):
            managers_rows.append({"manager_id": mgr_id, "manager_name": fake.name(), "department_id": dept_id})
            mgr_id += 1
    managers = pd.DataFrame(managers_rows)

    employees_rows = []
    salary_rows, perf_rows, survey_rows, attendance_rows, attrition_rows = [], [], [], [], []
    sal_id = perf_id = surv_id = att_id = 1

    n_attrit = int(n_employees * attrition_rate)
    attrit_flags = [True] * n_attrit + [False] * (n_employees - n_attrit)
    random.shuffle(attrit_flags)

    for emp_id in range(1, n_employees + 1):
        dept_row = departments.sample(1).iloc[0]
        dept_id = dept_row.department_id
        role_row = job_roles[job_roles.department_id == dept_id].sample(1).iloc[0]
        mgr_row = managers[managers.department_id == dept_id].sample(1).iloc[0]

        age = int(np.clip(np.random.normal(35, 9), 21, 60))
        hire_date = fake.date_between(start_date="-12y", end_date=start_month)
        will_leave = attrit_flags[emp_id - 1]

        # underlying "true" risk drivers used to bias simulated behaviour
        base_satisfaction = np.random.uniform(1, 5)
        base_worklife = np.random.uniform(1, 5)
        overtime_tendency = np.random.uniform(0, 1)
        if will_leave:
            base_satisfaction *= np.random.uniform(0.4, 0.75)
            base_worklife *= np.random.uniform(0.5, 0.8)
            overtime_tendency = min(1, overtime_tendency + np.random.uniform(0.1, 0.4))

        employees_rows.append({
            "employee_id": emp_id,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "gender": random.choice(["Male", "Female"]),
            "age": age,
            "marital_status": random.choice(["Single", "Married", "Divorced"]),
            "education_level": random.randint(1, 5),
            "education_field": random.choice(EDU_FIELDS),
            "department_id": dept_id,
            "job_role_id": role_row.job_role_id,
            "manager_id": mgr_row.manager_id,
            "hire_date": hire_date.isoformat(),
            "is_active": not will_leave,
        })

        base_salary = np.random.normal(45000, 15000)
        base_salary = max(18000, base_salary)
        exit_month_idx = random.randint(int(n_months * 0.3), n_months - 1) if will_leave else None

        cur_salary = base_salary
        for i, m in enumerate(timeline):
            if exit_month_idx is not None and i > exit_month_idx:
                break
            hike = 0
            if i > 0 and i % 12 == 0:
                hike = np.random.uniform(2, 6) if not will_leave else np.random.uniform(0, 2)
                cur_salary *= (1 + hike / 100)
            salary_rows.append({
                "salary_id": sal_id, "employee_id": emp_id, "effective_date": m.isoformat(),
                "monthly_salary": round(cur_salary, 2), "hike_percent": round(hike, 2),
            })
            sal_id += 1

            if i % 6 == 0:
                rating = int(np.clip(np.random.normal(3.3 if not will_leave else 2.6, 0.9), 1, 5))
                perf_rows.append({
                    "review_id": perf_id, "employee_id": emp_id, "review_date": m.isoformat(),
                    "rating": rating, "manager_id": mgr_row.manager_id,
                    "promotion_flag": rating >= 4 and random.random() < 0.15,
                })
                perf_id += 1

            if i % 3 == 0:
                sat = int(np.clip(np.random.normal(base_satisfaction, 0.6), 1, 5))
                wlb = int(np.clip(np.random.normal(base_worklife, 0.6), 1, 5))
                if sat <= 2:
                    comment = random.choice(NEGATIVE_COMMENTS)
                elif sat >= 4:
                    comment = random.choice(POSITIVE_COMMENTS)
                else:
                    comment = random.choice(NEUTRAL_COMMENTS)
                survey_rows.append({
                    "survey_id": surv_id, "employee_id": emp_id, "survey_date": m.isoformat(),
                    "satisfaction_score": sat, "worklife_balance": wlb, "comment_text": comment,
                })
                surv_id += 1

            absent = int(np.clip(np.random.poisson(1.2), 0, 15))
            overtime = round(max(0, np.random.normal(overtime_tendency * 20, 5)), 1)
            attendance_rows.append({
                "attendance_id": att_id, "employee_id": emp_id, "month": m.isoformat(),
                "absent_days": absent, "overtime_hours": overtime,
            })
            att_id += 1

        if will_leave and exit_month_idx is not None:
            exit_date = timeline[exit_month_idx]
            reason = random.choice(EXIT_REASONS)
            interview_text = random.choice(NEGATIVE_COMMENTS) if random.random() < 0.7 else random.choice(NEUTRAL_COMMENTS)
            attrition_rows.append({
                "event_id": emp_id, "employee_id": emp_id, "exit_date": exit_date.isoformat(),
                "exit_reason": reason, "exit_interview_text": interview_text,
            })

    employees = pd.DataFrame(employees_rows)
    salary_history = pd.DataFrame(salary_rows)
    performance_reviews = pd.DataFrame(perf_rows)
    engagement_surveys = pd.DataFrame(survey_rows)
    attendance = pd.DataFrame(attendance_rows)
    attrition_events = pd.DataFrame(attrition_rows)

    return {
        "departments": departments, "job_roles": job_roles, "managers": managers,
        "employees": employees, "salary_history": salary_history,
        "performance_reviews": performance_reviews, "engagement_surveys": engagement_surveys,
        "attendance": attendance, "attrition_events": attrition_events,
    }


def save_all(tables: dict):
    # CSVs for MySQL LOAD DATA / import
    for name, df in tables.items():
        df.to_csv(CSV_DIR / f"{name}.csv", index=False)

    # SQLite for local dev + Streamlit demo
    db_path = DATA_DIR / "talentpulse.db"
    conn = sqlite3.connect(db_path)
    for name, df in tables.items():
        df.to_sql(name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Saved {len(tables)} tables -> {db_path} and {CSV_DIR}/*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_employees", type=int, default=1500)
    parser.add_argument("--months", type=int, default=24)
    args = parser.parse_args()

    tables = generate(n_employees=args.n_employees, n_months=args.months)
    save_all(tables)
    print("Row counts:")
    for name, df in tables.items():
        print(f"  {name}: {len(df)}")
