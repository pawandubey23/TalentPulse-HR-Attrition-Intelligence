"""
Loads the generated CSV files into a real MySQL database (e.g. Railway, Aiven,
PlanetScale, or a local MySQL server). Run schema.sql on the target DB first.

Requires env vars (or a .env file loaded via os.environ):
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

Usage:
    export MYSQL_HOST=...  MYSQL_USER=...  MYSQL_PASSWORD=...  MYSQL_DB=talentpulse
    python load_to_mysql.py
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "data" / "csv"

# Load order matters — respects foreign key dependencies
LOAD_ORDER = [
    "departments", "job_roles", "managers", "employees",
    "salary_history", "performance_reviews", "engagement_surveys",
    "attendance", "attrition_events",
]


def get_engine():
    host = os.environ["MYSQL_HOST"]
    port = os.environ.get("MYSQL_PORT", "3306")
    user = os.environ["MYSQL_USER"]
    pwd = os.environ["MYSQL_PASSWORD"]
    db = os.environ.get("MYSQL_DB", "talentpulse")
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url)


def main():
    engine = get_engine()
    for table in LOAD_ORDER:
        csv_path = CSV_DIR / f"{table}.csv"
        df = pd.read_csv(csv_path)
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=1000)
        print(f"Loaded {len(df)} rows -> {table}")
    print("All tables loaded into MySQL.")


if __name__ == "__main__":
    main()
