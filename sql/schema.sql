-- =========================================================
-- TalentPulse: HR Attrition Intelligence Platform
-- MySQL Schema (3NF)
-- =========================================================

CREATE DATABASE IF NOT EXISTS talentpulse CHARACTER SET utf8mb4;
USE talentpulse;

-- ---------------------------------------------------------
-- Core reference tables
-- ---------------------------------------------------------
CREATE TABLE departments (
    department_id   INT AUTO_INCREMENT PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE job_roles (
    job_role_id   INT AUTO_INCREMENT PRIMARY KEY,
    job_role_name VARCHAR(100) NOT NULL,
    department_id INT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

CREATE TABLE managers (
    manager_id   INT AUTO_INCREMENT PRIMARY KEY,
    manager_name VARCHAR(100) NOT NULL,
    department_id INT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- ---------------------------------------------------------
-- Employees (static / slowly-changing attributes)
-- ---------------------------------------------------------
CREATE TABLE employees (
    employee_id      INT PRIMARY KEY,
    first_name       VARCHAR(50),
    last_name        VARCHAR(50),
    gender           ENUM('Male','Female','Other'),
    age              INT,
    marital_status   ENUM('Single','Married','Divorced'),
    education_level  TINYINT COMMENT '1=Below College 5=Doctorate',
    education_field  VARCHAR(50),
    department_id    INT NOT NULL,
    job_role_id      INT NOT NULL,
    manager_id       INT,
    hire_date        DATE NOT NULL,
    is_active        BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    FOREIGN KEY (job_role_id)   REFERENCES job_roles(job_role_id),
    FOREIGN KEY (manager_id)    REFERENCES managers(manager_id)
);

-- ---------------------------------------------------------
-- Longitudinal (time-series) fact tables — 24 monthly snapshots/employee
-- ---------------------------------------------------------
CREATE TABLE salary_history (
    salary_id     INT AUTO_INCREMENT PRIMARY KEY,
    employee_id   INT NOT NULL,
    effective_date DATE NOT NULL,
    monthly_salary DECIMAL(10,2) NOT NULL,
    hike_percent  DECIMAL(5,2) DEFAULT 0,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    INDEX idx_salary_emp_date (employee_id, effective_date)
);

CREATE TABLE performance_reviews (
    review_id     INT AUTO_INCREMENT PRIMARY KEY,
    employee_id   INT NOT NULL,
    review_date   DATE NOT NULL,
    rating        TINYINT COMMENT '1-5',
    manager_id    INT,
    promotion_flag BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    FOREIGN KEY (manager_id)  REFERENCES managers(manager_id),
    INDEX idx_perf_emp_date (employee_id, review_date)
);

CREATE TABLE engagement_surveys (
    survey_id       INT AUTO_INCREMENT PRIMARY KEY,
    employee_id     INT NOT NULL,
    survey_date     DATE NOT NULL,
    satisfaction_score TINYINT COMMENT '1-5',
    worklife_balance   TINYINT COMMENT '1-5',
    comment_text    TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    INDEX idx_survey_emp_date (employee_id, survey_date)
);

CREATE TABLE attendance (
    attendance_id  INT AUTO_INCREMENT PRIMARY KEY,
    employee_id    INT NOT NULL,
    month          DATE NOT NULL COMMENT 'first day of month',
    absent_days    INT DEFAULT 0,
    overtime_hours DECIMAL(5,1) DEFAULT 0,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    INDEX idx_attendance_emp_month (employee_id, month)
);

CREATE TABLE attrition_events (
    event_id      INT AUTO_INCREMENT PRIMARY KEY,
    employee_id   INT NOT NULL UNIQUE,
    exit_date     DATE NOT NULL,
    exit_reason   VARCHAR(100),
    exit_interview_text TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

-- ---------------------------------------------------------
-- ML output tables (written back by the Python ML layer)
-- ---------------------------------------------------------
CREATE TABLE attrition_predictions (
    employee_id       INT PRIMARY KEY,
    risk_score        DECIMAL(5,4) COMMENT 'probability 0-1',
    risk_band         ENUM('Low','Medium','High','Critical'),
    top_driver_1      VARCHAR(100),
    top_driver_2      VARCHAR(100),
    top_driver_3      VARCHAR(100),
    predicted_tenure_months INT COMMENT 'from survival model',
    cluster_label     VARCHAR(50),
    model_run_date    DATETIME,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE sentiment_scores (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    employee_id   INT NOT NULL,
    source_table  VARCHAR(30) COMMENT 'engagement_surveys or attrition_events',
    source_id     INT,
    sentiment_label VARCHAR(20),
    sentiment_score DECIMAL(5,4),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

-- ---------------------------------------------------------
-- Stored procedures (shows SQL depth beyond SELECT)
-- ---------------------------------------------------------
DELIMITER //

CREATE PROCEDURE sp_monthly_attrition_rate()
BEGIN
    SELECT
        DATE_FORMAT(ae.exit_date, '%Y-%m') AS month,
        d.department_name,
        COUNT(*) AS exits
    FROM attrition_events ae
    JOIN employees e ON e.employee_id = ae.employee_id
    JOIN departments d ON d.department_id = e.department_id
    GROUP BY month, d.department_name
    ORDER BY month;
END //

CREATE PROCEDURE sp_department_avg_tenure()
BEGIN
    SELECT
        d.department_name,
        AVG(
          TIMESTAMPDIFF(MONTH, e.hire_date, COALESCE(ae.exit_date, CURDATE()))
        ) AS avg_tenure_months
    FROM employees e
    JOIN departments d ON d.department_id = e.department_id
    LEFT JOIN attrition_events ae ON ae.employee_id = e.employee_id
    GROUP BY d.department_name;
END //

CREATE PROCEDURE sp_flight_risk_watchlist(IN min_score DECIMAL(5,4))
BEGIN
    SELECT
        e.employee_id, e.first_name, e.last_name,
        d.department_name, jr.job_role_name,
        p.risk_score, p.risk_band, p.top_driver_1, p.top_driver_2, p.top_driver_3
    FROM attrition_predictions p
    JOIN employees e ON e.employee_id = p.employee_id
    JOIN departments d ON d.department_id = e.department_id
    JOIN job_roles jr ON jr.job_role_id = e.job_role_id
    WHERE p.risk_score >= min_score AND e.is_active = TRUE
    ORDER BY p.risk_score DESC;
END //

DELIMITER ;
