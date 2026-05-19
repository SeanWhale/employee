import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = BASE_DIR / 'Employees（大型企业 HR，百万级数据）'
DATA_ZIP_PATH = RAW_DATA_DIR / 'test_db-master.zip'
ZIP_ROOT_DIR = 'test_db-master'
ASSETS_DIR = BASE_DIR / 'assets'
REPORTS_DIR = BASE_DIR / 'reports'
LOGS_DIR = BASE_DIR / 'logs'

RANDOM_SEED = 42
CURRENT_SENTINEL_DATE = '9999-01-01'

# Column names based on typical Employees database schema
SCHEMA = {
    'departments': ['dept_no', 'dept_name'],
    'employees': ['emp_no', 'birth_date', 'first_name', 'last_name', 'gender', 'hire_date'],
    'dept_emp': ['emp_no', 'dept_no', 'from_date', 'to_date'],
    'dept_manager': ['emp_no', 'dept_no', 'from_date', 'to_date'],
    'titles': ['emp_no', 'title', 'from_date', 'to_date'],
    'salaries': ['emp_no', 'salary', 'from_date', 'to_date']
}
