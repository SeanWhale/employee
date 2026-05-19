import os
import re
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, SCHEMA

def parse_sql_dump(file_name, table_name):
    """
    解析 MySQL dump 文件并将 VALUES 后面的数据提取为 DataFrame。
    """
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        print(f"File {file_path} not found.")
        return pd.DataFrame()
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取 VALUES 后面的内容
    values_data = content.split("VALUES")[1].strip()
    if values_data.endswith(';'):
        values_data = values_data[:-1]

    # 使用正则表达式提取每条记录的元组
    pattern = re.compile(r"\((.*?)\)", re.DOTALL)
    matches = pattern.findall(values_data)
    
    parsed_data = []
    for match in matches:
        # 简单分割，去除单引号和空白
        row = [val.strip(" '\"") for val in match.split(',')]
        parsed_data.append(row)

    df = pd.DataFrame(parsed_data, columns=SCHEMA.get(table_name))
    return df

def load_all_data():
    """
    加载所有 dump 文件并转换成 DataFrame
    """
    print("Loading departments...")
    departments = parse_sql_dump('load_departments.dump', 'departments')
    
    print("Loading employees...")
    employees = parse_sql_dump('load_employees.dump', 'employees')
    
    print("Loading dept_emp...")
    dept_emp = parse_sql_dump('load_dept_emp.dump', 'dept_emp')
    
    print("Loading dept_manager...")
    dept_manager = parse_sql_dump('load_dept_manager.dump', 'dept_manager')
    
    print("Loading titles...")
    titles = parse_sql_dump('load_titles.dump', 'titles')
    
    print("Loading salaries...")
    salaries1 = parse_sql_dump('load_salaries1.dump', 'salaries')
    salaries2 = parse_sql_dump('load_salaries2.dump', 'salaries')
    salaries3 = parse_sql_dump('load_salaries3.dump', 'salaries')
    salaries = pd.concat([salaries1, salaries2, salaries3], ignore_index=True)
    
    return {
        'departments': departments,
        'employees': employees,
        'dept_emp': dept_emp,
        'dept_manager': dept_manager,
        'titles': titles,
        'salaries': salaries
    }

if __name__ == "__main__":
    data = load_all_data()
    for name, df in data.items():
        print(f"{name}: {len(df)} records")
