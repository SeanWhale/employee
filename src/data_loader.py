import csv
import io
import pandas as pd
import zipfile

from src.config import DATA_DIR, DATA_ZIP_PATH, ZIP_ROOT_DIR, SCHEMA

def parse_sql_dump(file_name, table_name):
    """
    解析 MySQL dump 文件并将 VALUES 后面的数据提取为 DataFrame。
    优先读取 data/ 目录，其次从仓库提供的 test_db 压缩包读取。
    """
    columns = SCHEMA.get(table_name)
    if not columns:
        return pd.DataFrame()

    file_path = DATA_DIR / file_name
    file_stream = None
    zip_handle = None

    if file_path.exists():
        file_stream = open(file_path, 'r', encoding='utf-8')
    elif DATA_ZIP_PATH.exists():
        zip_member = f"{ZIP_ROOT_DIR}/{file_name}"
        zip_handle = zipfile.ZipFile(DATA_ZIP_PATH, 'r')
        if zip_member not in zip_handle.namelist():
            print(f"File {file_name} not found in {DATA_ZIP_PATH}.")
            zip_handle.close()
            return pd.DataFrame(columns=columns)
        file_stream = io.TextIOWrapper(zip_handle.open(zip_member, 'r'), encoding='utf-8')
    else:
        print(f"File {file_path} not found and zip {DATA_ZIP_PATH} does not exist.")
        return pd.DataFrame(columns=columns)

    rows = []
    values_started = False

    def parse_value_segment(segment):
        normalized = segment.strip().rstrip(',;')
        if normalized.startswith('('):
            normalized = normalized[1:]
        if normalized.endswith(')'):
            normalized = normalized[:-1]
        if not normalized:
            return
        try:
            parsed = next(
                csv.reader(
                    [normalized],
                    delimiter=',',
                    quotechar="'",
                    escapechar='\\',
                    skipinitialspace=True,
                )
            )
        except StopIteration as exc:
            preview = f"{segment[:120]}{'...' if len(segment) > 120 else ''}"
            raise ValueError(f"Failed to parse SQL values segment in {file_name}: {preview}") from exc
        rows.append(parsed)

    try:
        for line in file_stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith('INSERT INTO'):
                if 'VALUES' in line:
                    values_started = True
                    value_part = line.split('VALUES', 1)[1].strip()
                    if value_part:
                        for part in value_part.split('),('):
                            parse_value_segment(part)
                continue
            if not values_started:
                continue
            for part in line.split('),('):
                parse_value_segment(part)
    finally:
        if file_stream is not None:
            file_stream.close()
        if zip_handle is not None:
            zip_handle.close()

    return pd.DataFrame(rows, columns=columns)

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
