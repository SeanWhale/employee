import pandas as pd
import numpy as np
import logging
from src.config import CURRENT_SENTINEL_DATE

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def replace_null_strings(df):
    """
    将 SQL 文本中常见的空值表示符替换为 Pandas 的 np.nan
    """
    placeholders = ['NULL', 'null', 'None', 'N/A', 'NA', 'UNKNOWN', 'unknown', '', ' ']
    return df.replace(placeholders, np.nan)

def convert_types(data):
    """
    根据字段属性转换数据类型，如：emp_no转数值，日期转 datetime
    """
    for name, df in data.items():
        if 'emp_no' in df.columns:
            df['emp_no'] = pd.to_numeric(df['emp_no'], errors='coerce').astype('Int64')
            
        if 'salary' in df.columns:
            df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
            
        date_cols = ['birth_date', 'hire_date', 'from_date', 'to_date']
        for col in date_cols:
            if col in df.columns:
                # 数据库中 9999-01-01 通常代表 "至今"，转换为 NaT 以方便计算
                df[col] = df[col].replace(CURRENT_SENTINEL_DATE, pd.NaT)
                df[col] = pd.to_datetime(df[col], errors='coerce')
    return data

def fix_invalid_temporal_ranges(data):
    """
    修复无效日期范围：
    1) from_date > to_date 时交换日期。
    2) 员工 hire_date < birth_date 时按中位入职年龄回推 birth_date。
    """
    range_tables = ['dept_emp', 'dept_manager', 'titles', 'salaries']
    for name in range_tables:
        df = data.get(name)
        if df is None or 'from_date' not in df.columns or 'to_date' not in df.columns:
            continue
        invalid_mask = (
            df['from_date'].notna()
            & df['to_date'].notna()
            & (df['from_date'] > df['to_date'])
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            logging.warning(f"{name} 检测到 {invalid_count} 条 from_date > to_date，已自动交换修复。")
            from_vals = df.loc[invalid_mask, 'from_date'].copy()
            df.loc[invalid_mask, 'from_date'] = df.loc[invalid_mask, 'to_date'].values
            df.loc[invalid_mask, 'to_date'] = from_vals.values

    emp_df = data.get('employees')
    if emp_df is not None and {'birth_date', 'hire_date'}.issubset(emp_df.columns):
        age_years = (emp_df['hire_date'] - emp_df['birth_date']).dt.days / 365.25
        median_age_at_hire = age_years[(age_years > 12) & (age_years < 80)].median()
        if pd.isna(median_age_at_hire):
            median_age_at_hire = 28.0
        bad_age_mask = (
            emp_df['birth_date'].notna()
            & emp_df['hire_date'].notna()
            & (emp_df['birth_date'] > emp_df['hire_date'])
        )
        bad_age_count = int(bad_age_mask.sum())
        if bad_age_count:
            logging.warning(f"employees 检测到 {bad_age_count} 条 birth_date > hire_date，按中位入职年龄回推修复。")
            age_delta = pd.to_timedelta(median_age_at_hire * 365.25, unit='D')
            emp_df.loc[bad_age_mask, 'birth_date'] = emp_df.loc[bad_age_mask, 'hire_date'] - age_delta

    return data

def clean_data(data):
    """
    主数据清洗流水线：
    1. 修正空值字符串
    2. 类型转换
    3. 缺失值检测
    4. 缺失值插补
    """
    cleaned_data = {}
    
    # 浅拷贝并初步处理空值文本
    for name, df in data.items():
        cleaned_data[name] = replace_null_strings(df.copy())

    # 数据类型转换
    cleaned_data = convert_types(cleaned_data)
    cleaned_data = fix_invalid_temporal_ranges(cleaned_data)
        
    # 检测与处理缺失值
    for name, df in cleaned_data.items():
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        logging.info(f"--- 检查表格: {name} ---")
        
        # 若存在缺失值，按照数据类型分别处理
        if total_missing > 0:
            logging.warning(f"在 {name} 中发现缺失值:\n{missing_counts[missing_counts > 0]}")
            
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]) and col != 'emp_no':
                        # 薪资等数值型使用中位数填补
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        logging.info(f"  -> 已使用中位数 ({median_val}) 填充 {col} 的数值型缺失值。")
                        
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        if col == 'to_date':
                            # to_date 中的缺失通常是针对离职日期的，保留为空即可(代表目前在职)
                            logging.info(f"  -> 字段 {col} 存在日期空值，这通常表示 '至今/在职'，已保留为 NaT。")
                        else:
                            # 非 to_date 日期字段采用中位日期插补
                            ts = df[col].dropna().astype('int64')
                            if not ts.empty:
                                median_ts = int(ts.median())
                                median_date = pd.to_datetime(median_ts)
                                df[col] = df[col].fillna(median_date)
                                logging.info(f"  -> 已使用中位日期 ({median_date.date()}) 填充 {col} 的缺失值。")

                    else:
                        # 类别型数据使用众数插补
                        if col != 'emp_no':
                            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                            df[col].fillna(mode_val, inplace=True)
                            logging.info(f"  -> 已使用众数 ({mode_val}) 填充 {col} 的类别型缺失值。")
        else:
            logging.info(f"{name} 数据完整无缺失。")
            
    return cleaned_data

if __name__ == "__main__":
    from src.data_loader import load_all_data
    
    print("加载数据...")
    raw_data = load_all_data()
    
    print("\n执行数据清洗...")
    cleaned_data = clean_data(raw_data)
    
    print("\n数据清洗完成。")
