import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_master_dataset(cleaned_data):
    """
    将员工、最新薪资、最新职称与部门合并构建一张宽表，以供相关性和聚类使用。
    """
    # 获取原始数据池
    emp = cleaned_data['employees']
    sal = cleaned_data['salaries']
    tit = cleaned_data['titles']
    dept_emp = cleaned_data['dept_emp']
    dept = cleaned_data['departments']
    
    # 因为存在历史记录，我们只取每位员工当前的薪资、部门和职称（to_date 为最大的或 NaT(至今)）
    # pandas 中 NaT 无法通过 max 选出，这里我们为 NaT 填充一个极大值进行获取最新记录的排序
    
    def get_latest_records(df):
        df_sorted = df.assign(
            to_date_filled=df['to_date'].fillna(pd.Timestamp('2099-12-31'))
        ).sort_values(by=['emp_no', 'to_date_filled'])
        return df_sorted.drop_duplicates(subset=['emp_no'], keep='last').drop(columns=['to_date_filled'])

    logging.info("Building Master Dataset...")
    sal_latest = get_latest_records(sal)
    tit_latest = get_latest_records(tit)
    dept_emp_latest = get_latest_records(dept_emp)

    # 1. 员工 + 薪资
    df_merged = pd.merge(emp, sal_latest[['emp_no', 'salary']], on='emp_no', how='inner')
    # 2. + 职称
    df_merged = pd.merge(df_merged, tit_latest[['emp_no', 'title']], on='emp_no', how='left')
    # 3. + 部门关联
    df_merged = pd.merge(df_merged, dept_emp_latest[['emp_no', 'dept_no']], on='emp_no', how='left')
    # 4. + 部门名称
    df_merged = pd.merge(df_merged, dept, on='dept_no', how='left')
    
    # 提取有价值的数值型特征：如“工龄(年)”、年龄
    current_date = pd.Timestamp('2026-05-18') # 假设当前业务计算日期
    df_merged['tenure_years'] = (current_date - df_merged['hire_date']).dt.days / 365.25
    df_merged['age'] = (current_date - df_merged['birth_date']).dt.days / 365.25
    
    # 清理可能产生的无限值或空值
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['salary', 'tenure_years', 'age'])
    
    logging.info(f"Master Dataset created with shape {df_merged.shape}")
    return df_merged

def analyze_correlation(df_master):
    """
    计算数值型字段的相关系数矩阵（相依性）。
    """
    logging.info("Calculating Correlation Matrix...")
    numeric_df = df_master[['salary', 'tenure_years', 'age']]
    corr_matrix = numeric_df.corr(method='spearman') # 薪资等分布不一定正态，用皮尔斯曼秩相关
    return corr_matrix

def perform_clustering(df_master, n_clusters=4):
    """
    利用 K-Means 对员工进行聚类画像分析（根据薪资与司龄）。
    """
    logging.info("Performing K-Means Clustering on Salary & Tenure...")
    # 提取特征
    X = df_master[['salary', 'tenure_years']].copy()
    
    # 数据标准化，聚类前非常必要
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_master['cluster_label'] = kmeans.fit_predict(X_scaled)
    # 将 cluster label 转字符串方便绘图处理类别
    df_master['cluster_label'] = df_master['cluster_label'].astype(str)
    
    logging.info("Clustering finished.")
    return df_master, kmeans, scaler

if __name__ == "__main__":
    from src.data_loader import load_all_data
    from src.data_cleaning import clean_data
    
    raw = load_all_data()
    cleaned = clean_data(raw)
    master = build_master_dataset(cleaned)
    
    corr = analyze_correlation(master)
    print("\n--- 相关性矩阵 ---")
    print(corr)
    
    master_clustered, model, scaler = perform_clustering(master)
    print("\n--- 聚类结果分布 ---")
    print(master_clustered['cluster_label'].value_counts())
