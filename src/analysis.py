import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
MIN_DEPT_SIZE_FOR_ANOVA = 5

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
    current_mask = dept_emp_latest['to_date'].isna()
    active_lookup = dept_emp_latest.assign(is_active=current_mask)[['emp_no', 'is_active']]
    df_merged = pd.merge(df_merged, active_lookup, on='emp_no', how='left')
    df_merged['is_active'] = df_merged['is_active'].fillna(False)
    df_merged['attrition_flag'] = (~df_merged['is_active']).astype(int)
    
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

def dependency_analysis(df_master):
    """
    相依性分析：
    1) Spearman 相关矩阵
    2) 偏相关（salary 与 tenure，在 age 控制下）
    3) 假设检验（部门薪资差异 ANOVA；司龄与薪资 Spearman）
    """
    valid = df_master[['salary', 'tenure_years', 'age', 'dept_name']].dropna().copy()
    corr_matrix = analyze_correlation(valid)

    # 偏相关：残差法（X~Z, Y~Z 后对残差做相关）
    z = valid['age'].values
    x = valid['salary'].values
    y = valid['tenure_years'].values
    x_res = x - np.polyval(np.polyfit(z, x, 1), z)
    y_res = y - np.polyval(np.polyfit(z, y, 1), z)
    partial_corr = stats.spearmanr(x_res, y_res, nan_policy='omit')

    dept_groups = [
        g['salary'].values
        for _, g in valid.groupby('dept_name')
        if len(g) > MIN_DEPT_SIZE_FOR_ANOVA
    ]
    anova = stats.f_oneway(*dept_groups) if len(dept_groups) > 1 else None
    tenure_salary = stats.spearmanr(valid['tenure_years'], valid['salary'], nan_policy='omit')

    return {
        'correlation_matrix': corr_matrix,
        'partial_correlation_salary_tenure_given_age': {
            'coefficient': float(partial_corr.correlation),
            'p_value': float(partial_corr.pvalue),
        },
        'anova_salary_by_department': (
            None if anova is None else {'f_stat': float(anova.statistic), 'p_value': float(anova.pvalue)}
        ),
        'spearman_salary_vs_tenure': {
            'coefficient': float(tenure_salary.correlation),
            'p_value': float(tenure_salary.pvalue),
        },
    }

def compare_clustering_models(df_master, n_clusters=4, sample_size=20000):
    """
    比较两类聚类模型：KMeans 与 Agglomerative，输出轮廓系数与簇画像。
    """
    features = df_master[['salary', 'tenure_years', 'age']].dropna().copy()
    sample = features.sample(n=min(sample_size, len(features)), random_state=42) if len(features) > sample_size else features
    scaler = StandardScaler()
    x_scaled_sample = scaler.fit_transform(sample)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels_sample = kmeans.fit_predict(x_scaled_sample)
    kmeans_score = float(silhouette_score(x_scaled_sample, kmeans_labels_sample))

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels_sample = agglomerative.fit_predict(x_scaled_sample)
    agg_score = float(silhouette_score(x_scaled_sample, agg_labels_sample))

    preferred = 'kmeans' if kmeans_score >= agg_score else 'agglomerative'
    full_scaled = scaler.transform(features)
    full_kmeans_labels = kmeans.predict(full_scaled)
    label_series = pd.Series(full_kmeans_labels, index=features.index)

    profile = (
        features.assign(cluster=label_series.astype(str))
        .groupby('cluster')[['salary', 'tenure_years', 'age']]
        .mean()
        .round(2)
        .reset_index()
    )

    return {
        'scores': {'kmeans': kmeans_score, 'agglomerative': agg_score},
        'preferred_model': preferred,
        'label_source': 'kmeans',
        'labels': label_series.astype(str),
        'cluster_profile': profile,
    }

def perform_clustering(df_master, n_clusters=4):
    """
    利用 K-Means 对员工进行聚类画像分析（根据薪资与司龄）。
    """
    logging.info("Performing K-Means Clustering on Salary & Tenure...")
    result = compare_clustering_models(df_master, n_clusters=n_clusters)
    df_master['cluster_label'] = result['labels']
    # 将 cluster label 转字符串方便绘图处理类别
    df_master['cluster_label'] = df_master['cluster_label'].astype(str)
    
    logging.info("Clustering finished.")
    return df_master, result, None

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
