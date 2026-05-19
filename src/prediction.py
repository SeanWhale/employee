import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def prepare_features(df_master):
    """
    对宽表数据进行特征工程，主要是将类别型数据转换为可用于训练的数值型。
    在这里我们利用 年龄、工龄、职称和部门 来预测 薪水。
    """
    logging.info("Preparing features for model training...")
    # 选取用于预测的特征
    features = ['tenure_years', 'age', 'title', 'dept_name']
    target = 'salary'
    
    # 过滤掉缺失特征的记录
    df_clean = df_master.dropna(subset=features + [target]).copy()
    
    # 独热编码 (One-Hot Encoding) 处理类别型（职称、部门）
    df_encoded = pd.get_dummies(df_clean[features], drop_first=True)
    
    X = df_encoded
    y = df_clean[target]
    
    return X, y, df_clean

def train_and_predict(X, y, test_size=0.2):
    """
    划分训练集和测试集，训练随机森林回归模型，并返回预测结果和模型指标。
    （代表用 "前面/已知" 数据训练，用 "后面/未知" 数据预测并验证）
    """
    logging.info("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    logging.info("Training RandomForest model... (This might take a moment on large data)")
    # 为了避免百万数据跑太久，我们限制树的深度和数量，兼顾速度与演示效果
    model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    logging.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Model Evaluation -> MSE: {mse:.2f}, R2 Score: {r2:.4f}")
    
    return model, X_test, y_test, y_pred

def plot_prediction_difference(y_test, y_pred, sample_size=300):
    """
    可视化：真实数据与预测数据的差异对比图
    为了大屏展示性能，我们仅随机抽取一部分测试集样本进行绘制。
    """
    # 转为 DataFrame 方便提取索引和对齐
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
    
    # 抽样展示以防图表爆炸
    if len(results_df) > sample_size:
        results_df = results_df.sample(n=sample_size, random_state=42).sort_values(by='Actual').reset_index(drop=True)
        
    fig = go.Figure()
    
    # 添加真实值折线
    fig.add_trace(go.Scatter(
        x=results_df.index, 
        y=results_df['Actual'], 
        mode='lines+markers',
        name='真实薪资 (Actual)',
        marker=dict(color='blue', size=4)
    ))
    
    # 添加预测值折线
    fig.add_trace(go.Scatter(
        x=results_df.index, 
        y=results_df['Predicted'], 
        mode='lines+markers',
        name='预测薪资 (Predicted)',
        marker=dict(color='orange', size=4),
        opacity=0.8
    ))
    
    # 为了强调“差异对比”，填充两者之间的面积间隔或绘制误差柱
    fig.update_layout(
        title="模型预测薪资与实际薪资对比图 (局部抽样排序验证)",
        xaxis_title="测试集样本编号 (按薪资排序)",
        yaxis_title="薪金 ($)",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

if __name__ == "__main__":
    from src.data_loader import load_all_data
    from src.data_cleaning import clean_data
    from src.analysis import build_master_dataset
    
    print("Loading data for prediction pipeline test...")
    raw = load_all_data()
    cleaned = clean_data(raw)
    master = build_master_dataset(cleaned)
    
    X, y, df_source = prepare_features(master)
    model, X_test, y_test, y_pred = train_and_predict(X, y)
    
    fig_diff = plot_prediction_difference(y_test, y_pred)
    print("预测流与差异可视化完成。")
