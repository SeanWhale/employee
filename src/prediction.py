import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_monthly_salary_series(salaries_df):
    """
    按月汇总历史薪资，用于时间序列预测验证。
    """
    ts = salaries_df.dropna(subset=['from_date', 'salary']).copy()
    ts['year_month'] = ts['from_date'].dt.to_period('M').dt.to_timestamp()
    return ts.groupby('year_month', as_index=False)['salary'].mean().sort_values('year_month')

def train_and_predict(monthly_salary_df, test_periods=24):
    """
    使用历史月份训练线性趋势模型，并用后续月份做验证。
    """
    logging.info("Training time-based salary forecasting model...")
    if len(monthly_salary_df) <= test_periods + 2:
        raise ValueError("历史月份不足，无法进行训练/验证切分。")

    df = monthly_salary_df.copy().reset_index(drop=True)
    df['t'] = np.arange(len(df))
    split_idx = len(df) - test_periods
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    model = LinearRegression()
    model.fit(train_df[['t']], train_df['salary'])
    test_pred = model.predict(test_df[['t']])

    validation = test_df[['year_month', 'salary']].copy()
    validation['predicted_salary'] = test_pred
    validation['error'] = validation['salary'] - validation['predicted_salary']
    mae = float(mean_absolute_error(validation['salary'], validation['predicted_salary']))
    rmse = float(np.sqrt(mean_squared_error(validation['salary'], validation['predicted_salary'])))
    mape = float((np.abs(validation['error']) / validation['salary'].clip(lower=1)).mean())

    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape}
    logging.info(f"Forecast validation -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.4f}")
    return model, validation, metrics

def plot_prediction_difference(validation_df, metrics):
    """
    可视化：真实值 vs 预测值 + 误差条图
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=validation_df['year_month'],
        y=validation_df['salary'],
        mode='lines+markers',
        name='实际均薪',
        marker=dict(color='blue', size=4)
    ))

    fig.add_trace(go.Scatter(
        x=validation_df['year_month'],
        y=validation_df['predicted_salary'],
        mode='lines+markers',
        name='预测均薪',
        marker=dict(color='orange', size=4),
        opacity=0.8
    ))
    fig.add_trace(go.Bar(
        x=validation_df['year_month'],
        y=validation_df['error'],
        name='误差 (实际-预测)',
        yaxis='y2',
        opacity=0.35,
        marker=dict(color='green')
    ))

    fig.update_layout(
        title=(
            "历史薪资趋势预测验证（时间切分）"
            f" | MAE={metrics['mae']:.1f}, RMSE={metrics['rmse']:.1f}, MAPE={metrics['mape']:.2%}"
        ),
        xaxis_title="月份",
        yaxis_title="平均薪资 ($)",
        yaxis2=dict(title='误差', overlaying='y', side='right'),
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

    monthly = build_monthly_salary_series(cleaned['salaries'])
    model, validation, metrics = train_and_predict(monthly)
    fig_diff = plot_prediction_difference(validation, metrics)
    print("预测流与差异可视化完成。")
