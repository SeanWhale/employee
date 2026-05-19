from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# 导入我们的处理逻辑
from src.data_loader import load_all_data
from src.data_cleaning import clean_data
from src.analysis import build_master_dataset, dependency_analysis, perform_clustering
from src.prediction import build_monthly_salary_series, train_and_predict, plot_prediction_difference
from src.visualization import (
    plot_correlation_heatmap,
    plot_clustering_scatter,
    plot_salary_distribution,
    plot_dept_salary_boxplot,
    plot_tenure_attrition,
    plot_cluster_profile,
)
from src.external_data import get_simulated_noaa_taxi_data, plot_nyc_taxi_map, build_external_benchmark, plot_benchmark_comparison

print("Initializing Data for Dashboard... (This may take a minute due to large dataset)")
raw = load_all_data()
cleaned = clean_data(raw)
master = build_master_dataset(cleaned)

# 进行聚类与相关性计算
master, clustering_result, _ = perform_clustering(master)
dependency_result = dependency_analysis(master)
corr_matrix = dependency_result['correlation_matrix']
cluster_profile = clustering_result['cluster_profile']

# 预先训练模型获取预测数据以供展示
monthly_salary = build_monthly_salary_series(cleaned['salaries'])
model, validation_df, forecast_metrics = train_and_predict(monthly_salary)
fig_pred = plot_prediction_difference(validation_df, forecast_metrics)
fig_attrition = plot_tenure_attrition(master)
fig_cluster_profile = plot_cluster_profile(cluster_profile)

external_weather_df = get_simulated_noaa_taxi_data()
fig_map = plot_nyc_taxi_map(external_weather_df)
benchmark_df = build_external_benchmark(master)
fig_benchmark = plot_benchmark_comparison(benchmark_df)

anova_p = dependency_result['anova_salary_by_department']['p_value'] if dependency_result['anova_salary_by_department'] else None
anova_text = f"{anova_p:.3g}" if anova_p is not None else "NA"
dependency_summary_text = (
    f"偏相关(薪资~工龄|年龄): {dependency_result['partial_correlation_salary_tenure_given_age']['coefficient']:.3f}, "
    f"p={dependency_result['partial_correlation_salary_tenure_given_age']['p_value']:.3g}; "
    f"ANOVA p={anova_text}; "
    f"聚类模型对比: {clustering_result['scores']}"
)

def get_dynamic_time_figure(cleaned_data):
    """
    要求 8: 构建随时间产生数据变化和图像变化的动态视图
    计算每年各部门的平均薪资，用作动画条形图
    """
    sal = cleaned_data['salaries'].copy()
    dept_emp = cleaned_data['dept_emp'].copy()
    dept = cleaned_data['departments'].copy()
    
    # 抽取生效年份
    sal['year'] = sal['from_date'].dt.year
    sal = sal.dropna(subset=['year']).copy()
    sal['year'] = sal['year'].astype(int)
    
    # 关联部门
    merged = pd.merge(sal, dept_emp[['emp_no', 'dept_no']], on='emp_no', how='left')
    merged = pd.merge(merged, dept, on='dept_no', how='left')
    merged = merged.dropna(subset=['dept_name', 'salary'])
    
    # 按年和部门分组计算均薪
    yearly_dept_salary = merged.groupby(['year', 'dept_name'])['salary'].mean().reset_index()
    yearly_dept_salary = yearly_dept_salary[(yearly_dept_salary['year'] >= 1990) & (yearly_dept_salary['year'] <= 2002)] # 限制年份范围，保证动画效果平滑
    yearly_dept_salary = yearly_dept_salary.sort_values(by=['year', 'dept_name'])
    
    # 构建动画视图
    fig = px.bar(
        yearly_dept_salary, 
        x="dept_name", 
        y="salary", 
        color="dept_name",
        animation_frame="year", 
        animation_group="dept_name",
        range_y=[30000, 100000],
        title="各部门平均薪资年度动态变迁图 (要求8: 时间动态)",
        labels={"salary": "平均年薪 ($)", "dept_name": "业务部门"}
    )
    return fig

fig_dynamic_time = get_dynamic_time_figure(cleaned)
dept_options = [{'label': '所有部门 (All)', 'value': 'ALL'}] + [{'label': d, 'value': d} for d in master['dept_name'].dropna().unique()]

app = Dash(__name__, title="大型企业 HR 员工百万级数据分析舱")

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f4f6f9'}, children=[
    html.H1("🌟 HR 智能分析综合数据大屏", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.P("基于百万级数据的清理、相依性/聚类分析、机器学习推导验证及动态可视化", style={'textAlign': 'center', 'color': '#7f8c8d'}),
    
    html.Hr(),
    
    # 交互控制区 (要求9: 人机互动)
    html.Div(style={'width': '30%', 'margin': '0 auto', 'paddingBottom': '20px'}, children=[
        html.Label("🔍 人机交互 - 下拉图表筛选部门：", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='dept-dropdown',
            options=dept_options,
            value='ALL',
            clearable=False,
            style={'marginTop': '10px'}
        )
    ]),

    # 动态与时间动画区 (要求8)
    html.Div([
        dcc.Graph(figure=fig_dynamic_time)
    ], style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'marginBottom': '30px', 'backgroundColor': 'white', 'padding': '10px'}),
    
    # 图表区第一排
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
        html.Div(dcc.Graph(id='dist-plot'), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'}),
        html.Div(dcc.Graph(id='box-plot'), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'})
    ]),
    
    # 图表区第二排
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
        html.Div(dcc.Graph(figure=plot_correlation_heatmap(corr_matrix)), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'}),
        html.Div(dcc.Graph(id='scatter-plot'), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'})
    ]),

    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
        html.Div(dcc.Graph(figure=fig_attrition), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'}),
        html.Div(dcc.Graph(figure=fig_cluster_profile), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'})
    ]),

    # 预测区 (要求4)
    html.Div([
        dcc.Graph(figure=fig_pred)
    ], style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': 'white', 'padding': '10px', 'marginBottom': '20px'}),

    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
        html.Div(dcc.Graph(figure=fig_benchmark), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'}),
        html.Div(dcc.Graph(figure=fig_map), style={'width': '49%', 'backgroundColor': 'white', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '10px'})
    ]),

    html.Div(style={'backgroundColor': 'white', 'padding': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}, children=[
        html.H4("依赖分析摘要", style={'marginBottom': '10px'}),
        html.P(dependency_summary_text),
    ])
])

# 回调函数响应交互
@app.callback(
    Output('dist-plot', 'figure'),
    Output('box-plot', 'figure'),
    Output('scatter-plot', 'figure'),
    Input('dept-dropdown', 'value')
)
def update_graphs(selected_dept):
    if selected_dept == 'ALL':
        df_filtered = master
    else:
        df_filtered = master[master['dept_name'] == selected_dept]
        
    f1 = plot_salary_distribution(df_filtered)
    f2 = plot_dept_salary_boxplot(df_filtered)
    f3 = plot_clustering_scatter(df_filtered)
    return f1, f2, f3

if __name__ == '__main__':
    app.run(debug=True, port=8050)
