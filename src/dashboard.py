from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# 导入我们的处理逻辑
from src.data_loader import load_all_data
from src.data_cleaning import clean_data
from src.analysis import build_master_dataset, analyze_correlation, perform_clustering
from src.prediction import prepare_features, train_and_predict, plot_prediction_difference
from src.visualization import plot_correlation_heatmap, plot_clustering_scatter, plot_salary_distribution, plot_dept_salary_boxplot

print("Initializing Data for Dashboard... (This may take a minute due to large dataset)")
raw = load_all_data()
cleaned = clean_data(raw)
master = build_master_dataset(cleaned)

# 进行聚类与相关性计算
master, _, _ = perform_clustering(master)
corr_matrix = analyze_correlation(master)

# 预先训练模型获取预测数据以供展示
X, y, df_source = prepare_features(master)
model, X_test, y_test, y_pred = train_and_predict(X, y)
fig_pred = plot_prediction_difference(y_test, y_pred)

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
    
    # 预测区 (要求4)
    html.Div([
        dcc.Graph(figure=fig_pred)
    ], style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': 'white', 'padding': '10px'})
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
