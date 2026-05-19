from flask import Flask, render_template_string
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Scatter, Grid, Map, Boxplot
from pyecharts.globals import ThemeType
import pandas as pd
from src.data_loader import load_all_data
from src.data_cleaning import clean_data
from src.analysis import build_master_dataset, analyze_correlation, perform_clustering
from src.external_data import get_simulated_noaa_taxi_data
import os

app = Flask(__name__)

print("Initializing data... (might take some time)")
raw = load_all_data()
cleaned = clean_data(raw)
master = build_master_dataset(cleaned)
master_clustered, kmeans, scaler = perform_clustering(master)
# Optional: downsample for charting
df_sample = master_clustered.sample(frac=0.05, random_state=42) if len(master_clustered) > 5000 else master_clustered

def create_echarts_dashboard():
    # 1. 薪资分布柱状图 (模拟直方图)
    salaries = df_sample['salary'].dropna().values.tolist()
    # 简单分桶
    bins = pd.cut(salaries, bins=10)
    counts = pd.value_counts(bins).sort_index()
    x_axis = [str(interval) for interval in counts.index]
    y_axis = counts.values.tolist()
    
    bar_hist = (
        Bar()
        .add_xaxis(x_axis)
        .add_yaxis("人数", y_axis)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="薪酬分布", pos_top="5%"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
            legend_opts=opts.LegendOpts(pos_top="10%")
        )
    )

    # 2. 薪资 vs 工龄聚类散点图
    scatter = Scatter()
    scatter.add_xaxis(df_sample["tenure_years"].round(2).tolist())
    for cluster_id in df_sample['cluster_label'].unique():
        cluster_data = df_sample[df_sample['cluster_label'] == cluster_id]
        scatter.add_yaxis(
            series_name=f"Cluster {cluster_id}",
            y_axis=cluster_data["salary"].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
        )
    scatter.set_global_opts(
        title_opts=opts.TitleOpts(title="薪资与司龄聚类分析", pos_top="50%"),
        xaxis_opts=opts.AxisOpts(type_="value", name="司龄(年)"),
        yaxis_opts=opts.AxisOpts(type_="value", name="薪资($)"),
        legend_opts=opts.LegendOpts(pos_top="55%")
    )

    # 组合为 Grid 大屏显示
    grid = (
        Grid(init_opts=opts.InitOpts(width="100%", height="800px", theme=ThemeType.LIGHT))
        .add(bar_hist, grid_opts=opts.GridOpts(pos_bottom="60%"))
        .add(scatter, grid_opts=opts.GridOpts(pos_top="60%"))
    )

    return grid.render_embed()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ECharts HR 动态数据大屏</title>
    <script src="https://assets.pyecharts.org/assets/v5/echarts.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }
        .header { text-align: center; color: #2c3e50; }
        .chart-container { background: white; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px;}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌟 HR ECharts 实时动态可视化大屏</h1>
    </div>
    <div class="chart-container">
        {{ chart_html|safe }}
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    chart_html = create_echarts_dashboard()
    return render_template_string(HTML_TEMPLATE, chart_html=chart_html)

if __name__ == "__main__":
    print("============= 启动 Flask + ECharts 大屏 =============")
    print("访问 http://127.0.0.1:5000/ 查看效果")
    app.run(debug=True, port=5000)
