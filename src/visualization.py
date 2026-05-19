import plotly.express as px
import plotly.graph_objects as go
from src.config import REPORTS_DIR

def plot_correlation_heatmap(corr_matrix):
    """
    绘制相关性热力图 (满足要求5, 6)
    """
    fig = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="员工属性相关性矩阵热力图"
    )
    return fig

def plot_clustering_scatter(df_master, sample_frac=0.05):
    """
    绘制薪资与工龄的聚类散点图 (满足要求3)
    鉴于百万级数据直接绘制散点图会导致浏览器卡顿，通常采样绘图
    """
    # 随机采样以减轻前端渲染压力
    df_sample = df_master.sample(frac=sample_frac, random_state=42) if len(df_master) > 50000 else df_master
    
    fig = px.scatter(
        df_sample, 
        x="tenure_years", 
        y="salary", 
        color="cluster_label", 
        hover_data=['title', 'dept_name'],
        title="员工群体画像聚类散点图 (薪资 vs 司龄 - 采用抽样5%)",
        labels={"tenure_years": "司龄与工作年限 (年)", "salary": "年薪 ($)"},
        opacity=0.6
    )
    return fig

def plot_salary_distribution(df_master):
    """
    绘制薪资分布直方图
    """
    fig = px.histogram(
        df_master, 
        x="salary", 
        nbins=50,
        title="整体薪酬分布图",
        labels={"salary": "薪水 ($)"},
        marginal="box" # 顶部增加箱型图显示游离点
    )
    return fig

def plot_dept_salary_boxplot(df_master):
    """
    绘制各部门薪资的箱线图来反映部门收入差异 (可视化维度之一)
    """
    fig = px.box(
        df_master, 
        x="dept_name", 
        y="salary", 
        color="dept_name",
        title="各部门薪资待遇箱线图差异分析",
        labels={"dept_name": "部门", "salary": "薪水 ($)"}
    )
    fig.update_layout(xaxis={'categoryorder':'median descending'})
    return fig

if __name__ == "__main__":
    from src.data_loader import load_all_data
    from src.data_cleaning import clean_data
    from src.analysis import build_master_dataset, analyze_correlation, perform_clustering
    
    print("加载、清洗并组合数据用于制图测试...")
    raw = load_all_data()
    cleaned = clean_data(raw)
    master = build_master_dataset(cleaned)
    master, _, _ = perform_clustering(master)
    corr = analyze_correlation(master)
    
    # 建立输出目录
    img_dir = REPORTS_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print("生成并导出图片...")
    fig_corr = plot_correlation_heatmap(corr)
    # fig_corr.write_image(str(img_dir / "correlation.png"))  # 依赖 kaleido, 暂时注释以防未安装时测试报错
    
    fig_scatter = plot_clustering_scatter(master)
    # fig_scatter.write_image(str(img_dir / "clusters.png"))
    
    fig_dist = plot_salary_distribution(master)
    # fig_dist.write_image(str(img_dir / "salary_dist.png"))
    
    fig_box = plot_dept_salary_boxplot(master)
    # fig_box.write_image(str(img_dir / "dept_salary.png"))
    
    print("绘图逻辑构建完毕！通过 Plotly 将支持大屏上的交互。")
