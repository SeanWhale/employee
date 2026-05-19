import sys
import os
from pathlib import Path

# 将项目根目录加入到执行寻址路径
sys.path.append(str(Path(__file__).resolve().parent))

from src.dashboard import app

if __name__ == '__main__':
    print("============= 准备启动 HR 分析综合大屏界面 =============")
    print("-> 这将拉取百万级数据进行重组与机器学习建模，请等待几十秒。")
    print("-> 数据完成后，浏览器可以通过 http://127.0.0.1:8050/ 进行访问。")
    app.run(debug=False, port=8050)
