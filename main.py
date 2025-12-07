import sys
from data_preview import run_eda 
from classifier2d import run_2d_classifier_plot
from boundary3d import run_3d_boundary_plot
from probability3d import run_3d_probability_plot

def main():
    print("--- 任务一：开始 EDA 分析 ---")
    run_eda()
    
    print("\n--- 任务一：开始 4x4 分类器可视化 ---")
    run_2d_classifier_plot()
    
    print("\n--- 任务二：开始 3D 线性决策边界可视化 ---")
    run_3d_boundary_plot()
    
    print("\n--- 任务三：开始 3D 概率曲面可视化 ---")
    run_3d_probability_plot() # 调用 3D 概率曲面绘图函数

if __name__ == "__main__":
    main()