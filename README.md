# 鸢尾花数据分类与可视化

一款基于 Python 机器学习库（`sklearn`, `matplotlib`）的分类器可视化项目。实现了从二维平面到三维空间的决策边界绘制，重点展示了不同特征工程对线性模型的影响以及非线性概率曲面的可视化。

## 目录
- 简介
- 功能特性
- 运行环境与依赖
- 运行流程说明
- 项目结构

## 功能特性

- **多模型二维可视化**  
  构建 4×4 决策边界矩阵图，对比逻辑回归（结合 RBF、分箱、样条特征工程）与梯度提升树在 Iris 数据集上的表现。

- **3D 线性决策边界**  
  在三维空间中生成二分类数据，并绘制线性分类器的分割平面。

- **3D 概率曲面**  
  使用 Make Moons 数据，通过 SVM 展示非线性预测概率曲面。

- **EDA 数据探索分析**  
  通过 Plotly 交互式图表与 Seaborn 箱线图对原始数据进行初步探索。

## 运行环境与依赖

所需依赖：

- Python 3.x  
- scikit-learn  
- matplotlib  
- numpy  
- pandas  
- seaborn  
- plotly  

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行流程说明

程序启动后按顺序执行任务，关闭当前弹出的图形窗口后会自动进入下一个任务：

1. EDA 分析（Seaborn 箱线图与 Plotly 交互散点图）
2. 二维分类器矩阵图
3. 三维线性决策边界图
4. 三维非线性概率曲面

## 项目结构

```
project/
│── main.py             # 程序主入口，按顺序调度模块
│── data_preview.py     # 数据探索性分析 (EDA)
│── classifier2d.py     # 2D 分类器对比与矩阵绘图
│── boundary3d.py       # 3D 线性决策平面
│── probability3d.py    # 3D 非线性概率曲面与投影
│── requirements.txt    # 依赖文件
```
