# ML Visualization Project

一款基于 Python 机器学习库 (`sklearn`, `matplotlib`) 的分类器可视化项目。实现了从二维平面到三维空间的决策边界绘制，重点展示了不同特征工程对线性模型的影响以及非线性概率曲面的可视化。

## 目录

- 简介
- 功能特性
- 运行环境与依赖
- 快速上手（运行）
- 运行流程说明
- 项目结构

## 功能特性

- **多模型二维可视化**：构建 4x4 矩阵图，对比逻辑回归（结合 RBF、分箱、样条特征工程）与梯度提升树在 Iris 数据集上的决策边界差异。
- **3D 线性决策边界**：在三维空间中生成二分类数据，并绘制线性分类器的分割平面（透明网格）。
- **3D 概率曲面**：针对非线性数据（Make Moons），展示 SVM 模型的预测概率曲面，并包含底面及侧壁的轴向热力图投影。
- **数据探索分析 (EDA)**：集成 Plotly 交互式图表与 Seaborn 箱线图，用于原始数据的初步探索。

## 运行环境与依赖

- Python 3.x
- scikit-learn
- matplotlib
- numpy
- pandas
- seaborn
- plotly

安装依赖：

```powershell
pip install scikit-learn matplotlib numpy pandas seaborn plotly

运行流程说明
程序启动后将按以下顺序执行任务，关闭当前弹出的图形窗口后会自动进入下一任务：

EDA 分析 : 弹出 Seaborn 箱线图与 Plotly 交互式散点图。

2D 矩阵图 : 展示 4x4 的分类器决策边界与概率热力图。

3D 线性边界 : 弹出三维窗口，显示线性分类平面。

3D 概率曲面 : 弹出三维窗口，显示非线性概率曲面及三轴投影。

项目结构（主要文件与说明）
main.py : 程序主入口，按顺序调度各个模块

data_preview.py : 数据探索性分析 (EDA) 逻辑

classifier2d.py : 任务一实现（2D 分类器对比与矩阵绘图）

boundary3d.py : 任务二实现（3D 线性决策平面）

probability3d.py : 任务三实现（3D 非线性概率曲面与投影）

picture.png : 系统架构与数据流示意图
