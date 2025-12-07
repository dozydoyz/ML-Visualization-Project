from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors # 确保导入 mcolors

def run_3d_boundary_plot():
    # ----------------------------------------------------
    # 1. 准备数据和模型
    # ----------------------------------------------------
    
    # 创建一个具有3个特征的、有噪声的二分类数据集
    X, y = make_classification(n_samples=500, n_features=3, n_informative=3, 
                               n_redundant=0, n_classes=2, random_state=42, 
                               n_clusters_per_class=1, flip_y=0.1)
    
    # 归一化数据 (有利于线性模型训练)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用线性分类器
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # 获取模型的系数和截距 (W1, W2, W3, b)
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    # ----------------------------------------------------
    # 2. 计算决策平面
    # ----------------------------------------------------
    
    # 使用固定的坐标轴范围来计算平面，确保平面覆盖了我们想要显示的区域 [-2, 2]
    x1_min, x1_max = -2.0, 2.0 
    x2_min, x2_max = -2.0, 2.0 
    
    x1_surf = np.linspace(x1_min, x1_max, 50)
    x2_surf = np.linspace(x2_min, x2_max, 50)
    x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)
    
    # 计算决策平面上的 X3 值: X3 = -(W1*X1 + W2*X2 + b) / W3
    epsilon = 1e-10 
    
    if abs(coef[2]) < epsilon:
        print("警告: 决策边界几乎垂直于 X3 轴，可能无法完美绘制平面。")
        x3_surf = np.zeros_like(x1_surf) 
    else:
        x3_surf = -(coef[0] * x1_surf + coef[1] * x2_surf + intercept) / coef[2]

    # ----------------------------------------------------
    # 3. 绘图可视化
    # ----------------------------------------------------
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义颜色映射 (模仿参考图的红蓝两类)
    cmap = mcolors.ListedColormap(['#FF0000', '#0000FF']) # 红, 蓝
    
    # 绘制数据点
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                         c=y, cmap=cmap, marker='o', s=50, edgecolors='k')
    
    # 绘制决策平面
    ax.plot_surface(x1_surf, x2_surf, x3_surf, 
                    color='gray', alpha=0.6, linewidth=0, antialiased=False)
    
    # *** 关键修改：设置坐标轴范围为 [-2.0, 2.0] ***
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)
    
    # 设置轴标签
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_zlabel('X3', fontsize=12)
    
    ax.set_title('3D Linear Decision Boundary', fontsize=14)
    
    # 调整视角 (使其看起来更像参考图)
    ax.view_init(elev=15, azim=45) 
    
    plt.show()

if __name__ == "__main__":
    run_3d_boundary_plot()