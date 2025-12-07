from sklearn.datasets import make_moons
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def run_3d_probability_plot():
    # ==========================================
    # 1. 数据准备与模型训练
    # ==========================================
    # 创建非线性数据
    X, y = make_moons(n_samples=500, noise=0.1, random_state=0)
    
    # 放大数据范围，使其适配 [-40, 40] 的坐标系
    X_data = X * 20 
    
    # 训练 SVM (参数微调以获得漂亮的曲面)
    model = SVC(kernel='rbf', gamma=0.002, C=50.0, probability=True, random_state=0)
    model.fit(X_data, y)
    
    # ==========================================
    # 2. 网格计算
    # ==========================================
    # 定义坐标轴范围
    x_min, x_max = -40.0, 40.0
    y_min, y_max = -40.0, 40.0
    z_min, z_max = -100.0, 100.0
    
    # 生成网格
    grid_res = 50 # 网格密度，太密会导致网格线糊在一起，50左右比较像原图
    xx = np.linspace(x_min, x_max, grid_res)
    yy = np.linspace(y_min, y_max, grid_res)
    X_surf, Y_surf = np.meshgrid(xx, yy)
    XY_grid = np.c_[X_surf.ravel(), Y_surf.ravel()]
    
    # 预测概率并映射到 Z 轴 [-100, 100]
    Probs = model.predict_proba(XY_grid)[:, 1]
    Z_score = (Probs - 0.5) * 200 
    Z_surf = Z_score.reshape(X_surf.shape)

    # ==========================================
    # 3. 绘图 (严格按照你的要求修改)
    # ==========================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用 Coolwarm colormap (蓝-白-红)
    cmap = plt.cm.coolwarm
    
    # -------------------------------------------------------
    # A. 绘制三个轴的热力图投影 (Projections)
    # -------------------------------------------------------
    # 1. 底面投影 (Z = -100)
    ax.contourf(X_surf, Y_surf, Z_surf, zdir='z', offset=z_min, cmap=cmap, alpha=0.6)
    
    # 2. 左侧墙投影 (X = -40) -> 注意：offset设为 x_min
    ax.contourf(Y_surf, Z_surf, Z_surf, zdir='x', offset=x_min, cmap=cmap, alpha=0.6)
    
    # 3. 后侧墙投影 (Y = 40)  -> 注意：offset设为 y_max
    ax.contourf(X_surf, Z_surf, Z_surf, zdir='y', offset=y_max, cmap=cmap, alpha=0.6)

    # -------------------------------------------------------
    # B. 绘制中间透明网格线 (Transparent Grid at Z=0)
    # -------------------------------------------------------
    # 这是一个平面，高度为 0
    Z_zero = np.zeros_like(X_surf)
    
    # 关键：使用 plot_wireframe 画网格，不画 surface，就是透明的
    # color='blue' 或 'gray'，rstride/cstride 控制网格稀疏度
    ax.plot_wireframe(X_surf, Y_surf, Z_zero, 
                      rstride=5, cstride=5, 
                      color='#000080', alpha=0.3, linewidth=0.5)

    # -------------------------------------------------------
    # C. 绘制 3D 概率曲面 (Probability Surface)
    # -------------------------------------------------------
    # 绘制实体曲面，带透明度
    ax.plot_surface(X_surf, Y_surf, Z_surf, 
                    cmap=cmap, 
                    rstride=1, cstride=1, # 采样步长
                    alpha=0.6,            # 半透明
                    linewidth=0,          # 不显示三角网格线，只显示颜色
                    antialiased=False)
    
    # 为了增加原图那种“网格感”，我们在曲面上再叠加一层稀疏的线框
    ax.plot_wireframe(X_surf, Y_surf, Z_surf,
                      rstride=5, cstride=5,
                      color='gray', alpha=0.3, linewidth=0.5)

    # -------------------------------------------------------
    # D. 设置视口和坐标轴
    # -------------------------------------------------------
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 调整视角，使其更容易看到三个投影面
    ax.view_init(elev=30, azim=-60)
    
    plt.show()

if __name__ == "__main__":
    run_3d_probability_plot()