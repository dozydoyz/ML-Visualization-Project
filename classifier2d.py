from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer, SplineTransformer, StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

# 定义四种分类器配置 
def get_classifier_configs():
  
    
    classifiers = [
        # 1. Logistic Regression RBF features
        {
            "name": "Logistic regression\n(RBF features)", # 标题分行
            "model": make_pipeline(
                StandardScaler(),
                RBFSampler(gamma=1, n_components=100, random_state=42),
                LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=1.0)
            )
        },

        # 2. Gradient Boosting
        {
            "name": "Gradient Boosting", 
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        },

        # 3. Logistic Regression binned features
        {
            "name": "Logistic regression\n(binned features)", 
            "model": make_pipeline(
                KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='uniform'), 
                LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            )
        },

        # 4. Logistic Regression spline features
        {
            "name": "Logistic regression\n(spline features)", 
            "model": make_pipeline(
                StandardScaler(),
                SplineTransformer(n_knots=5, degree=3, include_bias=False),
                LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            )
        }
    ]
    MAX_CLASS_COLORS = ['#ADD8E6', '#FF7F00', '#4CAF50'] # 浅蓝, 橙, 绿
    
    return classifiers, MAX_CLASS_COLORS

def run_2d_classifier_plot():
    # 加载Iris数据集，仅使用后两个特征 (x2, x3)
    iris = load_iris()
    X = iris.data[:, 2:]  
    y = iris.target

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ----------------------------------------------------
    # 网格和颜色设置
    # ----------------------------------------------------
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2 
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    
    grid_n = 200 
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n), 
                         np.linspace(y_min, y_max, grid_n))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    
    configs, MAX_CLASS_COLORS = get_classifier_configs()
    CLASS_CMAP = mcolors.ListedColormap(MAX_CLASS_COLORS)
    
    
    PROB_CMAP = plt.cm.get_cmap('Blues_r') # 反转 Blues 渐变：浅蓝到深蓝
    
    # ----------------------------------------------------
    # 创建 4x4 大图
    # ----------------------------------------------------
    fig, axs = plt.subplots(4, 4, figsize=(16, 16), dpi=100, constrained_layout=True)
    
    # 定义列标题
    column_titles = ['Class 0', 'Class 1', 'Class 2', 'Max class']
    
    # 循环：模型 (行)
    for row_idx, config in enumerate(configs):
        name = config['name']
        model = config['model']
        print(f"正在训练和绘制行 {row_idx}: {name.replace('\n', ' ')}...")
        model.fit(X_train, y_train)
        
        # 1. 计算概率和预测结果
        probs = model.predict_proba(X_grid)
        probs = probs.reshape(xx.shape[0], xx.shape[1], 3)
        Z = np.argmax(probs, axis=2) 
        
        
        # 2. 绘制概率图 (列 0, 1, 2)
        for class_idx in range(3):
            ax = axs[row_idx, class_idx]
            class_prob = probs[:, :, class_idx]
            
        
            ax.contourf(xx, yy, class_prob, levels=50, cmap='Blues', vmin=0, vmax=1)
            
            # 绘制数据点：白色填充，黑色边缘
            ax.scatter(X[:, 0], X[:, 1], c='white', edgecolors='k', marker='o', s=30)
                       
            # ----------------------------------------------------
            # 标注和格式调整
            # ----------------------------------------------------
            if row_idx == 0:
                ax.set_title(column_titles[class_idx], fontsize=14)

            # 设置行标注
            if class_idx == 0:
                ax.set_yticks([]) 
                ax.set_ylabel(name, rotation=90, fontsize=12, labelpad=10, ha='right')
            
            # 隐藏坐标刻度
            ax.set_xticks([])
            ax.set_yticks([])
            
        # 3. 绘制 Max Class (决策边界) 图 (列 3)
        ax = axs[row_idx, 3]
        
        ax.contourf(xx, yy, Z, levels=np.arange(-0.5, len(MAX_CLASS_COLORS)), cmap=CLASS_CMAP, alpha=0.9)
       
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=30, cmap=CLASS_CMAP)
        
    
        if row_idx == 0:
            ax.set_title(column_titles[3], fontsize=14)
        
       
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')
        
        # 确保所有图的坐标范围一致
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
    # 调整布局
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_2d_classifier_plot()