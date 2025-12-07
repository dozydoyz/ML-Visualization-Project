import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris

def run_eda():
   
    try:
        df = sns.load_dataset('iris')
    except:
   
        iris_temp = load_iris()
        df = pd.DataFrame(iris_temp.data, columns=iris_temp.feature_names)
        df.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in df.columns]
        df['species'] = pd.Categorical.from_codes(iris_temp.target, iris_temp.target_names)
    
    print(df[50:100])

    # 数据预处理
    df_clean = df.dropna()

    # 创建多个子图：箱线图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.boxplot(x='species', y='sepal_length', data=df_clean, ax=axes[0, 0])
    axes[0, 0].set_title('Sepal Length by Species')

    sns.boxplot(x='species', y='sepal_width', data=df_clean, ax=axes[0, 1])
    axes[0, 1].set_title('Sepal Width by Species')

    sns.boxplot(x='species', y='petal_length', data=df_clean, ax=axes[1, 0])
    axes[1, 0].set_title('Petal Length by Species')

    sns.boxplot(x='species', y='petal_width', data=df_clean, ax=axes[1, 1])
    axes[1, 1].set_title('Petal Width by Species')

    plt.tight_layout()
    plt.show()

    # 使用Plotly绘制交互式散点图
    print("生成 Plotly 交互式图表...")
    pairs = [
        ('sepal_length', 'sepal_width'), ('sepal_length', 'petal_length'),
        ('sepal_length', 'petal_width'), ('sepal_width', 'petal_length'),
        ('sepal_width', 'petal_width'), ('petal_length', 'petal_width')
    ]
    
    for x_col, y_col in pairs:
        fig = px.scatter(df_clean, x=x_col, y=y_col, color='species', 
                           title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}")
        fig.show()

if __name__ == "__main__":
    run_eda()