import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import os

# ========== 样式设置 ==========
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15

# ========== 文件路径和标签 ==========
file_names = ['preds_PF.csv', 'preds_S.csv', 'preds_sigma.csv', 'preds_tc.csv', 'preds_zt.csv']
label_mapping = {
    'PF': r'${\rm PF} \times 10^5$',
    'S': r'$S$',
    'sigma': r'$\sigma$',
    'tc': r'$\kappa$',
    'zt': r'$zT$'
}
subfigure_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

# ========== 图像设置 ==========
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, file_name in enumerate(file_names):
    df = pd.read_csv(file_name)
    prefix = file_name.split('_')[1].split('.')[0]  # 'PF', 'S', etc.

    # 使用种子固定划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=2024)

    real_train = train_df['Real'].values
    pred_train = train_df['Pred'].values
    real_test = test_df['Real'].values
    pred_test = test_df['Pred'].values

    r2_train = r2_score(real_train, pred_train)
    r2_test = r2_score(real_test, pred_test)

    # 合并标注 Train/Test
    df_train = pd.DataFrame({'True': real_train, 'Predicted': pred_train, 'Dataset': 'Train'})
    df_test = pd.DataFrame({'True': real_test, 'Predicted': pred_test, 'Dataset': 'Test'})
    df_all = pd.concat([df_train, df_test])

    ax = axs[i]
    sns.scatterplot(data=df_all, x='True', y='Predicted', hue='Dataset', ax=ax,
                    palette={'Train': '#407BD0', 'Test': '#A32A31'}, s=70, alpha=0.6, legend=False)
    sns.regplot(data=df_train, x='True', y='Predicted', scatter=False, ax=ax, color='#407BD0')
    sns.regplot(data=df_test, x='True', y='Predicted', scatter=False, ax=ax, color='#A32A31')

    # 对角线参考线
    min_val = min(df_all['True'].min(), df_all['Predicted'].min())
    max_val = max(df_all['True'].max(), df_all['Predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # 标签与分数
    ax.set_xlabel(rf'{label_mapping[prefix]}$_{{\rm Exp}}$')
    ax.set_ylabel(rf'{label_mapping[prefix]}$_{{\rm DopNet}}$')
    ax.text(0.05, 0.9, f'Train $R^2$ = {r2_train:.3f}', transform=ax.transAxes, color='#407BD0', fontsize=15)
    ax.text(0.05, 0.8, f'Test $R^2$ = {r2_test:.3f}', transform=ax.transAxes, color='#A32A31', fontsize=15)

    # 添加小标号 (a) - 外部左上角
    ax.text(-0.15, 1.05, subfigure_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold')

    # 美化边框和刻度
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    ax.tick_params(direction='in', length=6, width=1.25, which='both')

# 隐藏第六个子图（空图）
axs[-1].axis('off')

plt.tight_layout()
plt.show()
