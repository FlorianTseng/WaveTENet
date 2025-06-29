import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# 设置全局字体和绘图风格
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15

# 设置文件夹和文件名
folder_path = ''
file_names = [
    'loss_PF.csv',
    'loss_sigma.csv',
    'loss_S.csv',
    'loss_tc.csv',
    'loss_zt.csv'
]

# 图标题映射（注意：不含后缀）
label_mapping = {
    'PF': r'(a) PF $\times$ 1e5',
    'sigma': '(b) $\sigma$',
    'S': '(c) $S$',
    'tc': '(d) $\kappa$',
    'zt': '(e) $zT$'
}

# 创建 2x3 子图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

for i, file_name in enumerate(file_names):
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)

    # 按 epoch 聚合
    df_avg = df.groupby('epoch')[['train_loss', 'test_loss']].mean().reset_index()
    epochs = df_avg['epoch']
    train_loss = df_avg['train_loss']
    test_loss = df_avg['test_loss']

    prefix = file_name.split('_')[1].split('.')[0]
    title_base = label_mapping.get(prefix, 'Unknown')
    title = f"{title_base} - DopNet on ESTM"

    ax = axs[i]
    ax.plot(epochs, train_loss, color='#4d79a6', linewidth=2, label='Train Loss')
    ax.plot(epochs, test_loss, color='#e05759', linewidth=2, label='Test Loss')

    if prefix == 'PF':
        ax.set_yscale('log')

    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_title(title)
    ax.set_xlim([epochs.min(), epochs.max()])
    ax.tick_params(direction='in', length=6, width=1.25, which='both')
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    ax.legend(frameon=False)

# 第六张图留空
axs[-1].axis('off')

plt.tight_layout()
plt.show()
