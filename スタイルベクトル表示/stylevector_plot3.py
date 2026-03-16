import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import os

# ---------------------------------------------------------
# 1. 設定
# ---------------------------------------------------------
folder_path_1 = './kinpaku_npy'
folder_path_2 = './teikinpaku_npy'
style_vectors_path = 'style_vectors.npy'

label_name_1 = 'Kinpaku'
label_name_2 = 'Tei-kinpaku'

# ---------------------------------------------------------
# 2. 関数・データ読み込み（元のコードと同じため省略可）
# ---------------------------------------------------------
def load_vectors_from_folder(folder_path):
    search_path = os.path.join(folder_path, '*.npy')
    file_list = glob.glob(search_path)
    if len(file_list) == 0: return None
    vector_list = []
    for f in file_list:
        try:
            v = np.load(f)
            if v.ndim == 1: v = v.reshape(1, -1)
            elif v.ndim > 2: v = v.reshape(v.shape[0], -1)
            vector_list.append(v)
        except: pass
    return np.vstack(vector_list) if vector_list else None

data_1 = load_vectors_from_folder(folder_path_1)
data_2 = load_vectors_from_folder(folder_path_2)

# ダミーデータ生成（ファイルがない場合）
if data_1 is None or data_2 is None:
    data_1 = np.random.normal(0, 1, (200, 256))
    data_2 = np.random.normal(1.5, 1, (200, 256))

# style_vectors.npy の読み込み
specific_styles_data = []
specific_styles_names = []
if os.path.exists(style_vectors_path):
    raw_data = np.load(style_vectors_path, allow_pickle=True)
    if raw_data.dtype == object and raw_data.size == 1:
        styles_dict = raw_data.item()
        for name, vec in styles_dict.items():
            specific_styles_names.append(name)
            specific_styles_data.append(vec.flatten())
    else:
        if raw_data.ndim == 1:
            specific_styles_names.append("Neutral")
            specific_styles_data.append(raw_data)
        elif raw_data.ndim == 2:
            for i in range(raw_data.shape[0]):
                name = "Neutral" if i == 0 else f"Style_{i}"
                specific_styles_names.append(name)
                specific_styles_data.append(raw_data[i])
    specific_styles_data = np.array(specific_styles_data)

all_data_list = [data_1, data_2]
if specific_styles_data.size > 0:
    all_data_list.append(specific_styles_data)
data_combined = np.vstack(all_data_list)

labels_combined = np.concatenate([
    np.zeros(len(data_1)),
    np.ones(len(data_2)),
    np.full(len(specific_styles_data), 2) if specific_styles_data.size > 0 else []
])

# ---------------------------------------------------------
# 4. 次元削減
# ---------------------------------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_combined)

perp = min(30, data_combined.shape[0] - 1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
tsne_result = tsne.fit_transform(data_combined)

# ---------------------------------------------------------
# 5. プロット関数 (メモリと重なり対策版)
# ---------------------------------------------------------
def plot_style_analysis(ax, points, labels, title):
    # 背景の点
    ax.scatter(points[labels == 0, 0], points[labels == 0, 1], c='blue', alpha=0.5, s=20, label=label_name_1)
    ax.scatter(points[labels == 1, 0], points[labels == 1, 1], c='orange', alpha=0.5, s=20, label=label_name_2)
    
    # 重心
    c1 = np.mean(points[labels == 0], axis=0)
    ax.scatter(c1[0], c1[1], c='blue', marker='X', s=350, edgecolors='black', label=f'{label_name_1} Center', zorder=5)
    c2 = np.mean(points[labels == 1], axis=0)
    ax.scatter(c2[0], c2[1], c='orange', marker='X', s=350, edgecolors='black', label=f'{label_name_2} Center', zorder=5)

    # 特定のスタイル（Neutral, Style等）
    if specific_styles_data.size > 0:
        style_points = points[labels == 2]
        for i, name in enumerate(specific_styles_names):
            p = style_points[i]
            color = 'red' if name == 'Neutral' else 'purple'
            ax.scatter(p[0], p[1], c=color, marker='*', s=600, edgecolors='black', linewidth=1, zorder=10)
            
            # 【重なり対策】annotateを使用して位置を微調整
            # xytextで(5, 5)ピクセル分だけ右上にずらしています
            offset = (-40, 18) if name == 'Neutral' else (-60, -40)
            ax.annotate(name, (p[0], p[1]),
                        fontsize=32, fontweight='bold', color=color, zorder=15,
                        xytext=offset, textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # 【メモリの調整】軸の目盛りサイズを大きくする
    ax.tick_params(axis='both', which='major', labelsize=18) # ← ここで目盛りの数字を大きく
    
    ax.set_title(title, fontsize=24, pad=15)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.4)

# グラフの全体サイズ
plt.figure(figsize=(20, 9))
plot_style_analysis(plt.subplot(1, 2, 1), pca_result, labels_combined, 'PCA Analysis')
plot_style_analysis(plt.subplot(1, 2, 2), tsne_result, labels_combined, 't-SNE Analysis')

plt.tight_layout()
plt.savefig('style_analysis_large.png', dpi=150) # 保存して確認しやすく
plt.show()
