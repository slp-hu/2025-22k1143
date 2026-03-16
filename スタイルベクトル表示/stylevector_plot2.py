import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import os

# ---------------------------------------------------------
# 1. 設定：ここに2つのフォルダのパスを指定してください
# ---------------------------------------------------------
folder_path_1 = './kinpaku_npy'  # ★1つ目のフォルダパス
folder_path_2 = './teikinpaku_npy'  # ★2つ目のフォルダパス

# グラフの凡例に表示する名前
label_name_1 = 'Folder A'
label_name_2 = 'Folder B'

# ---------------------------------------------------------
# 2. 関数: フォルダ内の全npyファイルを読み込む
# ---------------------------------------------------------
def load_vectors_from_folder(folder_path):
    """
    指定されたフォルダ内の全ての .npy ファイルを読み込み、
    1つの大きな numpy 配列 (N, dim) にして返します。
    """
    # フォルダ内の .npy ファイルのパスを全て取得
    search_path = os.path.join(folder_path, '*.npy')
    file_list = glob.glob(search_path)
    
    if len(file_list) == 0:
        print(f"警告: フォルダ '{folder_path}' に .npy ファイルが見つかりません。")
        return None

    vector_list = []
    print(f"フォルダ '{folder_path}' から {len(file_list)} 個のファイルを読み込み中...")

    for f in file_list:
        try:
            v = np.load(f)
            # ベクトルが (1, 256) や (256,) の場合など形状を統一
            # ここでは (N, features) の2次元配列として扱うため変形
            if v.ndim == 1:
                v = v.reshape(1, -1)
            elif v.ndim > 2:
                v = v.reshape(v.shape[0], -1)
            
            vector_list.append(v)
        except Exception as e:
            print(f"エラー: {f} の読み込みに失敗しました ({e})")

    if len(vector_list) > 0:
        # リストに溜めたベクトルを縦に結合
        combined_data = np.vstack(vector_list)
        print(f"  -> 完了: 合計 {combined_data.shape[0]} サンプル読み込みました。")
        return combined_data
    else:
        return None

# ---------------------------------------------------------
# 3. データの読み込みと結合
# ---------------------------------------------------------
data_1 = load_vectors_from_folder(folder_path_1)
data_2 = load_vectors_from_folder(folder_path_2)

# --- テスト用ダミーデータ生成（フォルダがない場合のみ動作） ---
if data_1 is None and data_2 is None:
    print("\n【注意】指定されたフォルダが見つからないため、デモ用ダミーデータを使用します。")
    data_1 = np.random.normal(0, 1, (200, 256)) # グループA
    data_2 = np.random.normal(2, 1, (200, 256)) # グループB
elif data_1 is None or data_2 is None:
    print("エラー: どちらかのフォルダの読み込みに失敗しました。パスを確認してください。")
    exit()

# ラベル作成 (0: Folder A, 1: Folder B)
labels_1 = np.zeros(len(data_1))
labels_2 = np.ones(len(data_2))

# データを結合
data_combined = np.vstack((data_1, data_2))
labels_combined = np.concatenate((labels_1, labels_2))

# データ数が多すぎる場合のサンプリング (t-SNEの速度対策)
max_samples = 3000
if data_combined.shape[0] > max_samples:
    print(f"\nデータ数が {data_combined.shape[0]} と多いため、ランダムに {max_samples} 個に間引いて表示します。")
    indices = np.random.choice(data_combined.shape[0], max_samples, replace=False)
    data_subset = data_combined[indices]
    labels_subset = labels_combined[indices]
else:
    data_subset = data_combined
    labels_subset = labels_combined

# ---------------------------------------------------------
# 4. 次元削減 (PCA & t-SNE)
# ---------------------------------------------------------
print("\nPCAを実行中...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_subset)

print("t-SNEを実行中... (データ量により数秒〜数分かかります)")
perp = min(30, data_subset.shape[0] - 1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
tsne_result = tsne.fit_transform(data_subset)

# # ---------------------------------------------------------
# 5. プロット (グループ重心 + 全体重心)
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))

def plot_scatter_with_all_centroids(ax, points, labels, title):
    # --- 1. 通常の散布図描画 ---
    # グループ A (青)
    idx_1 = labels == 0
    ax.scatter(points[idx_1, 0], points[idx_1, 1], 
               c='blue', alpha=0.2, s=15, label=label_name_1) # 点は薄く
    
    # グループ B (オレンジ)
    idx_2 = labels == 1
    ax.scatter(points[idx_2, 0], points[idx_2, 1], 
               c='orange', alpha=0.2, s=15, label=label_name_2) # 点は薄く

    # --- 2. 各グループの重心 (X印) ---
    # 青の重心
    if np.any(idx_1):
        center_1 = np.mean(points[idx_1], axis=0)
        ax.scatter(center_1[0], center_1[1], 
                   c='blue', marker='X', s=200, edgecolors='black', linewidth=1.5,
                   label=f'{label_name_1} Center', zorder=5)

    # オレンジの重心
    if np.any(idx_2):
        center_2 = np.mean(points[idx_2], axis=0)
        ax.scatter(center_2[0], center_2[1], 
                   c='orange', marker='X', s=200, edgecolors='black', linewidth=1.5,
                   label=f'{label_name_2} Center', zorder=5)

    # --- 3. すべての点の重心 (赤色の星) ---
    # フィルタリングせず、points 全体の平均をとる
    global_center = np.mean(points, axis=0)
    ax.scatter(global_center[0], global_center[1], 
               c='red', marker='*', s=400, edgecolors='white', linewidth=1.5,
               label='Total Center', zorder=10) # zorder=10 で一番手前に表示

    # --- グラフの装飾 ---
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

# PCA Plot
plot_scatter_with_all_centroids(plt.subplot(1, 2, 1), pca_result, labels_subset, '')
plt.xlabel('PC1'); plt.ylabel('PC2')

# t-SNE Plot
plot_scatter_with_all_centroids(plt.subplot(1, 2, 2), tsne_result, labels_subset, '')
plt.xlabel('Dim 1'); plt.ylabel('Dim 2')

plt.tight_layout()
plt.show()

print("全重心付きで表示完了")
