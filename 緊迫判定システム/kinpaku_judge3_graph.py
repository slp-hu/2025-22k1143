import librosa
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. 特徴抽出関数 (MFCC)
# ---------------------------------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0) # 時間平均をとる
        return mfcc_mean
    except Exception as e:
        print(f"読み込みエラー: {file_path} ({e})")
        return None

# ---------------------------------------------------------
# 2. 学習関数 (データを読んでAIを作る)
# ---------------------------------------------------------
def train_model():
    X = [] # 入力データ
    y = [] # 正解ラベル
    
    # フォルダの確認
    if not os.path.exists("dataset"):
        print("エラー: 'dataset' フォルダが見つかりません。")
        return None

    print("--- 1. 学習データを読み込んでいます... ---")
    
    # Normal (通常) フォルダ
    normal_files = glob.glob("dataset/normal/*.wav")
    print(f"  通常データ(normal): {len(normal_files)}件")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(0) # 0 = Normal

    # Urgent (緊急) フォルダ
    urgent_files = glob.glob("dataset/urgent/*.wav")
    print(f"  緊急データ(urgent): {len(urgent_files)}件")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(1) # 1 = Urgent

    if len(X) == 0:
        print("エラー: wavファイルが一つも見つかりません。")
        return None

    # AIモデル作成 (ランダムフォレスト)
    print("--- 2. AIを学習させています... ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✅ 学習完了！")
    return model

# ---------------------------------------------------------
# 3. グラフ化関数 (ブラックボックス解明)
# ---------------------------------------------------------
def visualize_blackbox(model):
    print("\n--- 3. ブラックボックスの中身（判断基準）を可視化します ---")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # 重要度順にソート
    
    # 上位10個を表示
    top_k = 10
    top_indices = indices[:top_k]
    
    plt.figure(figsize=(10, 6))
    plt.title("Important Features (What the AI is listening to)")
    plt.bar(range(top_k), importances[top_indices], align="center", color='skyblue')
    plt.xticks(range(top_k), [f"MFCC_{i}" for i in top_indices])
    plt.xlabel("MFCC Index (Voice Characteristic)")
    plt.ylabel("Importance")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    print("グラフを表示します...")
    plt.show()
    print("→ 棒グラフが高い項目(MFCC番号)ほど、津波と地震を見分けるのに役立ちました。")

# ---------------------------------------------------------
# 4. 未知データの判定関数
# ---------------------------------------------------------
def predict_new_file(model, target_file):
    print(f"\n--- 4. テスト判定: {target_file} ---")
    if not os.path.exists(target_file):
        print("テスト用のファイルが見つかりません。")
        return

    feat = extract_features(target_file)
    if feat is not None:
        feat = feat.reshape(1, -1)
        probs = model.predict_proba(feat)[0]
        urgency_score = probs[1] # 緊急(1)である確率
        
        print(f"🚨 緊急度スコア: {urgency_score * 100:.1f}%")
        if urgency_score > 0.5:
            print("判定: ⚠️ 緊急 (Urgent)")
        else:
            print("判定: ✅ 通常 (Normal)")

# =========================================================
# メイン実行ブロック (ここからスタート)
# =========================================================
if __name__ == "__main__":
    # 1. 学習を実行
    my_brain = train_model()
    
    # 学習が成功した場合のみ、続きを実行
    if my_brain is not None:
        
        # 2. 中身をグラフ化
        visualize_blackbox(my_brain)
        
        # 3. テストファイルを判定 (ファイル名は適宜変更してください)
        test_filename = "exp3_1_high.wav" 
        predict_new_file(my_brain, test_filename)
