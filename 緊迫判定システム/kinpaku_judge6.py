import librosa
import numpy as np
import os
import glob
import random # ランダムに選ぶために追加
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 特徴量を抽出する関数（強化版）
# ==========================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # MFCC（声色の特徴）を抽出
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # ★改良点: 平均だけでなく「標準偏差(std)」も追加
        # 平均(mean): どんな声か（高い、低い、太いなど）
        # 標準偏差(std): どれくらい激しく変化しているか（抑揚、必死さ）
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        
        # 2つを連結して80次元の特徴量にする
        return np.concatenate([mfcc_mean, mfcc_std])
        
    except Exception as e:
        print(f"エラー ({os.path.basename(file_path)}): {e}")
        return None

# ==========================================
# 2. 学習する関数（データ数自動調整機能付き）
# ==========================================
def train_my_model():
    X = []
    y = []

    print("--- 📚 学習データを読み込み中 ---")
    
    # ファイルリストを取得
    normal_files = glob.glob("dataset/normal/*.wav")
    urgent_files = glob.glob("dataset/urgent/*.wav")
    other_files = glob.glob("dataset/other/*.wav")

    # データ数をチェック
    n_normal = len(normal_files)
    n_urgent = len(urgent_files)
    n_other_total = len(other_files)

    if n_normal == 0 or n_urgent == 0:
        print("❌ エラー: NormalまたはUrgentのデータがありません。")
        return None

    # ★重要: 'Other'のデータを、'Urgent'や'Normal'の多い方に合わせる（最大でも2倍程度まで）
    # これにより「Otherが多すぎて全部Otherになる」のを防ぎます
    target_count = max(n_normal, n_urgent) * 2
    
    if n_other_total > target_count:
        print(f"ℹ️ JVSなどのデータが多すぎます({n_other_total}件)。バランスを取るため {target_count} 件だけランダムに使います。")
        other_files = random.sample(other_files, target_count)
    
    # --- 読み込み処理 ---
    
    # Normal
    print(f"   ✅ Normal (0): {len(normal_files)} ファイル")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(0) 
            
    # Urgent
    print(f"   ⚠️ Urgent (1): {len(urgent_files)} ファイル")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(1) 

    # Other
    print(f"   ⚪ Other  (2): {len(other_files)} ファイル")
    for file in other_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(2)

    X = np.array(X)
    y = np.array(y)

    print("🤖 AIモデルを作成中 (RandomForest)...")
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)

    print("✅ 学習完了！\n")
    return classifier

# ==========================================
# 3. 判定関数
# ==========================================
def predict_batch(classifier, target_folder):
    files = glob.glob(os.path.join(target_folder, "*.wav"))
    
    if len(files) == 0:
        print(f"⚠️ '{target_folder}' にwavファイルがありません。")
        return

    print(f"--- 📂 一括判定開始: {target_folder} ---")
    print(f"{'ファイル名':<20} | {'判定':<12} | {'詳細'}")
    print("-" * 60)

    count_normal = 0
    count_urgent = 0
    count_other = 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            
            probs = classifier.predict_proba(feat)[0]
            pred_label = np.argmax(probs)
            max_prob = probs[pred_label]

            # 確率の表示用文字列
            prob_str = f"通常:{probs[0]:.2f} 緊急:{probs[1]:.2f} 他:{probs[2]:.2f}"

            if pred_label == 0:
                result_str = "✅ 通常"
                count_normal += 1
            elif pred_label == 1:
                result_str = "⚠️ 緊急"
                count_urgent += 1
            else:
                result_str = "⚪ それ以外"
                count_other += 1
            
            print(f"{file_name:<20} | {result_str:<12} | {prob_str}")

    print("-" * 60)
    print("【 📊 判定結果まとめ 】")
    print(f"  ✅ 通常   : {count_normal} 件")
    print(f"  ⚠️ 緊急   : {count_urgent} 件")
    print(f"  ⚪ それ以外: {count_other} 件")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    if os.path.exists("dataset"):
        my_classifier = train_my_model()

        if my_classifier:
            # テストしたいフォルダを指定
            test_folder = "testdata/sbv2_JSUTfs" 
            if os.path.exists(test_folder):
                predict_batch(my_classifier, test_folder)
            else:
                print(f"'{test_folder}' がありません。")
    else:
        print("'dataset'フォルダが見つかりません。")
