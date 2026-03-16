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
# 3. 閾値付きの判定関数
# ==========================================
def predict_batch(classifier, target_folder, threshold=0.5):
    """
    threshold=0.6 : 確率が60%を超えないと「判定」と認めない設定
    """
    files = glob.glob(os.path.join(target_folder, "*.wav"))
    if len(files) == 0: return

    print(f"--- 📂 一括判定開始（閾値 {threshold*100:.0f}%）: {target_folder} ---")
    print(f"{'ファイル名':<20} | {'判定':<12} | {'詳細 (低/高/他)'}")
    print("-" * 65)

    c_norm, c_urg, c_oth, c_unknown = 0, 0, 0, 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            probs = classifier.predict_proba(feat)[0]
            
            # 一番高い確率とそのインデックスを取得
            max_prob = np.max(probs)
            pred_label = np.argmax(probs)

            # 詳細表示用
            prob_str = f"{probs[0]:.2f} / {probs[1]:.2f} / {probs[2]:.2f}"

            # ★ ここで「閾値チェック」を行う
            if max_prob < threshold:
                # どの確率も閾値を超えていない場合
                res = "❓ 判定不能"
                c_unknown += 1
                # 理由も表示（迷っているため）
                prob_str += " (自信なし)"
            else:
                # 閾値を超えている場合のみ判定を採用
                if pred_label == 0:
                    res = "✅ 低緊迫"
                    c_norm += 1
                elif pred_label == 1:
                    res = "⚠️ 高緊迫"
                    c_urg += 1
                else:
                    res = "⚪ それ以外"
                    c_oth += 1
            
            print(f"{file_name:<20} | {res:<12} | {prob_str}")

    print("-" * 65)
    print(f"  ✅ 低緊迫 : {c_norm}")
    print(f"  ⚠️ 高緊迫 : {c_urg}")
    print(f"  ⚪ 他     : {c_oth}")
    print(f"  ❓ 不能   : {c_unknown}")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    if os.path.exists("dataset"):
        my_classifier = train_my_model()

        if my_classifier:
            # テストしたいフォルダを指定
            test_folder = "testdata/koukinpaku" 
            if os.path.exists(test_folder):
                predict_batch(my_classifier, test_folder)
            else:
                print(f"'{test_folder}' がありません。")
    else:
        print("'dataset'フォルダが見つかりません。")

