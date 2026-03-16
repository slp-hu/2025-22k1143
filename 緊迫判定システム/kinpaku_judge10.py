import librosa
import numpy as np
import os
import glob
import random
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 特徴量を抽出する関数（シンプルにMFCCのみ）
# ==========================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # MFCC（声の特徴）だけを取得
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 平均値をとって1次元データにする
        # (もし精度が足りなければ、ここだけ np.std(mfcc.T, axis=0) も追加してください)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        return mfcc_mean
        
    except Exception as e:
        print(f"エラー: {e}")
        return None

# ==========================================
# 2. 2段階モデルの学習
# ==========================================
def train_hierarchical_models():
    print("--- 📚 データを読み込み中 (MFCCのみ) ---")
    
    # ファイルリスト
    normal_files = glob.glob("dataset/normal/*.wav")
    urgent_files = glob.glob("dataset/urgent/*.wav")
    other_files = glob.glob("dataset/other/*.wav")

    # バランス調整（Otherが多すぎる場合、Normal/Urgentの多い方の2倍に制限）
    target_count = max(len(normal_files), len(urgent_files)) * 2
    if len(other_files) > target_count:
        other_files = random.sample(other_files, target_count)

    # 特徴量抽出
    X_normal = [extract_features(f) for f in normal_files if extract_features(f) is not None]
    X_urgent = [extract_features(f) for f in urgent_files if extract_features(f) is not None]
    X_other  = [extract_features(f) for f in other_files if extract_features(f) is not None]

    if not X_normal or not X_urgent:
        print("❌ エラー: データが足りません。")
        return None, None

    # ----------------------------------------
    # 【Step 1】 アナウンス判定 (アナウンス vs その他)
    # ----------------------------------------
    print("🤖 Step 1: アナウンス判定モデル学習...")
    X_s1 = X_normal + X_urgent + X_other
    y_s1 = [1]*len(X_normal) + [1]*len(X_urgent) + [0]*len(X_other) # 1:アナウンス, 0:その他

    model_s1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_s1.fit(X_s1, y_s1)

    # ----------------------------------------
    # 【Step 2】 緊迫度判定 (低緊迫 vs 高緊迫)
    # ----------------------------------------
    print("🤖 Step 2: 緊迫度判定モデル学習...")
    X_s2 = X_normal + X_urgent
    y_s2 = [0]*len(X_normal) + [1]*len(X_urgent) # 0:低, 1:高

    model_s2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_s2.fit(X_s2, y_s2)

    print("✅ 学習完了！")
    return model_s1, model_s2

# ==========================================
# 3. 判定実行
# ==========================================
def predict_hierarchical(model1, model2, target_folder):
    files = glob.glob(os.path.join(target_folder, "*.wav"))
    if not files: return

    print(f"\n--- 📂 判定開始: {target_folder} ---")
    print(f"{'ファイル名':<20} | {'Step1':<10} | {'Step2':<10} | {'結果'}")
    print("-" * 65)

    c_norm, c_urg, c_oth = 0, 0, 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            
            # --- Step 1 ---
            is_announce_prob = model1.predict_proba(feat)[0][1]
            
            if is_announce_prob < 0.5: # 閾値50%
                res1 = "× その他"
                res2 = "-"
                final = "⚪ それ以外"
                c_oth += 1
            else:
                res1 = "○ 対象"
                
                # --- Step 2 ---
                urgency_prob = model2.predict_proba(feat)[0][1]
                
                if urgency_prob >= 0.5:
                    res2 = "高"
                    final = "⚠️ 高緊迫"
                    c_urg += 1
                else:
                    res2 = "低"
                    final = "✅ 低緊迫"
                    c_norm += 1
            
            print(f"{file_name:<20} | {res1:<10} | {res2:<10} | {final}")

    print("-" * 65)
    print(f"  ✅ 低緊迫 : {c_norm}")
    print(f"  ⚠️ 高緊迫 : {c_urg}")
    print(f"  ⚪ それ以外: {c_oth}")

# ==========================================
# メイン
# ==========================================
if __name__ == "__main__":
    if os.path.exists("dataset"):
        m1, m2 = train_hierarchical_models()
        if m1 and m2:
            test_folder = "testdata/teikinpaku" 
            if os.path.exists(test_folder):
                predict_hierarchical(m1, m2, test_folder)
