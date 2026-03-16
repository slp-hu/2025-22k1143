import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# ==========================================
# 1. 特徴量を抽出する関数
# ==========================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"エラー ({os.path.basename(file_path)}): {e}")
        return None

# ==========================================
# 2. 学習する関数（門番AIと判別AI）
# ==========================================
def train_my_model():
    X = []
    y = []

    print("--- 📚 学習データを読み込み中 ---")
    
    # 正常データ
    normal_files = glob.glob("dataset/normal/*.wav")
    print(f"   Normal: {len(normal_files)} ファイル")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(0) 
            
    # 緊急データ
    urgent_files = glob.glob("dataset/urgent/*.wav")
    print(f"   Urgent: {len(urgent_files)} ファイル")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(1) 

    if len(X) == 0:
        print("❌ エラー: データが見つかりません。")
        return None, None

    X = np.array(X)
    y = np.array(y)

    print("🤖 AIモデルを作成中...")
    
    # 1. 判別モデル（通常 vs 緊急）
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)

    # 2. 門番モデル（外れ値検知）
    # contamination=0.05 は「学習データの5%くらいはノイズかも」という設定。
    # これを小さく(0.01など)すると、より厳しく「それ以外」と判定するようになります。
    outlier_detector = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    outlier_detector.fit(X)

    print("✅ 学習完了！\n")
    return classifier, outlier_detector

# ==========================================
# 3. 指定フォルダを一括判定して集計する関数
# ==========================================
def predict_batch(classifier, outlier_detector, target_folder):
    files = glob.glob(os.path.join(target_folder, "*.wav"))
    
    if len(files) == 0:
        print(f"⚠️ '{target_folder}' にwavファイルがありません。")
        return

    print(f"--- 📂 一括判定開始: {target_folder} ---")
    print(f"{'ファイル名':<20} | {'判定':<12} | {'詳細':<15}")
    print("-" * 60)

    # ★ 集計用カウンター
    count_urgent = 0
    count_normal = 0
    count_other = 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            
            # --- 門番チェック（それ以外か？） ---
            is_known_data = outlier_detector.predict(feat)[0]
            
            if is_known_data == -1:
                # 学習データと特徴が違いすぎる場合
                print(f"{file_name:<20} | ⚪ それ以外 | 類似度低（未知）")
                count_other += 1  # カウント
                continue 

            # --- 判別チェック（通常 vs 緊急） ---
            probs = classifier.predict_proba(feat)[0]
            urgency_score = probs[1]
            
            if urgency_score > 0.5:
                result_str = "⚠️ 緊急"
                count_urgent += 1 # カウント
            else:
                result_str = "✅ 通常"
                count_normal += 1 # カウント
            
            print(f"{file_name:<20} | {result_str:<12} | 緊急度: {urgency_score*100:4.1f}%")

    print("-" * 60)
    # ★ 最後に集計結果を表示
    print("【 📊 判定結果まとめ 】")
    print(f"  ⚠️ 緊急   : {count_urgent} 件")
    print(f"  ✅ 通常   : {count_normal} 件")
    print(f"  ⚪ それ以外: {count_other} 件")
    print(f"  -----------")
    print(f"  合計      : {len(files)} 件")

# ==========================================
# メイン処理
# ==========================================
if os.path.exists("dataset"):
    my_classifier, my_detector = train_my_model()

    if my_classifier and my_detector:
        test_folder = "testdata/sbv2_teikinpaku"
        if os.path.exists(test_folder):
            predict_batch(my_classifier, my_detector, test_folder)
        else:
            print(f"'{test_folder}' がありません。")
else:
    print("'dataset'フォルダが見つかりません。")
