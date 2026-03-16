import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# ==========================================
# 1. 特徴量を抽出する関数 (変更なし)
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
# 2. 学習する関数（門番AIを追加）
# ==========================================
def train_my_model():
    X = []
    y = []

    print("--- 📚 学習データを読み込み中 ---")
    
    # 1. 正常データ (ラベル 0)
    normal_files = glob.glob("dataset/normal/*.wav")
    print(f"   Normal (通常): {len(normal_files)} ファイル")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(0) 
            
    # 2. 緊急データ (ラベル 1)
    urgent_files = glob.glob("dataset/urgent/*.wav")
    print(f"   Urgent (緊急): {len(urgent_files)} ファイル")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(1) 

    # 3. 除外データ (ラベル 2) ← ★ここを追加
    other_files = glob.glob("dataset/other/*.wav")
    print(f"   Other  (除外): {len(other_files)} ファイル")
    for file in other_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(2)  # ラベルを「2」にする

    if len(X) == 0:
        print("❌ エラー: データが見つかりません。")
        return None, None

    X = np.array(X)
    y = np.array(y)

    print("🤖 AIモデルを作成中...")
    
    # 3クラス分類として学習されます
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)

    # 門番も一応作っておく（全く未知の音対策）
    outlier_detector = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    outlier_detector.fit(X)

    print("✅ 学習完了！\n")
    return classifier, outlier_detector

# ==========================================
# 3. 指定フォルダを一括判定する関数
# ==========================================
def predict_batch(classifier, outlier_detector, target_folder):
    files = glob.glob(os.path.join(target_folder, "*.wav"))
    # (省略: ファイルがない場合のエラー処理などはそのまま)

    print(f"--- 📂 一括判定開始: {target_folder} ---")
    print(f"{'ファイル名':<20} | {'判定':<12} | {'詳細':<15}")
    print("-" * 60)

    for file_path in files:
        file_name = os.path.basename(file_path)
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            
            # --- 1. 門番チェック（未知のデータか？） ---
            is_known_data = outlier_detector.predict(feat)[0]
            if is_known_data == -1:
                print(f"{file_name:<20} | ⚪ 未知の音 | 学習データと乖離")
                continue 

            # --- 2. 3クラス判別 ---
            prediction = classifier.predict(feat)[0] # 0, 1, 2 のどれかが出る
            probs = classifier.predict_proba(feat)[0] # 確率
            
            if prediction == 0:
                result_str = "✅ 通常"
                score = probs[0]
            elif prediction == 1:
                result_str = "⚠️ 緊急"
                score = probs[1]
            else: # prediction == 2
                result_str = "⚪ それ以外" # 学習済みの「除外データ」
                score = probs[2]
            
            print(f"{file_name:<20} | {result_str:<12} | 確信度: {score*100:4.1f}%")

    # (省略: 集計表示部分も必要に応じて count_other を増やしてください)

# ==========================================
# メイン処理
# ==========================================
if os.path.exists("dataset"):
    # 2つのモデルを受け取る
    my_classifier, my_detector = train_my_model()

    if my_classifier and my_detector:
        test_folder = "testdata/sbv2_koukinpaku"
        if os.path.exists(test_folder):
            predict_batch(my_classifier, my_detector, test_folder)
        else:
            print(f"'{test_folder}' がありません。")
else:
    print("'dataset'フォルダが見つかりません。")

