import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 特徴量を抽出する関数
# ==========================================
def extract_features(file_path):
    try:
        # サンプリングレートは16kで統一
        y, sr = librosa.load(file_path, sr=16000)
        
        # MFCC特徴量抽出
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"エラー ({os.path.basename(file_path)}): {e}")
        return None

# ==========================================
# 2. 学習する関数（3クラス分類：通常/緊急/それ以外）
# ==========================================
def train_my_model():
    X = []
    y = []

    print("--- 📚 学習データを読み込み中 ---")
    
    # -------------------------------------------------
    # クラス0: 通常（Normal / 低緊迫）
    # -------------------------------------------------
    normal_files = glob.glob("dataset/normal/*.wav")
    print(f"   [0] Normal (通常): {len(normal_files)} ファイル")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(0) 
            
    # -------------------------------------------------
    # クラス1: 緊急（Urgent / 高緊迫）
    # -------------------------------------------------
    urgent_files = glob.glob("dataset/urgent/*.wav")
    print(f"   [1] Urgent (緊急): {len(urgent_files)} ファイル")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(1) 

    # -------------------------------------------------
    # クラス2: それ以外（JVNV / Other）
    # -------------------------------------------------
    jvnv_files = glob.glob("dataset/other/*.wav")
    print(f"   [2] JVNV   (他)  : {len(jvnv_files)} ファイル")
    for file in jvnv_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(2) 

    # --- データチェック ---
    if len(X) == 0:
        print("❌ エラー: 学習データが全く見つかりません。")
        return None
    
    # 各クラスのデータが最低限あるか確認（警告のみ）
    if len(normal_files) == 0 or len(urgent_files) == 0 or len(jvnv_files) == 0:
        print("⚠️ 注意: 一部のクラスのデータが0件です。正しく分類できない可能性があります。")

    X = np.array(X)
    y = np.array(y)

    print("🤖 ランダムフォレスト(3クラス分類)を作成中...")
    
    # ランダムフォレストのみで学習
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)

    print("✅ 学習完了！\n")
    return classifier

# ==========================================
# 3. 指定フォルダを一括判定して集計する関数
# ==========================================
def predict_batch(classifier, target_folder):
    files = glob.glob(os.path.join(target_folder, "*.wav"))
    
    if len(files) == 0:
        print(f"⚠️ '{target_folder}' にwavファイルがありません。")
        return

    print(f"--- 📂 一括判定開始: {target_folder} ---")
    # ヘッダー表示
    print(f"{'ファイル名':<20} | {'判定結果':<12} | {'確信度(%)':<10}")
    print("-" * 60)

    # ★ 集計用カウンター
    count_normal = 0 # クラス0
    count_urgent = 0 # クラス1
    count_other  = 0 # クラス2

    # クラスIDと表示名の対応
    LABEL_MAP = {
        0: "✅ 通常(低)",
        1: "⚠️ 緊急(高)",
        2: "⚪ それ以外"
    }

    for file_path in files:
        file_name = os.path.basename(file_path)
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            
            # --- 予測実行 ---
            pred_label = classifier.predict(feat)[0]       # 0, 1, 2 のいずれか
            probs = classifier.predict_proba(feat)[0]      # 確率分布 [prob0, prob1, prob2]
            confidence = probs[pred_label] * 100           # 選ばれたクラスの確率

            # 結果の表示用文字列
            result_str = LABEL_MAP.get(pred_label, "不明")

            # カウントアップ
            if pred_label == 0:
                count_normal += 1
            elif pred_label == 1:
                count_urgent += 1
            else:
                count_other += 1
            
            print(f"{file_name:<20} | {result_str:<12} | {confidence:5.1f}%")

    print("-" * 60)
    # ★ 最後に集計結果を表示
    print("【 📊 判定結果まとめ 】")
    print(f"  ✅ 通常(低緊迫): {count_normal} 件")
    print(f"  ⚠️ 緊急(高緊迫): {count_urgent} 件")
    print(f"  ⚪ それ以外(他): {count_other} 件")
    print(f"  -----------")
    print(f"  合計           : {len(files)} 件")

# ==========================================
# メイン処理
# ==========================================
if os.path.exists("dataset"):
    # フォルダチェック（JVNVがあるか確認）
    if not os.path.exists("dataset/jvnv"):
        print("⚠️ 'dataset/jvnv' フォルダが見つかりません。")
        print("   JVNVコーパス（それ以外の音声）を格納してください。")
    
    # モデル学習 (IsolationForestは不要になったため classifier のみ受け取る)
    my_classifier = train_my_model()

    if my_classifier:
        test_folder = "testdata/teikinpaku"
        if os.path.exists(test_folder):
            predict_batch(my_classifier, test_folder)
        else:
            print(f"'{test_folder}' がありません。テスト用音声を入れてください。")
else:
    print("'dataset'フォルダが見つかりません。")
