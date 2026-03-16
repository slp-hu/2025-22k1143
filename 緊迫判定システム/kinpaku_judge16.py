import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text

# ==========================================
# 1. 特徴量を抽出する関数（ΔMFCCのみ・一点突破版）
# ==========================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # 無音区間カット（ノイズ除去のため継続）
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 1024:
            y = y_trimmed
            
        feature_list = []
        
        # ① 基本のMFCCを計算（これはΔの計算に必要）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # ② Δ（デルタ）MFCCを計算
        # ※ここだけを使います！
        delta_mfcc = librosa.feature.delta(mfcc[1:])
        
        # ③ 統計量の算出
        # 平均(Mean)だけでなく、最大(Max)と分散(Std)をとることで
        # 「急激な変化」や「一瞬の震え」を捉えます。
        
        
        
        # 最大値：最も激しく変化した瞬間の値（これが緊急度のカギになる可能性大）
        feature_list.extend(np.max(delta_mfcc, axis=1))
        
        # 標準偏差：変化の激しさ・ばらつき
        feature_list.extend(np.std(delta_mfcc, axis=1))
        
        return np.array(feature_list)
        
    except Exception as e:
        print(f"エラー ({os.path.basename(file_path)}): {e}")
        return None
    
# ==========================================
# 2. 学習する関数
# ==========================================
def train_my_model():
    X = []
    y = []

    print("--- 📚 学習データを読み込み中 ---")
    
    # クラス0: 通常
    normal_files = glob.glob("dataset/normal/*.wav")
    print(f"   [0] Normal (通常): {len(normal_files)}")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None: 
            X.append(feat)
            y.append(0) 
            
    # クラス1: 緊急
    urgent_files = glob.glob("dataset/urgent/*.wav")
    print(f"   [1] Urgent (緊急): {len(urgent_files)}")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None: 
            X.append(feat)
            y.append(1) 

    # クラス2: それ以外
    other_files = glob.glob("dataset/other/*.wav") 
    print(f"   [2] Other   (他)  : {len(other_files)}")
    for file in other_files:
        feat = extract_features(file)
        if feat is not None: 
            X.append(feat)
            y.append(2) 

    if len(X) == 0: return None

    X = np.array(X)
    y_labels = np.array(y)

    print("🤖 AIモデルを作成中...")
    # 木の数を増やして、複雑な特徴量に対応しやすくする
    classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    classifier.fit(X, y_labels)
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
    print(f"{'ファイル名':<20} | {'判定結果':<12} | {'詳細(確率)'}")
    print("-" * 65)

    count_normal = 0
    count_urgent = 0
    count_other  = 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            
            # 確率を取得 [通常%, 緊急%, それ以外%]
            probs = classifier.predict_proba(feat)[0]
            prob_normal = probs[0]
            prob_urgent = probs[1]
            prob_other  = probs[2]
            
            # 連合軍ロジック: 「通常＋緊急」の合計確率
            prob_target_total = prob_normal + prob_urgent

            # 判定ロジック
            if prob_other > prob_target_total:
                result_str = "⚪ それ以外"
                confidence = prob_other * 100
                count_other += 1
                detail_str = f"他率: {confidence:.1f}%"
            else:
                if prob_urgent > prob_normal:
                    result_str = "⚠️ 緊急(高)"
                    confidence = prob_urgent * 100
                    count_urgent += 1
                else:
                    result_str = "✅ 通常(低)"
                    confidence = prob_normal * 100
                    count_normal += 1
                
                detail_str = f"対象計: {prob_target_total*100:.1f}% (内: {result_str[2:4]})"

            print(f"{file_name:<20} | {result_str:<12} | {detail_str}")

    print("-" * 65)
    print("【 📊 判定結果まとめ 】")
    print(f"  ✅ 通常(低緊迫): {count_normal} 件")
    print(f"  ⚠️ 緊急(高緊迫): {count_urgent} 件")
    print(f"  ⚪ それ以外    : {count_other} 件")
    print(f"  -----------")
    print(f"  合計           : {len(files)} 件")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    if os.path.exists("dataset"):
        if not os.path.exists("dataset/other"):
            print("⚠️ 注意: dataset/other フォルダを確認してください。")

        my_classifier = train_my_model()

        if my_classifier:
            test_folder = "testdata/teikinpaku"
            if os.path.exists(test_folder):
                predict_batch(my_classifier, test_folder)
            else:
                print(f"'{test_folder}' がありません。")
                
