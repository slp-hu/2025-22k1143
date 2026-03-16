import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_text

# ==========================================
# 1. 特徴量を抽出する関数（研究・論文ベースの強化版）
# ==========================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        feature_list = []
        
        # --- ① MFCC (基本 + Δ + ΔΔ) ---
        # 声質・何を言っているかを捉える
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 平均をとってリストに追加 (20次元×3)
        feature_list.extend(np.mean(mfcc, axis=1))
        feature_list.extend(np.mean(mfcc_delta, axis=1))
        feature_list.extend(np.mean(mfcc_delta2, axis=1))

        # --- ② Spectral Centroid (スペクトル重心) ---
        # 叫び声など「キンキンした声」で高くなる
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        feature_list.append(np.mean(cent))

        # --- ③ Spectral Contrast (スペクトルコントラスト) ---
        # 音のピークと谷の差。ノイズの多い環境や荒い声の識別に強い
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feature_list.extend(np.mean(contrast, axis=1))

        # --- ④ Zero Crossing Rate (ゼロ交差数) ---
        # 音の「荒さ」「ノイズ感」。息遣いや擦れ声で高くなる
        zcr = librosa.feature.zero_crossing_rate(y)
        feature_list.append(np.mean(zcr))
        
        # --- ⑤ RMS (エネルギー/音量) ---
        # 突発的な大声などを捉える
        rms = librosa.feature.rms(y=y)
        feature_list.append(np.mean(rms))

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
        # ★修正: if feat: ではなく if feat is not None: にする
        if feat is not None: 
            X.append(feat)
            y.append(0) 
            
    # クラス1: 緊急
    urgent_files = glob.glob("dataset/urgent/*.wav")
    print(f"   [1] Urgent (緊急): {len(urgent_files)}")
    for file in urgent_files:
        feat = extract_features(file)
        # ★修正
        if feat is not None: 
            X.append(feat)
            y.append(1) 

    # クラス2: それ以外
    other_files = glob.glob("dataset/other/*.wav") 
    print(f"   [2] Other   (他)  : {len(other_files)}")
    for file in other_files:
        feat = extract_features(file)
        # ★修正
        if feat is not None: 
            X.append(feat)
            y.append(2) 

    if len(X) == 0: return None

    X = np.array(X)
    y = np.array(y)

    print("🤖 AIモデルを作成中...")
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
            
            # ★ 連合軍ロジック: 「通常＋緊急」の合計確率
            prob_target_total = prob_normal + prob_urgent

            # 判定ロジック
            # 「それ以外」が「通常＋緊急の合計」より大きければ、文句なしで「それ以外」
            if prob_other > prob_target_total:
                result_str = "⚪ それ以外"
                confidence = prob_other * 100
                count_other += 1
                detail_str = f"他率: {confidence:.1f}%"

            else:
                # 「対象音声（通常＋緊急）」の合計が過半数
                # 中身で決戦
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
if os.path.exists("dataset"):
    # フォルダチェック
    if not os.path.exists("dataset/other"):
        print("⚠️ 注意: dataset/other フォルダを確認してください。")

    my_classifier = train_my_model()

    if my_classifier:
        test_folder = "testdata/teikinpaku"
        if os.path.exists(test_folder):
            predict_batch(my_classifier, test_folder)
        else:
            print(f"'{test_folder}' がありません。")
else:
    print("'dataset'フォルダが見つかりません。")


    
