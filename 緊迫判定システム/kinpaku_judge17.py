import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 0. 音声ファイルを取得する補助関数 (重複修正版)
# ==========================================
def get_audio_files(folder_path):
    """指定されたフォルダからwavとmp3ファイルの両方を取得する"""
    files = []
    files.extend(glob.glob(os.path.join(folder_path, "*.wav")))
    files.extend(glob.glob(os.path.join(folder_path, "*.mp3")))
    files.extend(glob.glob(os.path.join(folder_path, "*.WAV")))
    files.extend(glob.glob(os.path.join(folder_path, "*.MP3")))
    
    # 重複を削除（Windows等での2重読み込みを防止）して、名前順に並べ直す
    unique_files = sorted(list(set(files)))
    return unique_files

# ==========================================
# 1. 特徴量を抽出する関数（13次元・Δ＋ΔΔ強化版）
# ==========================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
            
        # ① MFCCを標準的な13次元で計算
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # ② Δ（デルタ：変化の速度）
        delta_mfcc = librosa.feature.delta(mfcc)
        
        # ③ ΔΔ（デルタデルタ：変化の加速度・爆発力） ★ここが13次元で戦うカギ！
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # ④ 特徴量の選定（StdとMax）
        # 次元数が少ない分、ΔとΔΔの両方を使って「動き」を徹底的にマークします
        
        # --- Δ (Speed) ---
        d_std = np.std(delta_mfcc, axis=1) # 13次元
        d_max = np.max(delta_mfcc, axis=1) # 13次元
        
        # --- ΔΔ (Acceleration) ---
        d2_std = np.std(delta2_mfcc, axis=1) # 13次元
        d2_max = np.max(delta2_mfcc, axis=1) # 13次元
        
        # 合計 52次元 (13*4)
        return np.hstack([d_std, d_max, d2_std, d2_max])
        
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
    normal_files = get_audio_files("dataset/normal")
    print(f"   [0] Normal (通常): {len(normal_files)}")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None: 
            X.append(feat)
            y.append(0) 
            
    # クラス1: 緊急
    urgent_files = get_audio_files("dataset/urgent")
    print(f"   [1] Urgent (緊急): {len(urgent_files)}")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None: 
            X.append(feat)
            y.append(1) 

    # クラス2: それ以外
    other_files = get_audio_files("dataset/other") 
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
    # 特徴量を絞ったので、決定木の数は標準的な100に戻します
    classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
    classifier.fit(X, y_labels)
    print("✅ 学習完了！\n")
    return classifier

# ==========================================
# 3. 指定フォルダを一括判定して集計する関数（シンプル多数決版＋拮抗カウント追加）
# ==========================================
def predict_batch(classifier, target_folder):
    files = get_audio_files(target_folder)
    
    if len(files) == 0:
        print(f"⚠️ '{target_folder}' にwavまたはmp3ファイルがありません。")
        return

    print(f"--- 📂 一括判定開始: {target_folder} ---")
    print(f"{'ファイル名':<25} | {'判定結果':<12} | {'詳細(確率)'}")
    print("-" * 70)

    count_normal = 0
    count_urgent = 0
    count_other  = 0
    count_borderline = 0  # 追加: 拮抗状態のカウント用

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
            
            # ★ 追加: 拮抗判定（低緊迫40%以上 かつ それ以外40%以上）
            if prob_normal >= 0.4 and prob_other >= 0.4:
                count_borderline += 1
            
            # ★ シンプル判定ロジック: 一番確率が高いインデックスを取得
            # 0: 通常, 1: 緊急, 2: それ以外
            pred_label = np.argmax(probs)

            if pred_label == 0:
                result_str = "✅ 通常(低)"
                count_normal += 1
            elif pred_label == 1:
                result_str = "⚠️ 緊急(高)"
                count_urgent += 1
            else:
                result_str = "⚪ それ以外"
                count_other += 1
                
            # 詳細表示用の文字列
            detail_str = f"通:{prob_normal*100:.0f}% 緊:{prob_urgent*100:.0f}% 他:{prob_other*100:.0f}%"

            # 拮抗している場合は詳細にマークを付けるなど工夫も可能です
            if prob_normal >= 0.4 and prob_other >= 0.4:
                detail_str += " [★拮抗]"

            print(f"{file_name:<25} | {result_str:<12} | {detail_str}")

    print("-" * 70)
    print("【 📊 判定結果まとめ 】")
    print(f"  ✅ 通常(低緊迫): {count_normal} 件")
    print(f"  ⚠️ 緊急(高緊迫): {count_urgent} 件")
    print(f"  ⚪ それ以外    : {count_other} 件")
    print(f"  -----------")
    print(f"  合計           : {len(files)} 件")
    # 追加: 拮抗状態の結果を表示
    print(f"  (※うち、「通常」と「それ以外」が拮抗(両方40%以上): {count_borderline} 件)")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    if os.path.exists("dataset"):
        # 学習実行
        my_classifier = train_my_model()

        if my_classifier:
            # テスト実行
            test_folder = "testdata/sbv2_teikinpaku"
            if os.path.exists(test_folder):
                predict_batch(my_classifier, test_folder)
            else:
                print(f"'{test_folder}' がありません。")
