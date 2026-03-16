import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
import parselmouth
from parselmouth.praat import call
import warnings

# Praat/librosa の細かい警告を非表示にする（ログが見づらくなるのを防ぐため）
warnings.filterwarnings('ignore')

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
    
    unique_files = sorted(list(set(files)))
    return unique_files

# ==========================================
# 1. 特徴量を抽出する関数（MFCCを減らし、音響特徴量を増やす）
# ==========================================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
            
        # ① MFCCを40次元から、標準的な13次元に減らす
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc[1:, :])
        delta2_mfcc = librosa.feature.delta(mfcc[1:, :], order=2)
        
        # MFCCの統計量を計算 (13次元 × 4 = 52次元に削減)
        d_std = np.std(delta_mfcc, axis=1)
        d_max = np.max(delta_mfcc, axis=1)
        d2_std = np.std(delta2_mfcc, axis=1)
        d2_max = np.max(delta2_mfcc, axis=1)
        
        # ② スペクトルエントロピー
        S, _ = librosa.magphase(librosa.stft(y))
        S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
        entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
        
        # エントロピーも平均だけでなく「最大値」と「ばらつき」を追加
        ent_mean = np.mean(entropy)
        ent_std = np.std(entropy)
        ent_max = np.max(entropy)

        # ③ Parselmouthを用いた音響特徴量
        sound = parselmouth.Sound(file_path)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        
        # ピッチの特徴量を大幅に増やす（緊迫感に直結するため）
        if len(pitch_values) > 0:
            p_mean = np.mean(pitch_values)
            p_std = np.std(pitch_values) # 声の高低のブレ
            p_max = np.max(pitch_values) # 悲鳴などの最高音
            p_min = np.min(pitch_values)
        else:
            p_mean, p_std, p_max, p_min = 0.0, 0.0, 0.0, 0.0
        
        # ジッター、シマー
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        try:
            jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            jitter = np.nan_to_num(jitter)
            shimmer = np.nan_to_num(shimmer)
        except:
            jitter, shimmer = 0.0, 0.0

        # ④ 音響特徴量をまとめる (9次元に増加)
        acoustic_features = np.array([
            ent_mean, ent_std, ent_max, 
            p_mean, p_std, p_max, p_min, 
            jitter, shimmer
        ])
        
        # 最終的な結合 (52次元 + 9次元 = 全61次元と非常にバランスが良くなる)
        final_features = np.hstack([d_std, d_max, d2_std, d2_max, acoustic_features])
        
        return final_features
        
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
    
    # max_features=None を追加することで、ランダム抽出をやめ、
    # 毎回すべての特徴量（61個）を比較して一番良い基準で分岐するようになる
    classifier = RandomForestClassifier(
        n_estimators=1000, 
        random_state=42,
        # max_features=None  # ← ここを追加！
    )
    classifier.fit(X, y_labels)
    
    # [追加] どの特徴量が重要だったかを簡易表示
    print("💡 特徴量の重要度トップ5:")
    importances = classifier.feature_importances_
    
    # 特徴量名のリストを作成 (全61次元に合わせて正確にマッピング)
    feature_names = []
    
    # ① MFCC関連 (13次元 × 4 = 52次元)
    feature_names.extend([f"MFCC_d_std_{i}" for i in range(1, 13)])
    feature_names.extend([f"MFCC_d_max_{i}" for i in range(1, 13)])
    feature_names.extend([f"MFCC_d2_std_{i}" for i in range(1, 13)])
    feature_names.extend([f"MFCC_d2_max_{i}" for i in range(1, 13)])
    
    # ② 音響特徴量関連 (9次元)
    feature_names.extend([
        "Entropy_mean (エントロピー平均)", 
        "Entropy_std (エントロピーばらつき)", 
        "Entropy_max (エントロピー最大値)", 
        "Pitch_mean (ピッチ平均)", 
        "Pitch_std (ピッチばらつき)", 
        "Pitch_max (ピッチ最大値/悲鳴等)", 
        "Pitch_min (ピッチ最小値)", 
        "Jitter (声の震え/周期の揺らぎ)", 
        "Shimmer (声の震え/振幅の揺らぎ)"
    ])
    
    # 重要度が高い順にソートしてトップ5を表示
    indices = np.argsort(importances)[::-1]
    for i in range(5):
        print(f"   {i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
        
    print("✅ 学習完了！\n")
    return classifier

# ==========================================
# 3. 指定フォルダを一括判定して集計する関数
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
            
            # シンプル判定ロジック: 一番確率が高いインデックスを取得
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
            test_folder = "testdata/koukinpaku"
            if os.path.exists(test_folder):
                predict_batch(my_classifier, test_folder)
            else:
                print(f"'{test_folder}' がありません。")
