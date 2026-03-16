import numpy as np
import os
import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
# train_test_split, accuracy_score は今回表示用には使いませんが、精度検証用に残しておきます
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 特徴量を抽出する関数
# ==========================================
def extract_features(file_path):
    try:
        # 音声を読み込む (sr=16000は音声認識の標準的設定)
        y, sr = librosa.load(file_path, sr=16000)
        
        # MFCC（声色の特徴）を抽出 
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 時間方向の平均を取る
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        return mfcc_mean
    except Exception as e:
        print(f"エラー ({os.path.basename(file_path)}): {e}")
        return None

# ==========================================
# 2. データを読み込んで学習する関数
# ==========================================
def train_my_model():
    X = []
    y = []

    print("--- 📚 学習データを読み込み中 ---")
    
    # 正常データの読み込み
    normal_files = glob.glob("dataset/normal/*.wav")
    print(f"   Normal: {len(normal_files)} ファイル")
    for file in normal_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(0) 
            
    # 緊急データの読み込み
    urgent_files = glob.glob("dataset/urgent/*.wav")
    print(f"   Urgent: {len(urgent_files)} ファイル")
    for file in urgent_files:
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(1) 

    if len(X) == 0:
        print("❌ エラー: データが見つかりません。'dataset'フォルダを確認してください。")
        return None

    X = np.array(X)
    y = np.array(y)

    # モデル作成と学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✅ 学習完了！モデルの準備ができました。\n")
    
    return model

# ==========================================
# 3. 指定フォルダを一括判定する関数（厳しめの判定）
# ==========================================
def predict_batch(model, target_folder):
    search_path = os.path.join(target_folder, "*.wav")
    files = glob.glob(search_path)
    
    if len(files) == 0:
        print(f"⚠️ '{target_folder}' にwavファイルがありません。")
        return

    print(f"--- 📂 一括判定開始: {target_folder} ---")
    print(f"{'ファイル名':<20} | {'判定':<12} | {'緊急度スコア':<10}")
    print("-" * 50)

    # ★ ここで「厳しさ」を調整します
    # 0.8 (80%) 以上なら「緊急」とみなす
    URGENT_THRESHOLD = 0.8  
    # 0.2 (20%) 以下なら「通常」とみなす（＝緊急度が低い）
    NORMAL_THRESHOLD = 0.2
    
    # その間は「それ以外（どっちつかず）」とする

    for file_path in files:
        file_name = os.path.basename(file_path)
        
        # 特徴量抽出（既存の関数を使用）
        feat = extract_features(file_path)
        
        if feat is not None:
            feat = feat.reshape(1, -1)
            
            # 確率を取得 [通常確率, 緊急確率]
            probs = model.predict_proba(feat)[0]
            urgency_score = probs[1] # 緊急である確率 (0.0 〜 1.0)
            
            # --- 判定ロジック ---
            if urgency_score >= URGENT_THRESHOLD:
                # 緊急度がとても高い場合
                result_str = "⚠️ 緊急"
            elif urgency_score <= NORMAL_THRESHOLD:
                # 緊急度がとても低い（＝通常である確率が高い）場合
                result_str = "✅ 通常"
            else:
                # どっちつかずの場合（グレーゾーン）
                result_str = "⚪ それ以外"
            
            # 結果表示
            print(f"{file_name:<20} | {result_str:<12} | {urgency_score*100:5.1f}%")

    print("-" * 50)
    print(f"※ 判定基準: スコアが {URGENT_THRESHOLD*100:.0f}% 以上のみ『緊急』と判定")

    # ==========================================
# メイン処理
# ==========================================

# 1. データセットがあるか確認して学習
if os.path.exists("dataset"):
    my_model = train_my_model()

    if my_model:
        # 2. テスト用データを一括判定
        # "test_data" というフォルダを作って、そこに判定させたいファイルを入れてください
        test_folder = "testdata/sbv2_teikinpaku"
        
        if os.path.exists(test_folder):
            predict_batch(my_model, test_folder)
        else:
            print(f"【案内】'{test_folder}' フォルダがありません。作成して判定したい音声を入れてください。")
            # フォルダがない場合、とりあえずカレントディレクトリのwavだけ探す例
            # predict_batch(my_model, ".") 
else:
    print("【注意】'dataset'フォルダが見つかりません。学習できません。")
