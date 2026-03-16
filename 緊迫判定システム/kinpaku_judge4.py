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
# 3. [新機能] 指定フォルダを一括判定する関数
# ==========================================
def predict_batch(model, target_folder):
    # フォルダ内のwavファイルをすべて取得
    search_path = os.path.join(target_folder, "*.wav")
    files = glob.glob(search_path)
    
    if len(files) == 0:
        print(f"⚠️ '{target_folder}' フォルダの中にwavファイルが見つかりません。")
        return

    print(f"--- 📂 一括判定開始: {target_folder} ({len(files)}件) ---")
    print(f"{'ファイル名':<20} | {'判定':<10} | {'緊急度スコア':<10}")
    print("-" * 50)

    count_urgent = 0

    for file_path in files:
        file_name = os.path.basename(file_path) # パスからファイル名だけ取り出す
        feat = extract_features(file_path)
        
        if feat is not None:
            # モデルに入力する形に変換
            feat = feat.reshape(1, -1)
            
            # 予測
            probs = model.predict_proba(feat)[0]
            urgency_score = probs[1]
            
            # 判定ロジック
            if urgency_score > 0.5:
                result_str = "⚠️ 緊急"
                count_urgent += 1
            else:
                result_str = "✅ 通常"
            
            # 結果を表示（見やすく整形）
            print(f"{file_name:<20} | {result_str:<10} | {urgency_score*100:5.1f}%")

    print("-" * 50)
    print(f"判定終了: 全{len(files)}件中、{count_urgent}件が『緊急』と判定されました。")

# ==========================================
# メイン処理
# ==========================================

# 1. データセットがあるか確認して学習
if os.path.exists("dataset"):
    my_model = train_my_model()

    if my_model:
        # 2. テスト用データを一括判定
        # "test_data" というフォルダを作って、そこに判定させたいファイルを入れてください
        test_folder = "testdata/sbv2_koukinpaku"
        
        if os.path.exists(test_folder):
            predict_batch(my_model, test_folder)
        else:
            print(f"【案内】'{test_folder}' フォルダがありません。作成して判定したい音声を入れてください。")
            # フォルダがない場合、とりあえずカレントディレクトリのwavだけ探す例
            # predict_batch(my_model, ".") 
else:
    print("【注意】'dataset'フォルダが見つかりません。学習できません。")
