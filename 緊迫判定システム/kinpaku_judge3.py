import librosa
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ==========================================
# 1. 特徴量を抽出する関数 (論文のMFCC手法を採用)
# ==========================================
def extract_features(file_path):
    try:
        # 音声を読み込む
        y, sr = librosa.load(file_path, sr=16000)
        
        # MFCC（声色の特徴）を抽出 
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # 時間方向の平均を取って、1つの音声につき1セットの数値にする
        # (CNN-LSTMなら時間の流れを使いますが、今回は簡易化のため平均を使います)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        return mfcc_mean
    except Exception as e:
        print(f"エラー ({file_path}): {e}")
        return None

# ==========================================
# 2. データを読み込んで学習する関数
# ==========================================
def train_my_model():
    X = [] # 特徴量（入力データ）
    y = [] # 正解ラベル（0:通常, 1:緊急）

    print("--- 学習データを読み込み中 ---")
    
    # "normal" フォルダの読み込み (ラベル: 0)
    for file in glob.glob("dataset/normal/*.wav"):
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(0) # 0 = Normal
            
    # "urgent" フォルダの読み込み (ラベル: 1)
    for file in glob.glob("dataset/urgent/*.wav"):
        feat = extract_features(file)
        if feat is not None:
            X.append(feat)
            y.append(1) # 1 = Urgent

    if len(X) == 0:
        print("エラー: データが見つかりません。'dataset'フォルダを確認してください。")
        return None

    # 配列に変換
    X = np.array(X)
    y = np.array(y)

    print(f"データ数: {len(X)}件 (通常: {np.sum(y==0)}, 緊急: {np.sum(y==1)})")

    # モデルの作成（ランダムフォレスト）
    # 論文でも比較対象として使用されているアルゴリズムです [cite: 148]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 学習実行！
    model.fit(X, y)
    print("✅ 学習完了！")
    
    return model

# ==========================================
# 3. 未知のデータを判定する関数
# ==========================================
def predict_urgency(model, file_path):
    print(f"\n--- 判定中: {file_path} ---")
    feat = extract_features(file_path)
    
    if feat is not None:
        # モデルに入力するために形を整える
        feat = feat.reshape(1, -1)
        
        # 確率を予測 (左がNormal確率, 右がUrgent確率)
        probs = model.predict_proba(feat)[0]
        urgency_score = probs[1] # 緊急である確率
        
        print(f"🚨 緊急度スコア: {urgency_score * 100:.1f}%")
        
        if urgency_score > 0.5:
            print("判定结果: ⚠️ 緊急 (Urgent)")
        else:
            print("判定结果: ✅ 通常 (Normal)")

# ==========================================
# メイン処理
# ==========================================

# 1. 自分で学習させる
# (datasetフォルダに音声を入れてから実行してください)
# データがないとエラーになるので、ダミーで動作確認したい場合は
# 以下の `if False:` を `if True:` に書き換える必要がありますが、
# 基本的には dataset フォルダを作って実行することを推奨します。

if os.path.exists("dataset"):
    my_model = train_my_model()

    # 2. テスト用ファイルを判定してみる
    # ここに判定したい新しいファイル名を指定
    if my_model:
        predict_urgency(my_model, "075.wav") 
else:
    print("【注意】'dataset'フォルダが見つかりません。")
    print("1. 現在の場所に 'dataset' フォルダを作成してください。")
    print("2. その中に 'urgent' と 'normal' フォルダを作り、wavファイルを入れてください。")
