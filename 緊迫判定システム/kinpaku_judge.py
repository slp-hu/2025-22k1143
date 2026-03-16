import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. 音声データから特徴量(MFCC)を取り出す関数
# ==========================================
def extract_features(file_path, max_len=100):
    """
    音声ファイルを読み込み、MFCC特徴量を抽出して
    AIモデルに入力できる形(固定長)に整えます。
    """
    try:
        # 音声のロード (sr=Noneで元のサンプリングレートを維持)
        audio, sr = librosa.load(file_path, sr=None)
        
        # MFCC特徴量の抽出 (論文で言及されている主要な特徴量)
        # n_mfcc=40 は一般的な設定値です
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # 形を (特徴量数, 時間) から (時間, 特徴量数) に変換
        mfccs = mfccs.T
        
        # 時間の長さを揃える処理 (パディングまたは切り取り)
        # AIに入力するには、全てのデータのサイズが同じである必要があります
        if mfccs.shape[0] < max_len:
            # 短い場合は0で埋める
            pad_width = max_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # 長い場合は切り取る
            mfccs = mfccs[:max_len, :]
            
        return mfccs
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

# ==========================================
# 2. CNN-LSTMモデルの定義 (前の会話の改良版)
# ==========================================
def create_model(input_shape):
    model = models.Sequential()
    
    # CNN層: 特徴の抽出
    # 入力形状: (時間ステップ, 特徴量数)
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # LSTM層: 時間的変化の学習
    # 論文ではCNNとLSTMのハイブリッド構成が推奨されています
    model.add(layers.LSTM(128))
    
    # 出力層: 緊急かそうでないかの2択 (0: Low Stress, 1: High Stress)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 3. 実行部分 (メイン処理)
# ==========================================

# --- 設定 ---
# ※ここにあなたの音声ファイルのパスを入れてください (例: 'my_voice.wav')
# テスト用にダミーファイルを作成する機能はありませんが、
# エラーが出ないよう、ファイルパスを指定して実行してください。
file_path = "kinpaku_k1.wav" 

# --- A. 特徴量の抽出 ---
# 実際のファイルがない場合のために、ここではランダムなダミーデータで動作確認します
# 本番では `features = extract_features(file_path)` を使ってください
print("音声を処理中...")
# features = extract_features(file_path) # ← 音声ファイルがある場合はこれを使う
features = np.random.rand(100, 40) # ← テスト用のダミーデータ

if features is not None:
    # データの形状確認
    print(f"抽出された特徴量の形: {features.shape}") # (100, 40) になるはずです

    # モデルに合わせて次元を追加 (1件のデータ, 時間, 特徴量)
    # Kerasは一度に複数のデータを処理する前提のため、先頭に次元が必要です
    X_input = np.expand_dims(features, axis=0) 

    # --- B. モデルの作成 ---
    model = create_model(input_shape=(100, 40))
    
    # --- C. 予測 (緊急度の判定) ---
    # まだ学習していないので結果はランダムですが、動くことを確認します
    prediction = model.predict(X_input)
    
    # 結果の表示
    stress_probability = prediction[0][1] # 1番目(High Stress)の確率
    print(f"--------------------------------")
    print(f"緊急度 (Stress Probability): {stress_probability * 100:.2f}%")
    
    if stress_probability > 0.5:
        print("判定: 緊急 (High Stress)")
    else:
        print("判定: 通常 (Low Stress)")
    print(f"--------------------------------")
