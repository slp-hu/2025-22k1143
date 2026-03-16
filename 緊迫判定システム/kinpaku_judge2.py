import librosa
import numpy as np
import warnings
from transformers import pipeline

# 警告メッセージを整理
warnings.filterwarnings("ignore")

# ==========================================
# 1. 学習済みモデルの読み込み (鉄板モデルに変更)
# ==========================================
# "superb/wav2vec2-base-superb-er" は音声処理の公式ベンチマークで使われる
# 非常に信頼性が高く、誰でも使える公開モデルです。
MODEL_NAME = "superb/wav2vec2-base-superb-er"

print("モデルを準備中... (初回はダウンロードに数分かかります)")
try:
    classifier = pipeline("audio-classification", model=MODEL_NAME)
except Exception as e:
    print(f"モデルの読み込みに失敗しました: {e}")
    exit()

# ==========================================
# 2. 判定と「緊急度」への翻訳
# ==========================================
def analyze_urgency(file_path):
    print(f"\n--- 音声ファイルを読み込み中: {file_path} ---")
    
    try:
        # 1. 音声を読み込む (librosaを使用することでFFmpegエラーを回避)
        # このモデルは 16kHz での入力を前提としています
        speech, sr = librosa.load(file_path, sr=16000)
        
        # 2. AIに判定させる
        results = classifier(speech, top_k=None)
        
        # --- 結果の整理 ---
        # このモデルのラベル: ang(怒り), hap(喜び), neu(中立), sad(悲しみ)
        scores = {result['label']: result['score'] for result in results}
        
        # 緊急度の計算: ここでは「怒り(ang)」を緊急ストレスとみなします
        urgency_score = scores.get('ang', 0)
        
        # 表示
        print(f"😡 怒り (Anger)  : {scores.get('ang', 0)*100:.1f}%")
        print(f"😐 平常 (Neutral): {scores.get('neu', 0)*100:.1f}%")
        print(f"😢 悲しみ (Sad)  : {scores.get('sad', 0)*100:.1f}%")
        print(f"😄 喜び (Happy)  : {scores.get('hap', 0)*100:.1f}%")
        
        print(f"\n🚨 緊急度スコア: {urgency_score*100:.1f}%")
        
        # 判定基準 (閾値は調整してください)
        if urgency_score > 0.4: 
            print("判定: ⚠️ 高い緊急性あり (HIGH STRESS / URGENT)")
        else:
            print("判定: ✅ 通常 (LOW STRESS)")
            
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

    print("-----------------------------")

# ==========================================
# 3. 実行
# ==========================================
# ここにテストしたい音声ファイル名を指定してください
# ファイルがない場合でも、実行するとモデルのダウンロードまでは確認できます
analyze_urgency("jvnv_M1_tei_kou_kinpaku_demo4_k15.wav")
