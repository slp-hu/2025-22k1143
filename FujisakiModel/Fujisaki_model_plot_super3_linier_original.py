import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks
import librosa
import matplotlib.ticker as ticker

# ==========================================
# 1. 藤崎モデル・解析ロジック (Super Lost準拠)
# ==========================================

class FujisakiModel:
    def __init__(self, alpha=3.0, beta=20.0, gamma=0.9):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def gp(self, t):
        val = np.zeros_like(t)
        mask = t >= 0
        if np.any(mask):
            val[mask] = (self.alpha ** 2) * t[mask] * np.exp(-self.alpha * t[mask])
        return val

    def ga(self, t):
        val = np.zeros_like(t)
        mask = t >= 0
        if np.any(mask):
            term = 1 - (1 + self.beta * t[mask]) * np.exp(-self.beta * t[mask])
            val[mask] = np.minimum(term, self.gamma)
        return val

    def generate_contour(self, times, fb, phrase_commands, accent_commands):
        log_fb = np.log(fb)
        y = np.full_like(times, log_fb)
        for t0, ap in phrase_commands:
            y += ap * self.gp(times - t0)
        for t1, t2, aa in accent_commands:
            y += aa * (self.ga(times - t1) - self.ga(times - t2))
        return np.exp(y)
    
    def generate_phrase_component(self, times, fb, phrase_commands):
        log_fb = np.log(fb)
        y = np.full_like(times, log_fb)
        for t0, ap in phrase_commands:
            y += ap * self.gp(times - t0)
        return np.exp(y)

    def generate_accent_component(self, times, accent_commands):
        y = np.zeros_like(times)
        for t1, t2, aa in accent_commands:
            y += aa * (self.ga(times - t1) - self.ga(times - t2))
        return y 

def find_contiguous_regions(bool_array):
    """Trueが連続する区間のインデックス(start, end)のリストを返す"""
    regions = []
    if not np.any(bool_array): return regions, 0
    changes = np.diff(np.concatenate(([0], bool_array.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    for s, e in zip(starts, ends):
        regions.append((s, e))
    return regions, len(regions)

def extract_initial_parameters_paper_logic(times, log_f0):
    """
    【計算部: Super Lost準拠】
    論文ロジックに基づき、アクセントを先に推定し、残差からフレーズを推定する
    """
    valid_mask = ~np.isnan(log_f0)
    if np.sum(valid_mask) == 0: return 100.0, [], []
    
    interp_log_f0 = np.interp(times, times[valid_mask], log_f0[valid_mask])
    smoothed_log_f0 = savgol_filter(interp_log_f0, window_length=21, polyorder=3)
    
    dt = times[1] - times[0]
    d1 = np.gradient(smoothed_log_f0, dt)
    d2 = np.gradient(d1, dt)
    
    accent_cmds_init = []
    curvature_threshold = -10.0 
    is_accent_region = d2 < curvature_threshold
    labeled_regions, num_regions =  find_contiguous_regions(is_accent_region)
    
    for start_idx, end_idx in labeled_regions:
        if times[end_idx] - times[start_idx] < 0.05: continue
        
        local_segment = smoothed_log_f0[start_idx:end_idx+1]
        peak_idx = start_idx + np.argmax(local_segment)
        t_peak = times[peak_idx]
        
        left_bound = max(0, start_idx - 10)
        right_bound = min(len(times)-1, end_idx + 10)
        local_base = min(smoothed_log_f0[left_bound], smoothed_log_f0[right_bound])
        
        aa_est = max(0.1, smoothed_log_f0[peak_idx] - local_base)
        
        half_width = (times[end_idx] - times[start_idx]) / 2.0
        t1 = max(0, t_peak - half_width)
        t2 = min(times[-1], t_peak + half_width)
        
        accent_cmds_init.append((t1, t2, aa_est * 1.5))

    temp_model = FujisakiModel()
    accent_contour = temp_model.generate_accent_component(times, accent_cmds_init)
    phrase_residual = smoothed_log_f0 - accent_contour
    
    fb_initial = np.exp(np.percentile(phrase_residual, 5))
    log_fb_init = np.log(fb_initial)
    
    phrase_motion = phrase_residual - log_fb_init
    phrase_motion[phrase_motion < 0] = 0
    
    d_phrase = np.gradient(phrase_motion, dt)
    phrase_cmds_init = [(times[0], 0.5)]
    last_t0 = times[0]
    
    p_peaks, p_props = find_peaks(d_phrase, height=0.5, distance=int(0.5/dt))
    
    for idx in p_peaks:
        t0 = times[idx]
        if t0 - last_t0 < 0.8: continue
        ap_est = p_props['peak_heights'][0] * 0.2
        ap_est = np.clip(ap_est, 0.1, 1.5)
        phrase_cmds_init.append((t0, ap_est))
        last_t0 = t0
        
    return fb_initial, phrase_cmds_init, accent_cmds_init

def objective_function(params, times, observed_log_f0, n_phrase, n_accent, model):
    fb = params[0]
    idx = 1
    phrase_cmds = []
    for _ in range(n_phrase):
        phrase_cmds.append((params[idx], params[idx+1]))
        idx += 2
    accent_cmds = []
    for _ in range(n_accent):
        accent_cmds.append((params[idx], params[idx+1], params[idx+2]))
        idx += 3
    
    generated_f0 = model.generate_contour(times, fb, phrase_cmds, accent_cmds)
    mask = ~np.isnan(observed_log_f0)
    residuals = np.log(generated_f0[mask] + 1e-6) - observed_log_f0[mask]
    return residuals

def analyze_single_file_paper_logic(file_path):
    """Super Lost準拠の解析関数"""
    y, sr = librosa.load(file_path, sr=None)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0[f0 > 800] = np.nan
    times = librosa.times_like(f0, sr=sr)
    log_f0 = np.log(f0)
    
    init_fb, init_phrases, init_accents = extract_initial_parameters_paper_logic(times, log_f0)
    
    n_phrase, n_accent = len(init_phrases), len(init_accents)
    x0 = [init_fb]
    bounds_min, bounds_max = [50.0], [500.0]
    
    for t0, ap in init_phrases:
        x0.extend([t0, ap]); bounds_min.extend([0, 0.01]); bounds_max.extend([times[-1], 5.0])
    for t1, t2, aa in init_accents:
        x0.extend([t1, t2, aa]); bounds_min.extend([0, 0, 0.01]); bounds_max.extend([times[-1], times[-1], 5.0])
        
    model = FujisakiModel()
    try:
        res = least_squares(objective_function, x0, bounds=(bounds_min, bounds_max), 
                            args=(times, log_f0, n_phrase, n_accent, model), loss='soft_l1', max_nfev=500)
        params = res.x
    except ValueError:
        print(f"Optimization failed for {file_path}")
        return None 

    fb_est = params[0]
    phrase_cmds = []
    idx = 1
    for _ in range(n_phrase):
        phrase_cmds.append((params[idx], params[idx+1]))
        idx += 2
    accent_cmds = []
    for _ in range(n_accent):
        t1, t2, aa = params[idx], params[idx+1], params[idx+2]
        if t1 > t2: t1, t2 = t2, t1
        accent_cmds.append((t1, t2, aa))
        idx += 3
        
    return {
        'times': times, 'f0': f0, 'fb': fb_est, 
        'phrase': phrase_cmds, 'accent': accent_cmds,
        'model_f0': model.generate_contour(times, fb_est, phrase_cmds, accent_cmds),
        'phrase_curve': model.generate_phrase_component(times, fb_est, phrase_cmds),
        'accent_curve': model.generate_accent_component(times, accent_cmds)
    }

# ==========================================
# 2. 面積計算ロジック (Super 4 準拠・積分法)
# ==========================================

def calculate_exact_area(times, accent_cmds):
    """
    【Super 4 準拠】
    アクセント指令から曲線を生成し、その面積を積分(台形則)で求める
    """
    if not accent_cmds:
        return 0.0
    
    model = FujisakiModel()
    # 時間軸全体にわたるアクセント曲線を生成
    accent_curve = model.generate_accent_component(times, accent_cmds)
    
    # 数値積分 (台形公式: np.trapz)
    area = np.trapz(np.abs(accent_curve), times)
    
    return area

def calculate_metrics_integral(res):
    """
    Super 4 と同じロジックで「積分面積(Integrated Area)」と「密度」を計算する
    """
    if res is None: return 0, 0, 0
    
    # 積分面積を計算
    total_area = calculate_exact_area(res['times'], res['accent'])
    
    # 時間
    duration = res['times'][-1] if len(res['times']) > 0 else 1.0
    
    # 密度
    density = total_area / duration
    
    return total_area, duration, density

# ==========================================
# 3. 可視化ロジック (Super Lost 準拠のグラフ + Super 4 の数値)
# ==========================================

def visualize_comparison_stacked(high_path, low_path):
    print(f"Analyzing High Urgency audio... {high_path}")
    res_high = analyze_single_file_paper_logic(high_path)
    
    print(f"Analyzing Low Urgency audio... {low_path}")
    res_low = analyze_single_file_paper_logic(low_path)
    
    if res_high is None or res_low is None:
        print("Analysis failed.")
        return

    # --- 数値計算 (積分面積は従来の対数ドメインでの計算値を維持します) ---
    h_area, h_dur, h_dens = calculate_metrics_integral(res_high)
    l_area, l_dur, l_dens = calculate_metrics_integral(res_low)

    print("\n" + "="*50)
    print("   QUANTITATIVE ANALYSIS (Integrated Area & Density)")
    print("="*50)
    print(f"High Urgency (Red):")
    print(f"  - Integrated Area (Energy) : {h_area:.2f}")
    print(f"  - Duration                 : {h_dur:.2f} s")
    print(f"  - Area Density (Area/s)    : {h_dens:.4f}")
    print(f"Low Urgency (Blue):")
    print(f"  - Integrated Area (Energy) : {l_area:.2f}")
    print(f"  - Duration                 : {l_dur:.2f} s")
    print(f"  - Area Density (Area/s)    : {l_dens:.4f}")
    
    ratio = h_dens / l_dens if l_dens > 0 else 0
    print("-" * 50)
    print(f"Density Ratio (High/Low): {ratio:.2f} times larger")
    print("="*50 + "\n")

    # --- プロット作成 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 上段: ピッチパターン & フレーズ成分 (ここは対数軸のままが一般的ですが、F0(Hz)表記なのでそのまま)
    ax1.plot(res_high['times'], res_high['f0'], '.', color='red', alpha=0.1, label='High Raw F0')
    ax1.plot(res_high['times'], res_high['model_f0'], '-', color='red', linewidth=2, label='High Model')
    ax1.plot(res_high['times'], res_high['phrase_curve'], '--', color='darkred', linewidth=1.5, alpha=0.7, label='High Phrase (Baseline)')

    ax1.plot(res_low['times'], res_low['f0'], '.', color='blue', alpha=0.1, label='Low Raw F0')
    ax1.plot(res_low['times'], res_low['model_f0'], '-', color='blue', linewidth=2, label='Low Model')
    ax1.plot(res_low['times'], res_low['phrase_curve'], '--', color='darkblue', linewidth=1.5, alpha=0.7, label='Low Phrase (Baseline)')

    ax1.set_title("1. F0 Contour & Phrase Component (Baseline)", fontsize=20, fontweight='bold')
    ax1.set_ylabel("Frequency (Hz)", fontsize=20)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax1.tick_params(axis='both', which='both', labelsize=20)
    ax1.grid(True, which='both', alpha=0.3)
    #ax1.legend(loc='upper right', ncol=2)

    # =================================================================
    # 下段: アクセント成分 (ここを修正：対数 -> 指数/リニア倍率)
    # =================================================================
    
    # データを指数変換 (Log Magnitude -> Frequency Ratio)
    # 例: 0 -> 1.0 (変化なし), 0.693 -> 2.0 (2倍)
    h_accent_linear = np.exp(res_high['accent_curve'])
    l_accent_linear = np.exp(res_low['accent_curve'])

    title_text = f"2. Accent Component (Frequency Ratio) - Linear Scale"
    
    # 凡例のAreaは対数計算のまま数値を表示（物理的エネルギー指標として）
    ax2.plot(res_high['times'], h_accent_linear, '-', color='red', linewidth=2, label=f'High Urgency (Log-Area={h_area:.1f})')
    # 基準線を 0 ではなく 1.0 に設定
    ax2.fill_between(res_high['times'], h_accent_linear, 1.0, color='red', alpha=0.2)

    ax2.plot(res_low['times'], l_accent_linear, '-', color='blue', linewidth=2, label=f'Low Urgency (Log-Area={l_area:.1f})')
    ax2.fill_between(res_low['times'], l_accent_linear, 1.0, color='blue', alpha=0.2)
    
    ax2.set_title(title_text, fontsize=20, fontweight='bold')
    ax2.set_ylabel("F0 Multiplier (Ratio)", fontsize=20) # ラベル変更: 倍率
    ax2.set_xlabel("Time (s)", fontsize=20)
    ax2.tick_params(axis='both', which='both', labelsize=20)
    ax2.grid(True, alpha=0.3)
    #ax2.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

# --- 実行 ---
# 分析したいファイル名を指定してください
visualize_comparison_stacked("tei_1.wav", "tei_17.wav")
