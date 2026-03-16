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
    if not accent_cmds:
        return 0.0
    model = FujisakiModel()
    accent_curve = model.generate_accent_component(times, accent_cmds)
    area = np.trapz(np.abs(accent_curve), times)
    return area

def calculate_metrics_integral(res):
    if res is None: return 0, 0, 0
    total_area = calculate_exact_area(res['times'], res['accent'])
    duration = res['times'][-1] if len(res['times']) > 0 else 1.0
    density = total_area / duration
    return total_area, duration, density

# ==========================================
# 3. 可視化ロジック (3ファイル比較版)
# ==========================================

def visualize_comparison_stacked_3(path1, path2, path3):
    print(f"Analyzing Audio 1... {path1}")
    res1 = analyze_single_file_paper_logic(path1)
    
    print(f"Analyzing Audio 2... {path2}")
    res2 = analyze_single_file_paper_logic(path2)

    print(f"Analyzing Audio 3... {path3}")
    res3 = analyze_single_file_paper_logic(path3)
    
    if res1 is None or res2 is None or res3 is None:
        print("Analysis failed for one or more files.")
        return

    # --- 数値計算 ---
    area1, dur1, dens1 = calculate_metrics_integral(res1)
    area2, dur2, dens2 = calculate_metrics_integral(res2)
    area3, dur3, dens3 = calculate_metrics_integral(res3)

    print("\n" + "="*50)
    print("   QUANTITATIVE ANALYSIS (Integrated Area & Density)")
    print("="*50)
    print(f"Audio 1 (Red):")
    print(f"  - Integrated Area (Energy) : {area1:.2f}")
    print(f"  - Duration                 : {dur1:.2f} s")
    print(f"  - Area Density (Area/s)    : {dens1:.4f}")
    print(f"Audio 2 (Green):")
    print(f"  - Integrated Area (Energy) : {area2:.2f}")
    print(f"  - Duration                 : {dur2:.2f} s")
    print(f"  - Area Density (Area/s)    : {dens2:.4f}")
    print(f"Audio 3 (Blue):")
    print(f"  - Integrated Area (Energy) : {area3:.2f}")
    print(f"  - Duration                 : {dur3:.2f} s")
    print(f"  - Area Density (Area/s)    : {dens3:.4f}")
    print("="*50 + "\n")

    # --- プロット作成 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # 上段: ピッチパターン & フレーズ成分
    # Audio 1 (Red)
    ax1.plot(res1['times'], res1['f0'], '.', color='red', alpha=0.1, label='Audio 1 Raw F0')
    ax1.plot(res1['times'], res1['model_f0'], '-', color='red', linewidth=2, label='Audio 1 Model')
    ax1.plot(res1['times'], res1['phrase_curve'], '--', color='darkred', linewidth=1.5, alpha=0.7, label='Audio 1 Phrase')

    # Audio 2 (Green)
    ax1.plot(res2['times'], res2['f0'], '.', color='green', alpha=0.1, label='Audio 2 Raw F0')
    ax1.plot(res2['times'], res2['model_f0'], '-', color='green', linewidth=2, label='Audio 2 Model')
    ax1.plot(res2['times'], res2['phrase_curve'], '--', color='darkgreen', linewidth=1.5, alpha=0.7, label='Audio 2 Phrase')

    # Audio 3 (Blue)
    ax1.plot(res3['times'], res3['f0'], '.', color='blue', alpha=0.1, label='Audio 3 Raw F0')
    ax1.plot(res3['times'], res3['model_f0'], '-', color='blue', linewidth=2, label='Audio 3 Model')
    ax1.plot(res3['times'], res3['phrase_curve'], '--', color='darkblue', linewidth=1.5, alpha=0.7, label='Audio 3 Phrase')

    # タイトル・軸名を大きく
    ax1.set_title("1. F0 Contour & Phrase Component (Baseline)", fontsize=20, fontweight='bold')
    ax1.set_ylabel("Frequency (Hz)", fontsize=20)
    
    # 対数表示と目盛りサイズの調整
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax1.tick_params(axis='both', which='both', labelsize=20)
    ax1.grid(True, which='both', alpha=0.3)
    # ax1.legend(loc='upper right', ncol=3, fontsize=10)

    # =================================================================
    # 下段: アクセント成分
    # =================================================================
    
    accent1_linear = np.exp(res1['accent_curve'])
    accent2_linear = np.exp(res2['accent_curve'])
    accent3_linear = np.exp(res3['accent_curve'])

    title_text = f"2. Accent Component (Frequency Ratio) - Log Scale"
    
    # Audio 1 (Red)
    ax2.plot(res1['times'], accent1_linear, '-', color='red', linewidth=2, label=f'Audio 1 (Log-Area={area1:.1f})')
    ax2.fill_between(res1['times'], accent1_linear, 1.0, color='red', alpha=0.2)

    # Audio 2 (Green)
    ax2.plot(res2['times'], accent2_linear, '-', color='green', linewidth=2, label=f'Audio 2 (Log-Area={area2:.1f})')
    ax2.fill_between(res2['times'], accent2_linear, 1.0, color='green', alpha=0.2)

    # Audio 3 (Blue)
    ax2.plot(res3['times'], accent3_linear, '-', color='blue', linewidth=2, label=f'Audio 3 (Log-Area={area3:.1f})')
    ax2.fill_between(res3['times'], accent3_linear, 1.0, color='blue', alpha=0.2)
    
    # タイトル・軸名を大きく
    ax2.set_title(title_text, fontsize=20, fontweight='bold')
    ax2.set_ylabel("F0 Multiplier (Ratio)", fontsize=20)
    ax2.set_xlabel("Time (s)", fontsize=20)
    
    # 対数表示と目盛りサイズの調整
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax2.tick_params(axis='both', which='both', labelsize=14)
    ax2.grid(True, which='both', alpha=0.3)
    # ax2.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

# --- 実行 ---
# 分析したい3つのファイル名を指定してください
visualize_comparison_stacked_3("津波警報_16.wav", "NHK_YI_news3_17.wav", "地震速報_0102.wav")
