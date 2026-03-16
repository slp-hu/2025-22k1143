import os
import glob
import numpy as np
import librosa
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks

# ==========================================
# 1. 共通クラス・関数
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
    
    def generate_accent_component(self, times, accent_commands):
        """アクセント成分だけの曲線を生成する"""
        y = np.zeros_like(times)
        for t1, t2, aa in accent_commands:
            y += aa * (self.ga(times - t1) - self.ga(times - t2))
        return y 

def find_contiguous_regions(bool_array):
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

# ==========================================
# 2. 積分計算ロジック
# ==========================================

def calculate_metrics_both(times, accent_cmds):
    """
    アクセント指令から曲線を生成し、
    1. 対数領域の面積 (Log Area)
    2. 指数(リニア)領域の面積 (Linear Area)
    の両方を計算する。
    """
    if not accent_cmds:
        return 0.0, 0.0
    
    model = FujisakiModel()
    # 時間軸全体にわたるアクセント曲線を生成 (対数領域)
    accent_curve = model.generate_accent_component(times, accent_cmds)
    
    # 1. 対数領域の積分 (従来のArea)
    area_log = np.trapz(np.abs(accent_curve), times)
    
    # 2. 指数領域の積分 (F0倍率としてのArea)
    # 対数値を指数変換 (例: 0 -> 1.0, 0.693 -> 2.0)
    accent_curve_linear = np.exp(accent_curve)
    
    # ベースライン 1.0 (倍率1倍) との差分を積分
    area_linear = np.trapz(np.abs(accent_curve_linear - 1.0), times)
    
    return area_log, area_linear

def analyze_single_file_integral(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0[f0 > 800] = np.nan
        times = librosa.times_like(f0, sr=sr)
        log_f0 = np.log(f0)
        
        # 初期値推定
        init_fb, init_phrases, init_accents = extract_initial_parameters_paper_logic(times, log_f0)
        
        # 最適化
        n_phrase, n_accent = len(init_phrases), len(init_accents)
        x0 = [init_fb]
        bounds_min, bounds_max = [50.0], [500.0]
        
        for t0, ap in init_phrases:
            x0.extend([t0, ap]); bounds_min.extend([0, 0.01]); bounds_max.extend([times[-1], 5.0])
        for t1, t2, aa in init_accents:
            x0.extend([t1, t2, aa]); bounds_min.extend([0, 0, 0.01]); bounds_max.extend([times[-1], times[-1], 5.0])
            
        model = FujisakiModel()
        res = least_squares(objective_function, x0, bounds=(bounds_min, bounds_max), 
                            args=(times, log_f0, n_phrase, n_accent, model), loss='soft_l1', max_nfev=200)
        params = res.x
        
        # 最適化後のパラメータ抽出
        accent_cmds = []
        idx = 1 + 2*n_phrase
        for _ in range(n_accent):
            t1, t2, aa = params[idx], params[idx+1], params[idx+2]
            accent_cmds.append((t1, t2, aa))
            idx += 3
            
        # ★計算実行★
        area_log, area_linear = calculate_metrics_both(times, accent_cmds)
        
        return area_log, area_linear, times[-1]
    
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return 0, 0, 0

# ==========================================
# 3. フォルダ解析・比較実行
# ==========================================

def analyze_folder_integral(folder_path, label):
    files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.mp3"))
    files.sort()
    
    print(f"\n--- 解析中: {label} ({len(files)} files) ---")
    print(f"{'Filename':<25} | {'Dur(s)':<6} | {'Log Area':<10} | {'Lin Area':<10} | {'Log Dens':<10} | {'Lin Dens':<10}")
    print("-" * 85)
    
    results = []
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        area_log, area_linear, duration = analyze_single_file_integral(file_path)
        
        if duration > 0:
            density_log = area_log / duration
            density_linear = area_linear / duration
        else:
            density_log = 0
            density_linear = 0
            
        print(f"{filename:<25} | {duration:>6.2f} | {area_log:>10.4f} | {area_linear:>10.4f} | {density_log:>10.4f} | {density_linear:>10.4f}")
        
        results.append({
            'area_log': area_log,
            'area_linear': area_linear,
            'duration': duration,
            'density_log': density_log,
            'density_linear': density_linear
        })
            
    return results

def compare_folders(tsunami_path, earthquake_path):
    tsunami_data = analyze_folder_integral(tsunami_path, "High Urgency (津波)")
    earthquake_data = analyze_folder_integral(earthquake_path, "Low Urgency (地震)")
    
    if not tsunami_data or not earthquake_data:
        print("\nエラー: 解析できませんでした。")
        return

    # --- 1. 密度 (Density) の集計 ---
    avg_dens_log_tsu = np.mean([d['density_log'] for d in tsunami_data])
    avg_dens_log_eq = np.mean([d['density_log'] for d in earthquake_data])
    
    avg_dens_lin_tsu = np.mean([d['density_linear'] for d in tsunami_data])
    avg_dens_lin_eq = np.mean([d['density_linear'] for d in earthquake_data])

    # --- 2. 総面積 (Total Area) の集計 ---
    sum_area_log_tsu = np.sum([d['area_log'] for d in tsunami_data])
    sum_area_log_eq = np.sum([d['area_log'] for d in earthquake_data])
    
    sum_area_lin_tsu = np.sum([d['area_linear'] for d in tsunami_data])
    sum_area_lin_eq = np.sum([d['area_linear'] for d in earthquake_data])

    print("\n" + "="*75)
    print("   【最終結果】 密度(Density) と 総面積(Total Area) の比較")
    print("="*75)
    
    # ----------------------------------------------------
    # 対数ベース (Log Domain)
    # ----------------------------------------------------
    print(f"■ 1. 対数ベース (Log Domain)")
    print(f"   [Density / 平均密度]")
    print(f"     - High: {avg_dens_log_tsu:.4f}")
    print(f"     - Low : {avg_dens_log_eq:.4f}")
    if avg_dens_log_eq > 0:
        print(f"     >>> Ratio: {avg_dens_log_tsu / avg_dens_log_eq:.2f}倍")
    
    print(f"   [Total Area / 総面積]")
    print(f"     - High: {sum_area_log_tsu:.2f}")
    print(f"     - Low : {sum_area_log_eq:.2f}")
    if sum_area_log_eq > 0:
        print(f"     >>> Ratio: {sum_area_log_tsu / sum_area_log_eq:.2f}倍")

    print("-" * 75)

    # ----------------------------------------------------
    # 指数/リニアベース (Linear Domain)
    # ----------------------------------------------------
    print(f"■ 2. 指数/リニアベース (Linear Domain - F0 Multiplier)")
    print(f"   [Density / 平均密度]")
    print(f"     - High: {avg_dens_lin_tsu:.4f}")
    print(f"     - Low : {avg_dens_lin_eq:.4f}")
    if avg_dens_lin_eq > 0:
        print(f"     >>> Ratio: {avg_dens_lin_tsu / avg_dens_lin_eq:.2f}倍")

    print(f"   [Total Area / 総面積]")
    print(f"     - High: {sum_area_lin_tsu:.2f}")
    print(f"     - Low : {sum_area_lin_eq:.2f}")
    if sum_area_lin_eq > 0:
        print(f"     >>> Ratio: {sum_area_lin_tsu / sum_area_lin_eq:.2f}倍")
        
    print("="*75 + "\n")

# --- 設定 ---
tsunami_folder = "./koukin"
earthquake_folder = "./teikin"

if __name__ == "__main__":
    if os.path.exists(tsunami_folder) and os.path.exists(earthquake_folder):
        compare_folders(tsunami_folder, earthquake_folder)
    else:
        print("フォルダが見つかりません。パスを確認してください。")
