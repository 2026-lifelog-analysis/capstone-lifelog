"""
regularity.py — 수면 리듬 탐지기와 수면 시각 흔들림 시나리오 (김민준, 모듈 1).

공통 코드(src/)는 고치지 않고 가져다 쓴다. 이 파일이 새로 하는 일은 세 가지다.

1) 지표  sleep_mid_sd : 수면 중앙 시각 (잠든 시각 + 깬 시각) / 2 의 최근 7일 표준편차 (그날 포함, t−6 ~ t).
          그날 기록이 있고 7일 중 5일 이상 기록이 있을 때만 값이 생긴다. 미래 날짜는 쓰지 않는다.
          이 값을 bl.personal_z (직전 28일 기준선) 와 CUSUM 에 넣으면 '수면 중앙 시각 SD 탐지기'가 된다.
2) 시나리오 jitter_sleep : 14일 구간 안에서 잠든 시각·깬 시각을 날마다 N(0, σ²) 만큼 옮긴다.
          구간 안 기록 날의 이동 평균을 0으로 맞추므로 14일 평균 취침 시각은 전과 같다. 수면량도 그대로다.
          구간은 공통 ev.inject_drift 와 같은 규칙으로 고른다 (효과 0으로 호출해 구간표만 받음).
3) 채점  방식 A = 공통 노트북 02 와 같다 (하루 경보율 1/30 기준값 하나, 14일 안 경보 한 번이면 탐지).
          방식 B = 14일 구간 오경보율을 0.20 에 맞춘 기준값으로 같은 계산을 한다 (비용을 같게 두고 비교).
          J = 탐지율 − 오경보율 (사람마다). 비교는 같은 사람끼리 짝지은 부트스트랩 CI 와 Wilcoxon.

비교 대상 탐지기: 걸음·취침·수면량 각각의 개인 z, |개인 z|, 개인+CUSUM (공통 코드와 같은 점수) 9개
                 + 수면 중앙 시각 SD 의 개인 z, 개인+CUSUM 2개.
"""
import numpy as np
import pandas as pd

from src import baseline as bl, evaluation as ev

RATE = ev.ALARM_RATE          # 방식 A 하루 경보율 1/30
WIN_FA = 0.20                 # 방식 B 14일 구간 오경보율
K = 0.5                       # CUSUM 허용치 (공통 기본값)
LENGTH = 14                   # 주입 구간 길이
N_REPS = 10                   # 반복 (seed 0 ~ 9)
SD_WINDOW, SD_MIN_DAYS = 7, 5 # 7일 표준편차, 최소 5일
CROP = 75                     # 주입 계산 때 구간 시작 전 며칠까지 잘라 쓰는지 (28일 기준선 + 7일 창 + 여유)
SIGMA = 90.0                  # 흔들림 시나리오 기본 σ (분)

LEVEL = ['steps', 'bed_min', 'sleep_min']
REG = 'mid_sd'
METRICS = LEVEL + [REG]
DIRECTION = {'steps': -1, 'bed_min': +1, 'sleep_min': -1, REG: +1}   # 나쁜 방향
KO = {'steps': '걸음', 'bed_min': '취침', 'sleep_min': '수면량', REG: '수면 중앙 시각 SD'}
DETS = {'z': '개인 z', 'az': '|개인 z|', 'cs': '개인+CUSUM'}
SCEN = {
    'S14': ('steps', '걸음 14일 −30% (공통 시나리오)'),
    'BED-S14': ('bed_min', '취침 14일 +120분 (공통 시나리오)'),
    'JB60': ('bed_min', '취침·기상 흔들림 SD 60분, 평균 0'),
    'JB90': ('bed_min', '취침·기상 흔들림 SD 90분, 평균 0 (주 시나리오)'),
    'JB120': ('bed_min', '취침·기상 흔들림 SD 120분, 평균 0'),
}
JB_SIGMA = {'JB60': 60.0, 'JB90': 90.0, 'JB120': 120.0}


def name(metric, det):
    return f'{metric}|{det}'


def label(det):
    m, k = det.split('|')
    return f'{KO[m]} {DETS[k]}'


def kinds_for(m):
    return ['z', 'az', 'cs'] if m in LEVEL else ['z', 'cs']


def all_detectors():
    return [name(m, d) for m in METRICS for d in kinds_for(m)]


# ---------------------------------------------------------------------------
# 1) 지표
# ---------------------------------------------------------------------------
def add_sleep_mid(nh):
    """하루 표에 wake_noon(정오 기준 기상 시각, 취침 뒤로 펼침)과 mid(수면 중앙 시각)를 붙인다.
    공통 하루 표의 wake_min 은 18시 기준 분이므로 +360 해서 정오 기준으로 바꾼다."""
    nh = nh.copy()
    nh['wake_noon'] = nh['bed_min'] + (nh['wake_min'] + 360 - nh['bed_min']) % 1440
    nh['mid'] = (nh['bed_min'] + nh['wake_noon']) / 2
    return nh


def roll_sd(x, k=SD_WINDOW, minp=SD_MIN_DAYS):
    """달력 하루 간격 배열 x 에서 t−k+1 ~ t 창의 표준편차(ddof=1). 그날 빈칸이거나 창 안 기록이 minp 미만이면 빈칸.
    창은 과거 쪽으로만 열린다 (미래 값은 들어가지 않는다)."""
    x = np.asarray(x, float)
    n = len(x)
    pad = np.concatenate([np.full(k - 1, np.nan), x])
    W = np.lib.stride_tricks.sliding_window_view(pad, k)[:n]
    cnt = np.sum(~np.isnan(W), 1)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sd = np.nanstd(W, axis=1, ddof=1)
    sd[(cnt < minp) | np.isnan(x)] = np.nan
    return sd


def sleep_mid_sd(df):
    """df: pid, date, mid. 반환: df 와 같은 index 의 Series (수면 중앙 시각 7일 SD)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for pid, idx, dates, full in bl._per_person_daily(df, 'mid'):
        sd = roll_sd(full.to_numpy(float))
        out.loc[idx] = pd.Series(sd, index=full.index).reindex(dates.values).to_numpy()
    return out


# ---------------------------------------------------------------------------
# 점수 (공통 코드와 같은 개인 z, |z|; CUSUM 은 아래 cusum_full 로)
# ---------------------------------------------------------------------------
def det_scores(df):
    out = {}
    for m in METRICS:
        z = bl.personal_z(df, m)
        out[name(m, 'z')] = z * DIRECTION[m]
        if m in LEVEL:
            out[name(m, 'az')] = z.abs()
    return pd.DataFrame(out, index=df.index)


def _parts(df, score):
    d = pd.DataFrame({'pid': df['pid'].to_numpy(), 'date': df['date'].to_numpy(),
                      'v': np.asarray(score, float)}, index=df.index)
    d = d.sort_values(['pid', 'date'], kind='stable')
    pids = d['pid'].to_numpy()
    cut = np.flatnonzero(pids[1:] != pids[:-1]) + 1
    return d, np.split(d['v'].to_numpy(float), cut)


def _run(vals, k, h):
    """bl._cusum_run 과 같은 규칙: S = max(0, S + v − k), S > h 면 경보 후 0. 빈 날은 S 유지·출력 빈칸."""
    s, r = 0.0, np.full(len(vals), np.nan)
    for i, v in enumerate(vals.tolist()):
        if v != v:
            continue
        s = max(0.0, s + v - k)
        r[i] = s
        if s > h:
            s = 0.0
    return r


def cusum_full(df, score, h, k=K):
    d, parts = _parts(df, score)
    r = np.concatenate([_run(v, k, h) for v in parts])
    return pd.Series(r, index=d.index).reindex(df.index)


def cusum_segment(df, clean_S, seg, inj_score, h, k=K):
    """주입 구간 날의 CUSUM S. 구간 전 점수는 대조군과 같으므로 구간 직전의 대조군 상태(경보였으면 0)에서 이어 계산."""
    cs = pd.DataFrame({'pid': df['pid'].to_numpy(), 'date': df['date'].to_numpy(), 'S': clean_S.to_numpy()})
    cs = cs.dropna(subset=['S']).sort_values(['pid', 'date'])
    by = {p: (g['date'].to_numpy(), g['S'].to_numpy()) for p, g in cs.groupby('pid', sort=False)}
    out = np.full(len(seg), np.nan)
    sv = seg[['rep', 'pid', 'date', 'start']].assign(_v=np.asarray(inj_score, float), _i=np.arange(len(seg)))
    sv = sv.sort_values(['rep', 'pid', 'date'])
    for (rep, pid), g in sv.groupby(['rep', 'pid'], sort=False):
        s = 0.0
        if pid in by:
            dts, S = by[pid]
            j = np.searchsorted(dts, g['start'].iloc[0].to_datetime64()) - 1
            if j >= 0:
                s = 0.0 if S[j] > h else float(S[j])
        for i, v in zip(g['_i'].to_numpy(), g['_v'].to_numpy()):
            if v != v:
                continue
            s = max(0.0, s + v - k)
            out[i] = s
            if s > h:
                s = 0.0
    return out


def clean_scores(nh):
    """대조군(주입 없음) 전체 점수. 반환 (지표 붙은 nh, 점수표 S, 방식 A 의 CUSUM 경보선 h_A)."""
    nh = add_sleep_mid(nh)
    nh[REG] = sleep_mid_sd(nh)
    S = det_scores(nh)
    h_A = {}
    for m in METRICS:
        z = S[name(m, 'z')]
        h_A[m] = bl.cusum_threshold(nh.assign(_b=z), '_b', k=K, rate=RATE)
        S[name(m, 'cs')] = cusum_full(nh, z, h_A[m])
    return nh, S[all_detectors()], h_A


# ---------------------------------------------------------------------------
# 2) 시나리오
# ---------------------------------------------------------------------------
def segments(nh, scen, seed):
    """공통 inject_drift 로 구간표(pid, start, end)만 받는다 (효과 0)."""
    _, seg = ev.inject_drift(nh, SCEN[scen][0], 0.0, mode='absolute', length=LENGTH, shape='sudden', seed=seed)
    return seg


def crop(nh, seg, before=CROP):
    """구간 사람마다 [start − before, end] 날만 남긴다. 개인 기준선(28일)과 7일 창이 과거만 보므로 결과는 전체 자료와 같다."""
    m = nh[['pid', 'date']].reset_index().merge(seg, on='pid')
    m = m[(m['date'] >= m['start'] - pd.Timedelta(days=before)) & (m['date'] <= m['end'])]
    return nh.loc[m['index'].to_numpy()].copy()


def jitter_sleep(d, seg, sigma, seed):
    """구간 날의 bed_min, wake_noon, wake_min 을 δ_d 만큼 옮긴다. δ_d = σ·(z_d − 구간 안 기록 날 z 평균), z ~ N(0,1).
    사람마다 난수 14개를 뽑고 σ 는 곱하기만 하므로 σ 가 달라도 같은 난수를 쓴다."""
    d = d.copy()
    rows = ev._segment_rows(d, seg)
    day = (rows['date'] - rows['start']).dt.days.to_numpy()
    rng = np.random.default_rng(10_000 + seed)
    Z = {p: rng.standard_normal(LENGTH) for p in seg['pid']}
    z = np.array([Z[p][k] for p, k in zip(rows['pid'], day)])
    b = d.loc[rows.index, 'bed_min'].to_numpy(float)
    dz = pd.Series(np.where(np.isnan(b), np.nan, z), index=rows.index)
    dz = dz - dz.groupby(rows['pid']).transform('mean')
    delta = (sigma * dz).to_numpy()
    for c in ['bed_min', 'wake_noon']:
        d.loc[rows.index, c] = d.loc[rows.index, c].to_numpy(float) + delta
    d.loc[rows.index, 'wake_min'] = (d.loc[rows.index, 'wake_min'].to_numpy(float) + delta) % 1440
    d['mid'] = (d['bed_min'] + d['wake_noon']) / 2
    return d


def apply_scenario(d, seg, scen, seed):
    if scen == 'S14':
        d = d.copy()
        rows = ev._segment_rows(d, seg)
        d.loc[rows.index, 'steps'] = d.loc[rows.index, 'steps'].to_numpy(float) * 0.7
        return d
    if scen == 'BED-S14':
        d = d.copy()
        rows = ev._segment_rows(d, seg)
        d.loc[rows.index, 'bed_min'] = d.loc[rows.index, 'bed_min'].to_numpy(float) + 120
        d['mid'] = (d['bed_min'] + d['wake_noon']) / 2
        return d
    if scen in JB_SIGMA:
        return jitter_sleep(d, seg, JB_SIGMA[scen], seed)
    raise ValueError(scen)


def scores_for(d):
    d = d.copy()
    d[REG] = sleep_mid_sd(d)
    return det_scores(d)


def collect(nh, S, scen, n_reps=N_REPS):
    """시나리오 하나의 구간 날 표. 열: rep, pid, date, start, day, idx, c|<탐지기>(대조군), i|<탐지기>(주입)."""
    cols = [c for c in S.columns if not c.endswith('|cs')]
    parts = []
    for r in range(n_reps):
        seg = segments(nh, scen, r)
        d = apply_scenario(crop(nh, seg), seg, scen, r)
        sc = scores_for(d)
        rows = ev._segment_rows(nh, seg)
        t = rows.reset_index().rename(columns={'index': 'idx'})
        t['rep'] = r
        t['day'] = (t['date'] - t['start']).dt.days
        for c in cols:
            t[f'c|{c}'] = S.loc[rows.index, c].to_numpy()
            t[f'i|{c}'] = sc.loc[rows.index, c].to_numpy()
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3) 채점
# ---------------------------------------------------------------------------
def window_fa(seg, a):
    g = pd.DataFrame({'rep': seg['rep'].to_numpy(), 'pid': seg['pid'].to_numpy(), 'a': np.asarray(a, bool)})
    return float(g.groupby(['rep', 'pid'])['a'].any().mean())


def alarms(nh, clean, seg, det, scheme, h_A):
    """반환 (기준값, 대조군 구간 경보, 주입 구간 경보). 빈 점수는 경보 아님."""
    m, kind = det.split('|')
    if kind == 'cs':
        base = name(m, 'z')
        idx = seg['idx'].to_numpy()
        if scheme == 'A':
            h = h_A[m]
        else:
            def fa(hh):
                S = cusum_full(nh, clean[base], hh)
                return window_fa(seg, S.loc[idx].to_numpy() > hh)
            lo, hi = 0.0, 60.0
            for _ in range(22):
                mid = (lo + hi) / 2
                if fa(mid) > WIN_FA:
                    lo = mid
                else:
                    hi = mid
            h = hi
        S = cusum_full(nh, clean[base], h)
        a_ctl = S.loc[idx].to_numpy() > h
        a_inj = cusum_segment(nh, S, seg, seg[f'i|{base}'], h) > h
        return h, a_ctl, a_inj
    if scheme == 'A':
        t = float(clean[det].quantile(1 - RATE))
    else:
        sid = seg['rep'].astype(str) + '_' + seg['pid'].astype(str)
        Mc = pd.Series(seg[f'c|{det}'].to_numpy()).groupby(sid.to_numpy()).max().fillna(-np.inf)
        t = float(np.quantile(Mc.to_numpy(), 1 - WIN_FA))
    return t, seg[f'c|{det}'].to_numpy() > t, seg[f'i|{det}'].to_numpy() > t


def seg_metrics(seg, a_ctl, a_inj):
    """사람별 탐지율·오경보율·J·지연 (ev.evaluate_injection 의 4단계 이후와 같은 계산)."""
    g = pd.DataFrame({'rep': seg['rep'].to_numpy(), 'pid': seg['pid'].to_numpy(), 'day': seg['day'].to_numpy(),
                      'ai': np.asarray(a_inj, bool), 'ac': np.asarray(a_ctl, bool)})
    g['dd'] = g['day'].where(g['ai'])
    w = g.groupby(['rep', 'pid']).agg(detected=('ai', 'any'), fa=('ac', 'any'), delay=('dd', 'min')).reset_index()
    p = w.groupby('pid').agg(탐지율=('detected', 'mean'), 오경보율=('fa', 'mean'), 탐지지연=('delay', 'median'))
    p['차이'] = p['탐지율'] - p['오경보율']
    return p.reset_index(), w


def evaluate(nh, S, h_A, seg, scen):
    rows, pps = [], []
    for det in all_detectors():
        for sch in ['A', 'B']:
            thr, a_ctl, a_inj = alarms(nh, S, seg, det, sch, h_A)
            pp, w = seg_metrics(seg, a_ctl, a_inj)
            cd = w[w['detected'] & ~w['fa']]
            rows.append({'시나리오': scen, '방식': sch, '탐지기': det, '이름': label(det),
                         '종류': '규칙성' if det.startswith(REG) else '수준', '기준값': thr,
                         '탐지율': w['detected'].mean(), '구간FA': w['fa'].mean(),
                         'J평균': pp['차이'].mean(), 'J중앙': pp['차이'].median(),
                         '지연중앙': cd['delay'].median(), '사람수': len(pp)})
            pps.append(pp.assign(시나리오=scen, 방식=sch, 탐지기=det))
    return pd.DataFrame(rows), pd.concat(pps, ignore_index=True)


def paired_boot_ci(pa, pb, col='차이', B=2000, seed=0):
    x = pa[['pid', col]].merge(pb[['pid', col]], on='pid', suffixes=('_a', '_b')).dropna()
    d = (x[f'{col}_a'] - x[f'{col}_b']).to_numpy(float)
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)), len(d)


def paired_wilcoxon(pa, pb, col='차이'):
    """같은 사람끼리 짝 Wilcoxon. 반복 10번이라 J 는 0.1 격자이므로 0.1 단위로 반올림해 동률을 보존."""
    from scipy.stats import wilcoxon
    x = pa[['pid', col]].merge(pb[['pid', col]], on='pid', suffixes=('_a', '_b')).dropna()
    d = np.round((x[f'{col}_a'] - x[f'{col}_b']).to_numpy(float), 1)
    return float(wilcoxon(d).pvalue) if (d != 0).any() else np.nan
