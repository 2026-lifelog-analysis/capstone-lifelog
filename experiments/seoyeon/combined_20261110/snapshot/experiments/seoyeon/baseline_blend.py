"""
baseline_blend.py — ③ 개인·집단 기준선 결합 (오늘 밤 수면 예측, 공통 프로토콜 위에서)

질문: 개인 기록이 적을 때 개인 기준과 집단 기준을 섞으면 오차가 줄어드는가.
      기록량에 따라 비율을 바꾸는 결합이 고정 비율 결합보다 나은가.

    from src import forecast_dataset as fd, forecast_run as fr
    from experiments.seoyeon import baseline_blend as bb
    data, split, man, out = fr.prepare()
    table = fd.add_features(data, bb.feat_personal_history)
    fr.run(data, split, man, out, experiments=bb.experiments(), table=table, baseline='personal_mean7')

비교하는 기준선 (목표 sleep / bed / wake 각각 따로)
  group_median      : 학습 참가자 정답 중앙값 하나 (공통 fe.GroupMedian)
  personal_mean7    : 과거 달력 7밤 평균 (공통 fe.PersonalMean, 팀 기본 비교 대상)
  personal_median28 : 과거 달력 28밤 중앙값. 기록이 하나도 없으면 집단 값
  fixed_blend28     : w × 개인 + (1 − w) × 집단. w는 학습 참가자 밤에서 MAE가 가장 작은 값 (W_GRID)
  shrink_blend28    : w = n / (n + k), n = 과거 28밤 중 유효 기록 수. 기록이 적을수록 집단 쪽으로 당김.
                      k는 학습 참가자 밤에서 MAE가 가장 작은 값 (K_GRID)

규칙
  - 개인 과거 값은 fd.lag_features와 같은 규칙으로 가린다: 밤 D−1 ~ D−WINDOW (달력), 예측 기준 시각까지
    끝나지 않은 밤·겹침 밤·minsasleep 무효 밤은 NaN. 밤 D 자신과 그 뒤는 쓰지 않는다.
  - 집단 값·w·k는 fit(train, ...) 안에서 학습 참가자 밤으로만 정한다. 검증 참가자의 정답은 쓰지 않는다.
  - 공통 예측 표의 열은 바꾸지 않고 열만 더한다 (fd.add_features).
"""
import numpy as np
import pandas as pd

from src import forecast_dataset as fd
from src import forecast_eval as fe

WINDOW = 28                                    # 개인 과거 창 (달력 밤 수)
K_GRID = (0, 0.5, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 60, 100)
W_GRID = tuple(round(w, 2) for w in np.arange(0, 1.0001, 0.05))

# 개인 과거 기록 수 구간 (WINDOW=28 기준)
BIN_EDGES = [-0.5, 0.5, 3.5, 7.5, 14.5, 21.5, 28.5]
BIN_LABELS = ['0', '1~3', '4~7', '8~14', '15~21', '22~28']

TARGET_OF_COL = {col: t for t, col in fd.TARGETS.items()}

# fit 때마다 고른 값을 남김 (run_cv 순서: 목표 → fold → 실험). run_blend.py가 fold 번호를 붙여 저장
PARAM_LOG = []


# ---------- 개인 과거 특징 ----------

def _names(target, window=WINDOW):
    p = fd.LAG_PREFIX[target]
    return f'{p}_pmed{window}', f'{p}_n{window}'


def blend_features(target, window=WINDOW):
    """목표별 추가 입력 이름: [개인 과거 중앙값, 개인 과거 유효 기록 수]."""
    return list(_names(target, window))


def personal_history(table, nights, window=WINDOW):
    """
    table의 줄마다 밤 D−1 ~ D−window의 개인 기록으로 중앙값·개수를 계산.
    가리는 규칙은 fd.lag_features와 같다 (예측 시점 미확정·겹침·무효 밤 → NaN).
    반환: table과 같은 index, 목표마다 {prefix}_pmed{window}, {prefix}_n{window}
    """
    lags, _ = fd.lag_features(table[['pid', 'date', 'forecast_at']], nights, n_lags=window)
    out = pd.DataFrame(index=table.index)
    for t in fd.TARGETS:
        p = fd.LAG_PREFIX[t]
        vals = lags[[f'{p}_lag{k}' for k in range(1, window + 1)]]
        med, cnt = _names(t, window)
        out[med] = vals.median(axis=1, skipna=True)
        out[cnt] = vals.notna().sum(axis=1).astype(int)
    return out


def feat_personal_history(data):
    """fd.add_features에 넘기는 특징 함수 (WINDOW=28)."""
    return personal_history(data.table, data.nights)


# ---------- 모델 (공통 fit/predict 형식) ----------

class _Blend:
    """개인 과거 중앙값과 집단 중앙값을 섞는 기준선의 공통 부분."""
    name = 'blend'

    def fit(self, train, y, features):
        self.target_ = TARGET_OF_COL[y]
        self.med_, self.cnt_ = _names(self.target_)
        missing = [c for c in (self.med_, self.cnt_) if c not in features]
        if missing:
            raise ValueError(f'{type(self).__name__}에는 입력 {missing}가 필요함 (bb.blend_features)')
        self.group_ = float(train[y].median())
        self._choose(train, train[y].to_numpy(dtype=float))
        PARAM_LOG.append({'model': type(self).__name__, 'target': self.target_,
                          'n_train_nights': int(len(train)), **self.params_()})
        return self

    def _choose(self, train, y):
        """학습 참가자 밤에서 매개변수를 고름 (기본: 고를 것 없음)."""

    @staticmethod
    def _best(grid, preds_for, y):
        """grid 값마다 MAE를 내고 가장 작은 값 (같으면 먼저 나온 값)."""
        maes = [float(np.mean(np.abs(y - preds_for(g)))) for g in grid]
        return grid[int(np.argmin(maes))]

    def predict(self, df):
        return self._blend(df)

    def params_(self):
        return {'group': self.group_}


class PersonalMedian(_Blend):
    """개인 과거 28밤 중앙값. 기록이 없으면 학습 참가자 정답 중앙값."""
    def _blend(self, df):
        return df[self.med_].fillna(self.group_).to_numpy(dtype=float)


class FixedBlend(_Blend):
    """w × 개인 + (1 − w) × 집단. 개인 기록이 없으면 집단 값. w ∈ W_GRID."""
    def _mix(self, df, w):
        pmed = df[self.med_].to_numpy(dtype=float)
        out = w * pmed + (1 - w) * self.group_
        return np.where(np.isnan(pmed), self.group_, out)

    def _choose(self, train, y):
        self.w_ = self._best(W_GRID, lambda w: self._mix(train, w), y)

    def _blend(self, df):
        return self._mix(df, self.w_)

    def params_(self):
        return {'group': self.group_, 'w': self.w_}


class ShrinkBlend(_Blend):
    """집단 + n/(n+k) × (개인 − 집단). n = 과거 28밤 유효 기록 수. n = 0이면 집단 값. k ∈ K_GRID."""
    def _mix(self, df, k):
        pmed = df[self.med_].to_numpy(dtype=float)
        n = df[self.cnt_].to_numpy(dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            w = np.where(n > 0, n / (n + k), 0.0)
        out = self.group_ + w * (pmed - self.group_)
        return np.where(np.isnan(pmed), self.group_, out)

    def _choose(self, train, y):
        self.k_ = self._best(K_GRID, lambda k: self._mix(train, k), y)

    def _blend(self, df):
        return self._mix(df, self.k_)

    def params_(self):
        return {'group': self.group_, 'k': self.k_}


# ---------- 실험 목록 ----------

def _with_history(target):
    return fd.base_features(target) + blend_features(target)


def experiments():
    """③에서 비교하는 다섯 기준선 (공통 기본 두 개 + 새 세 개)."""
    return [fe.EXPERIMENTS['group_median'],
            fe.EXPERIMENTS['personal_mean7'],
            fe.Experiment('personal_median28', PersonalMedian, _with_history),
            fe.Experiment('fixed_blend28', FixedBlend, _with_history),
            fe.Experiment('shrink_blend28', ShrinkBlend, _with_history)]


# ---------- 기록량 구간별 오차 ----------

def mae_by_history(preds, table, window=WINDOW):
    """
    OOF 예측을 개인 과거 기록 수 구간으로 나눠 MAE를 냄.
    preds: pid | date | target | experiment | fold | y_true | y_pred (fe.run_cv / predictions.csv)
    반환: target | experiment | bin | n_nights | n_people | mae | person_mae_mean
    """
    p = preds.copy()
    p['date'] = pd.to_datetime(p['date'])
    parts = []
    for t, g in p.groupby('target', sort=False):
        cnt = _names(t, window)[1]
        g = g.merge(table[['pid', 'date', cnt]], on=['pid', 'date'], how='left', validate='many_to_one')
        g['bin'] = pd.cut(g[cnt], BIN_EDGES, labels=BIN_LABELS)
        g['err'] = (g['y_true'] - g['y_pred']).abs()
        for (e, b), h in g.groupby(['experiment', 'bin'], sort=False, observed=True):
            parts.append({'target': t, 'experiment': e, 'bin': str(b), 'n_nights': int(len(h)),
                          'n_people': int(h['pid'].nunique()), 'mae': float(h['err'].mean()),
                          'person_mae_mean': float(h.groupby('pid')['err'].mean().mean())})
    out = pd.DataFrame(parts)
    order = {b: i for i, b in enumerate(BIN_LABELS)}
    return out.sort_values(['target', 'experiment', 'bin'], key=lambda s: s.map(order) if s.name == 'bin' else s)\
              .reset_index(drop=True)
