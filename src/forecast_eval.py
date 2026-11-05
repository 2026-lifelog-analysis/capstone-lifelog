"""
forecast_eval.py — 공통 채점 + 5-fold 실행 (개발 참가자만)

    from src import forecast_eval as fe
    exps = [fe.EXPERIMENTS['catboost_base'], fe.Experiment('catboost_mine', fe.CatBoostModel, my_features)]
    scores, preds = fe.run_cv(table, split, exps)       # 최종 평가 참가자는 쓰지 않음
    fe.summarize(scores, preds)
    fe.compare(scores, 'catboost_base', 'catboost_mine')   # 기본 MAE − 개선 MAE (양수 = 개선)

채점 규칙
  - 목표마다 따로. 세 목표의 MAE를 평균 내 하나로 합치지 않음. 단위는 모두 분
  - 평가 밤(eval_nights) = 검증 참가자 중 그 목표의 정답이 있는 모든 밤. 모델·입력과 무관하게 고정.
    입력이 NaN이라는 이유로 빼지 않음.
  - 모델이 밤을 빠뜨리거나 NaN·±inf를 내면 그 fold는 '불완전'(mae=NaN). 남은 밤만으로 점수를 내지 않음
  - eval_keys_sha256 = 그 목표·fold의 평가 (pid, date) 목록 해시. 비교할 때 밤 수가 아니라 이 해시로 같은 밤인지 확인
  - fold MAE   = 그 fold 평가 밤 전체의 |정답 − 예측| 평균 (밤 단위)
  - fold 평균·표준편차 = 다섯 fold MAE의 평균, 표본 표준편차(ddof=1)
  - OOF MAE    = 다섯 fold 검증 예측을 모두 합친 밤 단위 평균. fold 크기가 달라 fold 평균과 다를 수 있음
  - 사람별 MAE 평균 = 사람마다 MAE를 낸 뒤 평균 (보조)
  - 요약·비교는 목표마다 fold 1..5가 정확히 한 번씩, 모두 완전할 때만 숫자를 냄 (status='complete').
    fold가 빠졌으면 'partial', 불완전 fold가 있으면 'incomplete'로 표시하고 평균·개선 fold 수를 비움
  - 비교 차이 = 기본 MAE − 개선 MAE. 양수면 개선. 다섯 fold 중 개선된 fold 수도 같이 봄
    (5개 fold에서 개선됐다고 통계적으로 유의하다는 뜻은 아님)

모델 인터페이스 (fold·목표마다 새로 만들어 처음부터 학습. 저장된 모델을 이어서 학습하지 않음)
    m = make_model()
    m.fit(train_df, y_col, features)     # train_df: 학습 참가자의 정답 있는 밤
    m.predict(valid_df) → 길이 len(valid_df) 배열   # valid_df: 키 + forecast_at + 입력 열만 (정답 없음)
  집단 통계·결측 대체·스케일러처럼 학습이 필요한 것은 fit 안에서 train_df로만 맞춤.
"""
import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import forecast_dataset as fd
from . import forecast_split as fs

KEY = ['pid', 'date']
ALPHA = 1.0
SEED = 0

# CatBoost 고정 설정 (튜닝하지 않음. 바꾸려면 CatBoostModel(**params)로 실험 이름을 따로 붙일 것)
CATBOOST_PARAMS = {
    'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'loss_function': 'RMSE',
    'nan_mode': 'Min',          # 숫자 입력의 NaN은 채우지 않고 CatBoost가 '가장 작은 값' 쪽 가지로 보냄
    'one_hot_max_size': 7,      # dow(7개 값)는 CatBoost 안에서 원-핫. 정답 기반 통계(CTR)를 쓰지 않음
    'random_seed': SEED, 'thread_count': 4, 'verbose': False, 'allow_writing_files': False,
}


# ---------- 모델 ----------

def _lag_cols(features):
    return [c for c in features if '_lag' in c]


def _split_cols(features):
    cat = [c for c in features if c in fd.CATEGORICAL_FEATURES]
    return [c for c in features if c not in cat], cat


class GroupMedian:
    """학습 참가자 정답의 중앙값 하나 (입력 안 씀)."""
    def fit(self, train, y, features):
        self.value_ = float(train[y].median())
        return self

    def predict(self, df):
        return np.full(len(df), self.value_)


class PersonalMean:
    """
    그 사람의 과거 7밤 값(입력 중 *_lag 열) 평균. 7밤이 모두 NaN이면 학습 참가자 정답 중앙값.
    (그 사람의 예측 시점 이전 기록만 쓰므로 검증 참가자의 과거 기록을 입력으로 쓰는 것은 허용됨)
    """
    def fit(self, train, y, features):
        self.lags_ = _lag_cols(features)
        if not self.lags_:
            raise ValueError('PersonalMean에는 *_lag 입력이 필요함')
        self.fallback_ = float(train[y].median())
        return self

    def predict(self, df):
        return df[self.lags_].mean(axis=1).fillna(self.fallback_).to_numpy()


class RidgeModel:
    """
    Ridge 회귀. 숫자 입력 = 중앙값 대체(+결측 표시) → 표준화, dow = 원-핫.
    대체값·스케일러·계수는 모두 train_df로만 맞춤.
    """
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha

    def fit(self, train, y, features):
        self.features_ = list(features)
        num, cat = _split_cols(self.features_)
        tr = []
        if num:
            tr.append(('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                                          keep_empty_features=True),
                                            StandardScaler()), num))
        if cat:
            tr.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat))
        self.model_ = make_pipeline(ColumnTransformer(tr), Ridge(alpha=self.alpha))
        self.model_.fit(train[self.features_], train[y])
        return self

    def predict(self, df):
        return self.model_.predict(df[self.features_])


class CatBoostModel:
    """
    CatBoost 회귀 어댑터 (pip install catboost).
      - 숫자 입력: NaN을 채우지 않고 그대로 넘김 (nan_mode='Min')
      - dow: 범주형(cat_features)으로 넘김 (정수 0~6 → 문자열). one_hot_max_size=7이라 원-핫으로 처리됨
      - 경계값(quantization)·트리는 train_df로만 만듦. 검증 데이터를 eval_set·조기 종료에 쓰지 않음
      - random_seed 고정 (CATBOOST_PARAMS). fold·목표마다 run_cv가 새 객체를 만든다
    """
    def __init__(self, **params):
        self.params = {**CATBOOST_PARAMS, **params}

    def _X(self, df):
        X = df[self.features_].copy()
        for c in self.cat_:
            X[c] = X[c].astype('Int64').astype(str)
        return X

    def fit(self, train, y, features):
        from catboost import CatBoostRegressor
        self.features_ = list(features)
        self.num_, self.cat_ = _split_cols(self.features_)
        self.model_ = CatBoostRegressor(**self.params, cat_features=self.cat_ or None)
        self.model_.fit(self._X(train), train[y].to_numpy())
        return self

    def predict(self, df):
        return self.model_.predict(self._X(df))


@dataclass
class Experiment:
    """
    name       : 결과에 남는 이름
    make_model : 인자 없이 부르면 새 모델을 돌려주는 것 (클래스나 lambda)
    features   : 목표 이름 → 입력 열 목록을 돌려주는 함수. 기본 = fd.base_features (목표별 고정 기본 입력)
    """
    name: str
    make_model: Callable
    features: Callable = fd.base_features


# 명령줄 --experiments로 고를 수 있는 기본 실험 (모두 목표별 기본 입력 fd.base_features)
EXPERIMENTS = {
    'group_median': Experiment('group_median', GroupMedian),
    'personal_mean7': Experiment('personal_mean7', PersonalMean),
    'ridge_base': Experiment('ridge_base', RidgeModel),
    'catboost_base': Experiment('catboost_base', CatBoostModel),
}


# ---------- 채점 ----------

def keys_hash(df):
    """(pid, date) 목록의 sha256 (정렬 후)."""
    k = df[KEY].sort_values(KEY)
    lines = (k['pid'].astype(str) + ',' + pd.to_datetime(k['date']).dt.strftime('%Y-%m-%d'))
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def eval_nights(table, split, target):
    """
    목표별 공통 평가 밤: 개발 참가자 중 그 목표 정답이 있는 밤 전부.
    반환: pid | date | fold | y_true (pid, date 순). 모델과 무관.
    """
    y = fd.TARGETS[target]
    dev = split[split['role'] == 'dev'][['pid', 'fold']]
    ev = table[['pid', 'date', y]].merge(dev, on='pid', how='inner')
    ev = ev[ev[y].notna()].rename(columns={y: 'y_true'})
    if ev.duplicated(KEY).any():
        raise ValueError('예측 표에 (pid, date) 중복이 있음')
    return ev[['pid', 'date', 'fold', 'y_true']].sort_values(KEY).reset_index(drop=True)


def score(pred, ev):
    """
    pred: pid | date | y_pred,  ev: eval_nights 결과(한 fold 또는 전체)
    키 중복·평가 밤 밖의 예측은 오류. 빠진 밤·NaN·±inf 예측이 있으면 complete=False, mae=NaN.
    반환: dict(mae, person_mae_mean, n_people, n_nights, n_missing, n_nonfinite, complete, eval_keys_sha256, note)
    """
    if pred.duplicated(KEY).any():
        raise ValueError('예측에 (pid, date) 중복이 있음')
    m = ev.merge(pred[KEY + ['y_pred']], on=KEY, how='outer', indicator=True)
    extra = int((m['_merge'] == 'right_only').sum())
    if extra:
        raise ValueError(f'평가 밤 목록에 없는 예측 {extra}개 (다른 참가자·밤과 섞였음)')
    n_missing = int((m['_merge'] == 'left_only').sum())
    yp = pd.to_numeric(m['y_pred'], errors='coerce').to_numpy(dtype=float)
    n_bad = int(((m['_merge'] == 'both').to_numpy() & ~np.isfinite(yp)).sum())
    complete = n_missing == 0 and n_bad == 0
    err = pd.Series(np.abs(m['y_true'].to_numpy(dtype=float) - yp), index=m.index)
    note = '' if complete else f'불완전: 예측 없음 {n_missing}밤, NaN/inf 예측 {n_bad}밤'
    return {'mae': float(err.mean()) if complete else np.nan,
            'person_mae_mean': float(err.groupby(m['pid']).mean().mean()) if complete else np.nan,
            'n_people': int(ev['pid'].nunique()), 'n_nights': int(len(ev)),
            'n_missing': n_missing, 'n_nonfinite': n_bad, 'complete': complete,
            'eval_keys_sha256': keys_hash(ev), 'note': note}


# ---------- 실행 ----------

def run_cv(table, split, experiments, targets=None, folds=fs.FOLDS):
    """
    개발 참가자 5-fold. fold마다 학습 4 fold / 검증 1 fold, 모델은 매번 새로 만들어 학습.
    최종 평가 참가자(role='final')는 학습·검증·예측 어디에도 쓰지 않음.
    folds를 일부만 주면 결과는 부분 실행(summarize/compare에서 'partial')으로 표시됨.
    반환: (scores, preds)
      scores : target | experiment | fold | mae | person_mae_mean | n_people | n_nights | n_missing |
               n_nonfinite | complete | eval_keys_sha256 | note
      preds  : pid | date | target | experiment | fold | y_true | y_pred
    """
    targets = list(targets or fd.TARGETS)
    names = [e.name for e in experiments]
    if len(set(names)) != len(names):
        raise ValueError('실험 이름이 중복됨')
    fs.check_split(split)
    final = fs.final_pids(split)
    score_rows, pred_parts = [], []
    for target in targets:
        y = fd.TARGETS[target]
        ev_all = eval_nights(table, split, target)
        for fold in folds:
            train_p, valid_p = fs.fold_pids(split, fold)
            assert not (train_p | valid_p) & final and not train_p & valid_p
            train = table[table['pid'].isin(train_p) & table[y].notna()]
            ev = ev_all[ev_all['fold'] == fold]
            valid = ev[KEY].merge(table, on=KEY, how='left')        # 평가 밤 그대로, 순서 고정
            for e in experiments:
                feats = fd.check_features(e.features(target), table)
                model = e.make_model()
                model.fit(train, y, feats)
                # 검증 쪽에는 정답 열을 넘기지 않음 (키 + 예측 기준 시각 + 입력만)
                p = np.asarray(model.predict(valid[KEY + fd.META_COLS + feats]), dtype=float)
                if p.shape != (len(valid),):
                    raise ValueError(f'{e.name}: 예측 길이 {p.shape} ≠ 평가 밤 {len(valid)}')
                pred = valid[KEY].assign(y_pred=p)
                s = score(pred, ev)
                score_rows.append({'target': target, 'experiment': e.name, 'fold': fold, **s})
                pred_parts.append(ev.merge(pred, on=KEY, how='left')
                                    .assign(target=target, experiment=e.name)
                                  [['pid', 'date', 'target', 'experiment', 'fold', 'y_true', 'y_pred']])
    scores = pd.DataFrame(score_rows)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return scores, preds


def fold_status(g, folds=fs.FOLDS):
    """한 목표·한 실험의 scores 줄 → ('complete'|'partial'|'incomplete'|'duplicate', 설명)."""
    got = list(g['fold'])
    if len(got) != len(set(got)):
        return 'duplicate', f'fold 중복 {sorted(got)}'
    missing = sorted(set(folds) - set(got))
    if missing or set(got) - set(folds):
        return 'partial', f'부분 실행: fold {sorted(got)} (빠진 fold {missing})'
    bad = [f"fold {r.fold}: {r.note}" for r in g.itertuples() if not r.complete]
    if bad:
        return 'incomplete', '; '.join(bad)
    return 'complete', ''


def summarize(scores, preds, folds=fs.FOLDS):
    """
    목표 × 실험 요약. status가 'complete'(fold 1..5가 한 번씩, 모두 완전)일 때만 숫자를 채움.
      fold_mae_mean, fold_mae_std(ddof=1) : 다섯 fold MAE의 평균·표본 표준편차
      oof_mae          : 다섯 fold 검증 예측을 합친 밤 단위 MAE
      oof_person_mae_mean : 사람별 OOF MAE의 평균 (보조)
      n_people, n_nights : OOF 평가 참가자·밤 수
    """
    rows = []
    for (t, e), g in scores.groupby(['target', 'experiment'], sort=False):
        p = preds[(preds['target'] == t) & (preds['experiment'] == e)]
        status, note = fold_status(g, folds)
        ok = status == 'complete'
        err = (p['y_true'] - p['y_pred']).abs()
        rows.append({'target': t, 'experiment': e, 'status': status, 'n_folds': len(g),
                     'fold_mae_mean': g['mae'].mean() if ok else np.nan,
                     'fold_mae_std': g['mae'].std(ddof=1) if ok else np.nan,
                     'oof_mae': err.mean() if ok else np.nan,
                     'oof_person_mae_mean': err.groupby(p['pid']).mean().mean() if ok else np.nan,
                     'n_people': int(p['pid'].nunique()), 'n_nights': int(len(p)),
                     'complete': ok, 'note': note})
    return pd.DataFrame(rows)


def _true(x):
    return isinstance(x, (bool, np.bool_)) and bool(x)


def compare(scores, base, new, folds=fs.FOLDS):
    """
    같은 목표·같은 fold에서 기본 MAE − 개선 MAE (양수 = 개선).
    반환: target | fold | base_mae | new_mae | diff | n_improved | status | note
      fold별 줄 + 목표별 요약 줄(fold='mean'). 요약 줄은 두 실험 모두 status='complete'일 때만 숫자를 냄.
    같은 목표·fold의 평가 키 해시(eval_keys_sha256)가 다르면 오류 (밤 수가 같아도).
    """
    a_all, b_all = scores[scores['experiment'] == base], scores[scores['experiment'] == new]
    if a_all.empty or b_all.empty:
        raise KeyError('실험 이름을 확인하세요')
    out = []
    for t in dict.fromkeys(list(a_all['target']) + list(b_all['target'])):
        a, b = a_all[a_all['target'] == t], b_all[b_all['target'] == t]
        sa, na = fold_status(a, folds) if len(a) else ('partial', '기본 실험에 이 목표 없음')
        sb, nb = fold_status(b, folds) if len(b) else ('partial', '개선 실험에 이 목표 없음')
        m = a[['fold', 'mae', 'complete', 'eval_keys_sha256']].merge(
            b[['fold', 'mae', 'complete', 'eval_keys_sha256']], on='fold', how='outer', suffixes=('_a', '_b'))
        both = m['eval_keys_sha256_a'].notna() & m['eval_keys_sha256_b'].notna()
        if (m.loc[both, 'eval_keys_sha256_a'] != m.loc[both, 'eval_keys_sha256_b']).any():
            raise ValueError(f'{t}: 같은 fold인데 평가 밤(pid, date) 목록이 다름 — 같은 평가 밤으로 채점해야 함')
        for r in m.sort_values('fold').itertuples():
            ok = _true(r.complete_a) and _true(r.complete_b)
            out.append({'target': t, 'fold': r.fold, 'base_mae': r.mae_a, 'new_mae': r.mae_b,
                        'diff': r.mae_a - r.mae_b if ok else np.nan, 'n_improved': np.nan,
                        'status': 'complete' if ok else 'incomplete', 'note': ''})
        if sa == 'complete' and sb == 'complete':
            d = m['mae_a'] - m['mae_b']
            out.append({'target': t, 'fold': 'mean', 'base_mae': m['mae_a'].mean(), 'new_mae': m['mae_b'].mean(),
                        'diff': d.mean(), 'n_improved': int((d > 0).sum()), 'status': 'complete', 'note': ''})
        else:
            st = 'partial' if 'partial' in (sa, sb) else 'incomplete'
            out.append({'target': t, 'fold': 'mean', 'base_mae': np.nan, 'new_mae': np.nan, 'diff': np.nan,
                        'n_improved': np.nan, 'status': st,
                        'note': f'{base}: {sa} {na} | {new}: {sb} {nb}'.strip()})
    return pd.DataFrame(out)
