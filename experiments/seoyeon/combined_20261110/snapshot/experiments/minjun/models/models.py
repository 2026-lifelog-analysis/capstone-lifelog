"""
② 모델 구조 비교 — 모델 정의 (Ridge, CatBoost, MLP, FT-Transformer)

모든 모델은 같은 형식을 따른다.
    m = Model(config, seed)
    m.fit(train_df, y_train, valid_df, y_valid, features)   # valid = 학습 참가자 안에서 뗀 10% (조기 종료·설정 선택용)
    m.predict(df) -> np.ndarray
전처리(결측 채움값, 표준화, 정답 표준화)는 모두 train_df로만 맞춘다. 검증 fold 참가자는 여기에 들어오지 않는다.

  Ridge          : 공통 fe.RidgeModel과 같은 전처리 (중앙값 채움 + 결측 표시 + 표준화, 요일 원-핫)
  CatBoost       : 공통 fe.CATBOOST_PARAMS 기반 (결측 그대로, 요일 범주형). depth·learning_rate만 바꾸고 조기 종료
  MLP            : Ridge와 같은 전처리 → ReLU 은닉층 + dropout, AdamW, MSE, 조기 종료
  FT-Transformer : 숫자 열마다 토큰(결측이면 열별 '결측' 토큰), 요일 임베딩 토큰, CLS 토큰 → Transformer 블록
                   (Gorishniy et al. 2021, "Revisiting Deep Learning Models for Tabular Data"를 간단히 구현)
"""
import time

import numpy as np
import pandas as pd

from src import forecast_dataset as fd
from src import forecast_eval as fe

CAT = 'dow'


def _num_cols(features):
    return [c for c in features if c != CAT]


class Prep:
    """숫자 열: train 중앙값으로 채움 + 결측 표시, train 평균·표준편차로 표준화. 요일: 0~6 정수."""

    def fit(self, df, features):
        self.num = _num_cols(features)
        X = df[self.num].to_numpy(dtype=np.float64)
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        Xf = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = Xf.mean(0), Xf.std(0)
        self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)
        return self

    def transform(self, df):
        X = df[self.num].to_numpy(dtype=np.float64)
        miss = np.isnan(X)
        Xs = (np.where(miss, self.med, X) - self.mu) / self.sd
        dow = df[CAT].to_numpy(dtype=np.int64)
        return Xs.astype(np.float32), miss, dow


# ---------- Ridge ----------

class RidgeM:
    GRID = [{'alpha': a} for a in (1.0, 100.0, 10000.0)]

    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def fit(self, tr, ytr, va, yva, features):
        self.m = fe.RidgeModel(alpha=self.config['alpha'])
        self.m.fit(tr.assign(_y=ytr), '_y', features)
        self.info = {}
        return self

    def predict(self, df):
        return np.asarray(self.m.predict(df), dtype=float)


# ---------- CatBoost ----------

class CatBoostM:
    GRID = [{'depth': 6, 'learning_rate': 0.1}, {'depth': 8, 'learning_rate': 0.1}, {'depth': 8, 'learning_rate': 0.05}]
    MAX_ITER, OD_WAIT = 3000, 100

    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def fit(self, tr, ytr, va, yva, features):
        from catboost import CatBoostRegressor
        self.features = list(features)
        self.cat = [c for c in self.features if c in fd.CATEGORICAL_FEATURES]
        params = {**fe.CATBOOST_PARAMS, **self.config, 'iterations': self.MAX_ITER, 'random_seed': self.seed,
                  'od_type': 'Iter', 'od_wait': self.OD_WAIT, 'use_best_model': True}
        self.m = CatBoostRegressor(**params, cat_features=self.cat or None)
        self.m.fit(self._X(tr), np.asarray(ytr, float), eval_set=(self._X(va), np.asarray(yva, float)))
        self.info = {'best_iteration': int(self.m.get_best_iteration())}
        return self

    def _X(self, df):
        X = df[self.features].copy()
        for c in self.cat:
            X[c] = X[c].astype('Int64').astype(str)
        return X

    def predict(self, df):
        return np.asarray(self.m.predict(self._X(df)), dtype=float)


# ---------- 신경망 공통 ----------

class _TorchM:
    MAX_EPOCHS, PATIENCE, BATCH, WD = 100, 5, 1024, 1e-5

    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def _net(self, n_num):
        raise NotImplementedError

    def _tensors(self, df):
        import torch
        Xs, miss, dow = self.prep.transform(df)
        dev = self.dev
        return (torch.from_numpy(Xs).to(dev), torch.from_numpy(miss).to(dev), torch.from_numpy(dow).to(dev))

    def fit(self, tr, ytr, va, yva, features):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prep = Prep().fit(tr, features)
        ytr, yva = np.asarray(ytr, float), np.asarray(yva, float)
        self.ymu, self.ysd = ytr.mean(), ytr.std()
        Ttr, Tva = self._tensors(tr), self._tensors(va)
        y = torch.from_numpy(((ytr - self.ymu) / self.ysd).astype(np.float32)).to(self.dev)
        self.net = self._net(len(self.prep.num)).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.config['lr'], weight_decay=self.WD)
        g = torch.Generator(device='cpu').manual_seed(self.seed)
        n = len(ytr)
        best, best_state, bad, best_ep = np.inf, None, 0, 0
        for ep in range(self.MAX_EPOCHS):
            self.net.train()
            perm = torch.randperm(n, generator=g).to(self.dev)
            for i in range(0, n, self.BATCH):
                idx = perm[i:i + self.BATCH]
                loss = torch.nn.functional.mse_loss(self.net(*(t[idx] for t in Ttr)), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            mae = float(np.mean(np.abs(self._pred_t(Tva) - yva)))
            if mae < best - 1e-4:
                best, bad, best_ep = mae, 0, ep
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.info = {'best_epoch': best_ep + 1, 'epochs_run': ep + 1}
        return self

    def _pred_t(self, T, batch=8192):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(T[0]), batch):
                out.append(self.net(*(t[i:i + batch] for t in T)).float().cpu().numpy())
        return np.concatenate(out) * self.ysd + self.ymu

    def predict(self, df):
        return self._pred_t(self._tensors(df))


def _mlp_net(n_num, hidden, dropout):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 4)                   # 요일 (원-핫과 같은 역할의 작은 임베딩)
            layers, d = [], 2 * n_num + 4                   # 표준화 값 + 결측 표시 + 요일
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x, miss, dow):
            import torch
            z = torch.cat([x, miss.float(), self.emb(dow)], dim=1)
            return self.mlp(z).squeeze(1)
    return Net()


class MLPM(_TorchM):
    GRID = [{'hidden': [128, 128], 'dropout': 0.0, 'lr': 1e-3},
            {'hidden': [512, 512, 256], 'dropout': 0.2, 'lr': 1e-3},
            {'hidden': [1024, 512], 'dropout': 0.3, 'lr': 1e-3}]

    def _net(self, n_num):
        return _mlp_net(n_num, self.config['hidden'], self.config['dropout'])


def _ftt_net(n_num, d, n_blocks, dropout=0.1, n_heads=8):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(n_num, d) * 0.02)       # 숫자 열별 토큰: x * w_j + b_j
            self.b = nn.Parameter(torch.zeros(n_num, d))
            self.miss_tok = nn.Parameter(torch.randn(n_num, d) * 0.02)  # 결측이면 열별 '결측' 토큰
            self.dow = nn.Embedding(7, d)
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=dropout,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, x, miss, dow):
            tok = x.unsqueeze(2) * self.w + self.b
            tok = torch.where(miss.unsqueeze(2), self.miss_tok.expand_as(tok), tok)
            z = torch.cat([self.cls.expand(len(x), -1, -1), tok, self.dow(dow).unsqueeze(1)], dim=1)
            return self.head(self.blocks(z)[:, 0]).squeeze(1)
    return Net()


class FTTM(_TorchM):
    GRID = [{'d': 64, 'blocks': 3, 'lr': 3e-4},
            {'d': 128, 'blocks': 2, 'lr': 3e-4},
            {'d': 128, 'blocks': 3, 'lr': 1e-4}]

    def _net(self, n_num):
        return _ftt_net(n_num, self.config['d'], self.config['blocks'])


MODELS = {'ridge': RidgeM, 'catboost': CatBoostM, 'mlp': MLPM, 'ftt': FTTM}


def timed_fit(cls, config, seed, tr, ytr, va, yva, features):
    t0 = time.perf_counter()
    m = cls(config, seed).fit(tr, ytr, va, yva, features)
    return m, time.perf_counter() - t0


class CatBoostWideM(CatBoostM):
    """감사 반영: CatBoost가 탐색 범위 끝(깊이 8)을 주로 골라, 깊이 10과 l2 규제를 더한 넓은 탐색 (5개)."""
    GRID = CatBoostM.GRID + [{'depth': 10, 'learning_rate': 0.05},
                             {'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 10}]


MODELS['catboost_wide'] = CatBoostWideM

