"""
forecast_split.py — 참가자 기준 분할: 최종 평가 약 15% 고정 + 나머지 약 85%로 5-fold

    from src import forecast_split as fs
    split = fs.load_or_make_split(data.table, out_dir)   # 있으면 읽고 검사, 없으면 만들고 저장
    train_pids, valid_pids = fs.fold_pids(split, fold=1)

규칙 (같은 참가자 목록 + 같은 설정이면 언제나 같은 결과)
  1. 참가자 = 예측 표(forecast_dataset.table)에 한 밤이라도 있는 pid. 문자열로 바꾸고 중복 제거, 사전순 정렬
  2. n_final = floor(n × 0.15 + 0.5) (반올림, .5는 올림)
  3. np.random.default_rng(0).permutation(n)으로 섞은 순서의 앞 n_final명 = 최종 평가(role='final', fold=0)
  4. 나머지(role='dev')는 섞은 순서대로 i번째 사람이 fold = (i mod 5) + 1 → fold 1~5, 크기 차이 최대 1명
  매회: 학습 = dev 중 4 fold (전체의 약 68%), 검증 = dev 중 1 fold (약 17%), 최종 평가 약 15%는 안 씀.
  70/15/15 분할이나 다섯 번 독립 랜덤 분할이 아님.

분할표(split.csv)에는 원본 참가자 ID가 있으므로 results/ 아래에만 저장한다(.gitignore). 커밋하지 않음.
분할표 폴더는 데이터 프로토콜별로 나뉜다 (results/forecast/<protocol>/). 규칙이 바뀌면 새 폴더에 새로 만든다.
팀원끼리는 split_manifest.json의 split_sha256 / data_fingerprint를 비교해 같은지 확인한다.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
FINAL_FRAC = 0.15
N_FOLDS = 5
FOLDS = tuple(range(1, N_FOLDS + 1))
SPLIT_FILE = 'split.csv'
MANIFEST_FILE = 'split_manifest.json'


class SplitMismatch(RuntimeError):
    """저장된 분할표가 지금 데이터·설정과 다름. 조용히 다시 만들지 않음."""


def participants(table):
    """예측 표의 참가자 목록 (문자열, 중복 제거, 정렬)."""
    return sorted(set(table['pid'].astype(str)))


def n_final_for(n, final_frac=FINAL_FRAC):
    """최종 평가 인원 = floor(n × final_frac + 0.5)."""
    return int(np.floor(n * final_frac + 0.5))


def make_split(pids, seed=SEED, final_frac=FINAL_FRAC, n_folds=N_FOLDS):
    """참가자 목록 → pid | role('final'|'dev') | fold (final은 0, dev는 1..n_folds). pid 순 정렬."""
    ids = sorted(set(str(p) for p in pids))
    n = len(ids)
    n_final = n_final_for(n, final_frac)
    if n - n_final < n_folds:
        raise ValueError(f'개발 참가자 {n - n_final}명으로 {n_folds}-fold를 만들 수 없음')
    order = np.random.default_rng(seed).permutation(n)
    perm = [ids[i] for i in order]
    rows = [(p, 'final', 0) for p in perm[:n_final]]
    rows += [(p, 'dev', i % n_folds + 1) for i, p in enumerate(perm[n_final:])]
    return pd.DataFrame(rows, columns=['pid', 'role', 'fold']).sort_values('pid').reset_index(drop=True)


def check_split(split, n_folds=N_FOLDS):
    """분할 무결성 검사. 문제가 있으면 AssertionError."""
    assert split['pid'].is_unique, '한 참가자가 여러 줄에 있음'
    assert set(split['role']) <= {'final', 'dev'}, '알 수 없는 role'
    fin, dev = split[split['role'] == 'final'], split[split['role'] == 'dev']
    assert (fin['fold'] == 0).all(), '최종 평가 참가자에 fold 번호가 있음'
    assert set(dev['fold']) == set(range(1, n_folds + 1)), 'dev fold 번호가 1..n_folds가 아님'
    sizes = dev['fold'].value_counts()
    assert sizes.max() - sizes.min() <= 1, 'fold 크기 차이가 1명보다 큼'
    return True


def fold_pids(split, fold):
    """fold번 차례의 (학습 pid 집합, 검증 pid 집합). 최종 평가 참가자는 어느 쪽에도 없음."""
    dev = split[split['role'] == 'dev']
    if fold not in set(dev['fold']):
        raise ValueError(f'fold는 {sorted(set(dev["fold"]))} 중 하나')
    valid = set(dev.loc[dev['fold'] == fold, 'pid'])
    train = set(dev.loc[dev['fold'] != fold, 'pid'])
    return train, valid


def final_pids(split):
    return set(split.loc[split['role'] == 'final', 'pid'])


def _sha(lines):
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def split_hash(split):
    """분할표 내용의 sha256 (pid,role,fold를 pid 순으로)."""
    s = split.sort_values('pid')
    return _sha(f'{p},{r},{int(f)}' for p, r, f in zip(s['pid'], s['role'], s['fold']))


def _settings(seed, final_frac, n_folds):
    return {'seed': seed, 'final_frac': final_frac, 'n_folds': n_folds,
            'rounding': 'n_final = floor(n * final_frac + 0.5)',
            'fold_rule': 'dev i-th in permuted order -> fold (i mod n_folds) + 1',
            'rng': 'numpy.random.default_rng(seed).permutation(n) over sorted str pids'}


def manifest_for(split, seed, final_frac, n_folds, data_fingerprint=None, protocol=None):
    dev = split[split['role'] == 'dev']
    return {**_settings(seed, final_frac, n_folds),
            'n_participants': int(len(split)), 'n_final': int((split['role'] == 'final').sum()),
            'n_dev': int(len(dev)),
            'fold_sizes': {str(k): int(v) for k, v in dev['fold'].value_counts().sort_index().items()},
            'pid_list_sha256': _sha(sorted(split['pid'])),
            'split_sha256': split_hash(split),
            'data_fingerprint': data_fingerprint, 'protocol': protocol}


def save_split(split, out_dir, manifest):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split.to_csv(out / SPLIT_FILE, index=False)
    manifest = {**manifest, 'created_at': datetime.now().isoformat(timespec='seconds')}
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def load_split(out_dir):
    out = Path(out_dir)
    split = pd.read_csv(out / SPLIT_FILE, dtype={'pid': str, 'role': str, 'fold': int})
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding='utf-8'))
    return split, manifest


def load_or_make_split(table, out_dir, seed=SEED, final_frac=FINAL_FRAC, n_folds=N_FOLDS,
                       data_fingerprint=None, protocol=None, rebuild=False):
    """
    out_dir에 분할표가 있으면 읽고, 지금 참가자 목록·설정·(주면) 데이터 지문과 같은지 검사.
    다르면 SplitMismatch (rebuild=True일 때만 새로 만들어 덮어씀). 없으면 새로 만들어 저장.
    반환: (split, manifest)
    """
    pids = participants(table)
    out = Path(out_dir)
    if (out / SPLIT_FILE).exists() and not rebuild:
        split, man = load_split(out)
        check_split(split, n_folds)
        problems = []
        if sorted(split['pid']) != pids:
            problems.append('참가자 목록이 다름')
        for k, v in _settings(seed, final_frac, n_folds).items():
            if man.get(k) != v:
                problems.append(f'설정 {k}: 저장 {man.get(k)!r} / 지금 {v!r}')
        if man.get('split_sha256') != split_hash(split):
            problems.append('split.csv가 manifest와 다름 (직접 수정됨?)')
        if data_fingerprint and man.get('data_fingerprint') not in (None, data_fingerprint):
            problems.append('데이터 지문이 다름 (원본 또는 예측 표 생성 규칙이 바뀜)')
        if protocol and man.get('protocol') != protocol:
            problems.append(f"프로토콜이 다름: 저장 {man.get('protocol')!r} / 지금 {protocol!r}")
        if not split.equals(make_split(pids, seed, final_frac, n_folds)):
            problems.append('같은 시드로 다시 만든 분할과 다름')
        if problems:
            raise SplitMismatch(f'{out / SPLIT_FILE}: ' + '; '.join(problems)
                                + ' — 의도한 변경이면 rebuild=True(--rebuild-split)로 다시 만드세요')
        return split, man
    split = make_split(pids, seed, final_frac, n_folds)
    check_split(split, n_folds)
    man = save_split(split, out, manifest_for(split, seed, final_frac, n_folds, data_fingerprint, protocol))
    return split, man
