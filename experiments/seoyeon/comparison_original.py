"""고정 승호 커밋의 원설정 12조건을 기존 출력 파일 변경 없이 재실행합니다.

동일 주입 입력의 점수는 경보율 간에만 재사용합니다. 점수식은 원 함수를
직접 호출하며 원 평가기의 전체 기간 분위수 q와 내부 CUSUM h를 보존합니다.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from experiments.seoyeon.audit import REFERENCE, ROOT, reference_modules
from src import baseline as bl, data_loader as dl, evaluation as ev, preprocessing as pp

_ACTIVE = False


@contextmanager
def reference_context():
    """바이트 검증된 원본을 읽고 import·주입 함수의 전역 상태를 복원합니다."""
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('A nested reference context is not supported.')
    _ACTIVE = True
    original_inject, original_path = ev.inject_drift, list(sys.path)
    names = ['common', 'final', 'seoyeon_reference_rule']
    original_modules = {name: sys.modules.get(name) for name in names}
    try:
        yield reference_modules()
    finally:
        ev.inject_drift = original_inject
        sys.path[:] = original_path
        for name, value in original_modules.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        _ACTIVE = False


class OriginalScoreCache:
    """경보율과 무관한 원본 점수를 최근 24개 입력에 한해 재사용합니다."""

    def __init__(self, final, reference):
        self.final, self.reference = final, reference
        self._cache = OrderedDict()
        self.computed_inputs = 0
        self.hits = 0

    def components(self, data):
        hashes = pd.util.hash_pandas_object(data[['pid', 'date', 'steps']], index=True)
        key = hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        parts = {'무작위': np.random.default_rng(ev.SEED).random(len(data)),
                 '집단 기준선': self.final.group_bad(data, self.reference)}
        unique = {}
        for method, (dow, anchor, _) in self.final.STEPS.items():
            if (dow, anchor) not in unique:
                unique[dow, anchor] = self.final.personal_bad(data, self.reference, dow, anchor)
            parts[method] = unique[dow, anchor]
        value = pd.DataFrame(parts, index=data.index)
        self._cache[key] = value
        self.computed_inputs += 1
        if len(self._cache) > 24:
            self._cache.popitem(last=False)
        return value

    def score_fn(self, clean, rate):
        parts = self.components(clean)
        hs = {method: bl.cusum_threshold(clean.assign(_b=parts[method]), '_b', k=k, rate=rate)
              for method, (_, _, k) in self.final.STEPS.items()}

        def score(data):
            p = self.components(data)
            columns = {name: p[name] for name in ['무작위', '집단 기준선']}
            for method, (_, _, k) in self.final.STEPS.items():
                columns[method] = bl.cusum(data.assign(_b=p[method]), '_b', k=k, h=hs[method])
            return pd.DataFrame(columns, index=data.index)[self.final.ORDER]

        score.h = hs
        return score


def run(output, input_path=None, n_reps=20):
    """원 S1·S3·S5 × 네 경보율, 20회 반복. 개인 행은 지정 output만 사용."""
    if n_reps < 1:
        raise ValueError('n_reps must be positive.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = (pd.read_csv(input_path, parse_dates=['date'], dtype={'pid': str}, float_precision='round_trip')
            if input_path else dl.load_nethealth())
    reference_folder = ROOT / '_archive' / 'seungho_1806f0a' / 'experiments'
    manifest = dict(status='running', reference_commit=REFERENCE, n_reps=n_reps,
                    python=platform.python_version(), data_rows=len(data),
                    people=int(data.pid.nunique()), tasks_completed=[],
                    input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest() if input_path else None,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    reference_sha256={name: hashlib.sha256((reference_folder / name).read_bytes()).hexdigest()
                                      for name in ['common.py', 'final.py', 'rule_baseline.py']},
                    versions={name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn']},
                    command=sys.argv,
                    evaluation='Original participant 70/30; original h and quantile q; not future-time test')
    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    save_manifest()
    try:
        with reference_context() as (common, final, _):
            dev, hold = common.split_pids(data, frac=.7, seed=0)
            reference = data[data.pid.isin(dev)].reset_index(drop=True)
            held = data[data.pid.isin(hold)].reset_index(drop=True)
            manifest.update(dev_people=len(dev), holdout_people=len(hold), holdout_rows=len(held))
            cache = OriginalScoreCache(final, reference)
            scorers = {name: cache.score_fn(held, rate) for name, rate in final.RATES.items()}
            rows, comparisons = [], []
            for scenario, (effects, shape, _) in final.SCEN.items():
                (col, effect, mode), *extra = effects
                common.INJECT_EXTRA[:] = extra
                for rate_name, rate in final.RATES.items():
                    tick = time.perf_counter()
                    result = ev.evaluate_injection(held, col, effect, mode=mode, shape=shape,
                                                   n_reps=n_reps, rate=rate,
                                                   score_fn=scorers[rate_name])
                    label = f'{scenario}_{rate_name.replace("/", "in")}'
                    result['per_person'].to_csv(output / f'{label}_per_person.csv.gz', index=False)
                    result['segments'].to_csv(output / f'{label}_segments.csv.gz', index=False)
                    info = {key: value for key, value in result.items()
                            if key not in ['per_person', 'segments']}
                    info['reset_h'] = scorers[rate_name].h
                    (output / f'{label}_settings.json').write_text(
                        json.dumps(info, ensure_ascii=False, indent=2))
                    rows.append(common.summary(result, '').drop(columns='실험').assign(
                        시나리오=scenario, 하루경보=rate_name))
                    for method in final.ORDER:
                        if method != final.BASE:
                            comparisons.append({'시나리오': scenario, '하루경보': rate_name,
                                '방법': method, '비교 대상': final.BASE,
                                **common.compare_methods(result, final.BASE, method)})
                    pd.concat(rows, ignore_index=True).to_csv(output / 'final_table.csv', index=False)
                    pd.DataFrame(comparisons).to_csv(output / 'final_compare.csv', index=False)
                    elapsed = time.perf_counter() - tick
                    manifest['tasks_completed'].append(dict(scenario=scenario, rate=rate_name, seconds=elapsed))
                    save_manifest()
                    print(f'{scenario} {rate_name}: completed {n_reps} repetitions, {elapsed:.1f}s', flush=True)
            manifest.update(status='passed', cache_computed_inputs=cache.computed_inputs,
                            cache_hits=cache.hits)
            common._CACHE[pp.WEAR_MIN] = data
            final.calendar().to_csv(output / 'final_calendar.csv', index=False)
            if input_path and hashlib.sha256(Path(input_path).read_bytes()).hexdigest() != manifest['input_sha256']:
                raise RuntimeError('The input snapshot changed during execution.')
            manifest['seconds'] = time.perf_counter() - start
    except BaseException as error:
        manifest.update(status='failed', error_type=type(error).__name__, seconds=time.perf_counter() - start)
        raise
    finally:
        save_manifest()
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--repetitions', type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.input, args.repetitions)
