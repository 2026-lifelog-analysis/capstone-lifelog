"""공통 구현 재현과 승호 실험의 경보선 정의를 실제 NetHealth에서 대조합니다.

개인 행은 results/에만 저장합니다. 지표는 무라벨 경보 분포이며 정확도가 아닙니다.
승호의 전체 탐색 또는 확인용 주입 실험을 재실행한 것으로 해석하지 않습니다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import baseline as bl, data_loader as dl, preprocessing as pp

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = '1806f0a5604ab08111a94d62c01ac874a2fcb91c'


def _divide(a, b):
    return float(a / b) if b else np.nan


def alarm_comparison(df, personal, group, rate):
    """공통 유한 점수 인일에서 각 분위수와 엄격한 초과로 경보를 대조합니다."""
    mask = np.isfinite(df[personal]) & np.isfinite(df[group])
    x = df.loc[mask, ['pid', personal, group]]
    a_threshold = x[personal].quantile(1 - rate)
    b_threshold = x[group].quantile(1 - rate)
    a, b = x[personal] > a_threshold, x[group] > b_threshold
    ca, cb = a.groupby(x['pid']).sum(), b.groupby(x['pid']).sum()
    top_n = max(1, int(len(ca) * .1))
    return dict(common_days=len(x), people=len(ca), target_rate=rate,
                personal_threshold=a_threshold, group_threshold=b_threshold,
                personal_alarm_days=int(a.sum()), group_alarm_days=int(b.sum()),
                personal_alarm_rate=_divide(a.sum(), len(x)),
                group_alarm_rate=_divide(b.sum(), len(x)),
                overlap_days=int((a & b).sum()),
                overlap_over_personal=_divide((a & b).sum(), a.sum()),
                personal_zero_people=int((ca == 0).sum()),
                group_zero_people=int((cb == 0).sum()), top_people=top_n,
                personal_top_share=_divide(ca.nlargest(top_n).sum(), ca.sum()),
                group_top_share=_divide(cb.nlargest(top_n).sum(), cb.sum()))


def threshold_comparison(scores, reset_h, rate):
    """CUSUM 자체 초기화선과 평가기의 분위수선을 동일 점수에 적용합니다."""
    s = scores[np.isfinite(scores)]
    q = s.quantile(1 - rate)
    a, b = s > reset_h, s > q
    return dict(valid_days=len(s), reset_h=float(reset_h), quantile=float(q),
                quantile_minus_h=float(q - reset_h),
                reset_h_alarm_days=int(a.sum()), quantile_alarm_days=int(b.sum()),
                disagreement_days=int((a != b).sum()),
                reset_h_alarm_rate=_divide(a.sum(), len(s)),
                quantile_alarm_rate=_divide(b.sum(), len(s)))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_manifest():
    pairs = [('activity.csv', 'FitbitActivity(1-30-20).csv'),
             ('sleep.csv', 'FitbitSleep(1-30-20).csv'), ('calendar.csv', 'Calendar.csv')]
    records = []
    for original, copied in pairs:
        source = ROOT.parent / '모듈1_NetHealth' / '00_원자료' / original
        target = ROOT / 'data' / 'nethealth' / copied
        record = dict(source=str(source), target=str(target),
                      source_sha256=_sha(source), target_sha256=_sha(target))
        assert record['source_sha256'] == record['target_sha256'], copied
        records.append(record)
    return records


def preparation(nh):
    a, s = dl.read_raw()
    dedup = pp.drop_duplicate_days(a)
    act, sleep = pp.build_activity(a), pp.merge_sleep(s)
    union = act.merge(sleep, on=['pid', 'date'], how='outer', validate='one_to_one')
    rows = []
    for name, frame in [('raw_activity', a), ('raw_sleep', s),
                        ('activity_dedup', dedup), ('sleep_days', sleep),
                        ('outer_union', union), ('after_min42', nh)]:
        id_col = 'pid' if 'pid' in frame else 'egoid'
        rows.append(dict(stage=name, rows=len(frame), people=frame[id_col].nunique()))
    counts = pd.DataFrame(rows)
    quality = dict(low_wear_days=int((act['complypercent'] < 80).sum()),
                   missing_wear_days=int(act['complypercent'].isna().sum()),
                   retained_days_with_no_steps_or_sleep=int(
                       nh[['steps', 'sleep_min']].isna().all(axis=1).sum()),
                   all_three_valid_days=int(nh[['steps', 'sleep_min', 'bed_min']].notna().all(axis=1).sum()),
                   first_date=str(nh['date'].min().date()), last_date=str(nh['date'].max().date()),
                   duplicate_person_dates=int(nh.duplicated(['pid', 'date']).sum()))
    return counts, quality


def common_comparisons(nh):
    records = []
    for feature in ['steps', 'sleep_min', 'bed_min']:
        group = bl.bad_score(bl.group_z(nh, feature), feature)
        for dow in [False, True]:
            personal = bl.bad_score(bl.personal_z(nh, feature, dow_adjust=dow), feature)
            for rate in [1 / 30, .05]:
                r = alarm_comparison(nh.assign(p=personal, g=group), 'p', 'g', rate)
                records.append(dict(feature=feature, dow_adjust=dow, **r))
    return pd.DataFrame(records)


def reference_modules():
    """검토한 고정 커밋의 보관 사본만 읽고 Git 객체와 바이트를 대조합니다."""
    folder = ROOT / '_archive' / 'seungho_1806f0a' / 'experiments'
    for name in ['common.py', 'final.py', 'rule_baseline.py']:
        expected = subprocess.check_output(['git', 'show', f'{REFERENCE}:experiments/{name}'], cwd=ROOT)
        if (folder / name).read_bytes() != expected:
            raise ValueError(f'Reference bytes differ: {name}')
    sys.path.insert(0, str(folder))
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, folder / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    common = load('common', 'common.py')
    final = load('final', 'final.py')
    rule = load('seoyeon_reference_rule', 'rule_baseline.py')
    return common, final, rule


def reference_audit(nh):
    """확인용 집단의 원자료와 seed 0 단일 주입으로 판정선 차이만 진단합니다."""
    from src import evaluation as ev
    original_inject = ev.inject_drift
    saved_modules = {key: sys.modules.get(key) for key in ['common', 'final', 'seoyeon_reference_rule']}
    original_path = list(sys.path)
    try:
        common, final, rule = reference_modules()
        dev_ids, hold_ids = common.split_pids(nh)
        hold = nh[nh['pid'].isin(hold_ids)].reset_index(drop=True)
        ref = nh[nh['pid'].isin(dev_ids)].reset_index(drop=True)
        fn = final.make_score_fn(hold, ref, 1 / 30)
        clean = fn(hold)
        injected, segments = original_inject(hold, 'steps', -.30, seed=0)
        changed = fn(injected)
        segment_rows = ev._segment_rows(hold, segments).index
        records = []
        for method, h in fn.h.items():
            r = threshold_comparison(clean[method], h, 1 / 30)
            q = r['quantile']
            z = changed.loc[segment_rows, method]
            valid = np.isfinite(z)
            r.update(method=method, injected_valid_days=int(valid.sum()),
                     injected_disagreement_days=int(((z[valid] > h) != (z[valid] > q)).sum()))
            records.append(r)
        scores = rule.rule_score(hold).dropna()
        rules = []
        for drop in rule.DROPS:
            fixed = scores >= drop
            rate = float(fixed.mean())
            q = scores.quantile(1 - rate)
            remapped = scores > q
            rules.append(dict(drop=drop, quantile=q, valid_days=len(scores),
                              fixed_alarm_days=int(fixed.sum()),
                              remapped_alarm_days=int(remapped.sum()),
                              disagreement_days=int((fixed != remapped).sum())))
        info = dict(commit=REFERENCE, dev_people=len(dev_ids), holdout_people=len(hold_ids),
                    holdout_rows=len(hold), development_rows=len(ref),
                    diagnostic_injection='steps -30%, sudden, 14 days, seed 0, one replicate',
                    injected_people=len(segments), full_search_reproduced=False,
                    holdout_is_future_test=False)
        return pd.DataFrame(records), pd.DataFrame(rules), info
    finally:
        ev.inject_drift = original_inject
        sys.path[:] = original_path
        for key, value in saved_modules.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def write_graph(table, output):
    import matplotlib.pyplot as plt
    focus = table[(table['target_rate'] == 1 / 30) & ~table['dow_adjust']]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(focus))
    for ax, metric, title in zip(axes, ['zero_people', 'top_share', 'alarm_rate'],
                                ['People with zero alarms', 'Top 10% alarm share', 'Observed alarm rate']):
        for side, label, offset in [('personal', 'Personal', -.18), ('group', 'Group', .18)]:
            ax.bar(x + offset, focus[f'{side}_{metric}'], .36, label=label)
        ax.set_xticks(x, focus['feature'], rotation=20)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend()
    fig.suptitle('Same finite person-days; target 1/30; no day-of-week adjustment')
    fig.tight_layout()
    fig.savefig(output / 'alarm_comparison.png', dpi=180)
    fig.savefig(output / 'alarm_comparison.pdf')
    plt.close(fig)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = ROOT.parent / '모듈1_NetHealth' / '결과보고서.docx'
    preserved_before = _sha(report)
    inputs = input_manifest()
    nh = dl.load_nethealth()
    counts, quality = preparation(nh)
    comparisons = common_comparisons(nh)
    thresholds, rules, reference = reference_audit(nh)
    counts.to_csv(output / 'preprocessing_counts.csv', index=False)
    nh.to_csv(output / 'daily_preprocessed.csv.gz', index=False)
    comparisons.to_csv(output / 'common_alarm_comparison.csv', index=False)
    thresholds.to_csv(output / 'seungho_cusum_threshold_audit.csv', index=False)
    rules.to_csv(output / 'seungho_fixed_rule_audit.csv', index=False)
    old = ROOT.parent / '모듈1_NetHealth' / '03_처리기록' / 'primary40_summary.csv'
    old_table = pd.read_csv(old)
    old_table.to_csv(output / 'existing_primary40_summary.csv', index=False)
    write_graph(comparisons, output)
    assert input_manifest() == inputs
    assert _sha(report) == preserved_before
    evidence = dict(quality=quality, reference=reference, inputs=inputs,
                    existing_report_sha256=preserved_before, existing_report_preserved=True,
                    existing_results_source=str(old), existing_results_sha256=_sha(old),
                    main_commit=subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip(),
                    branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                    note='Existing report values are preserved prior runs, not newly rerun here.')
    (output / 'audit_manifest.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(output=str(output), counts=counts, comparisons=comparisons,
                cusum_thresholds=thresholds, fixed_rules=rules, evidence=evidence)
