# 작성 2026-10-20 10:20
"""
연구 탭 그림. 한 그림 = 한 함수, plotly Figure를 돌려준다.
숫자는 results_data.py, 참가자 A 기록은 export_person.py가 만든 JSON에서만 읽는다.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import results_data as RD
from style import BAND, BLUE, GRAY, NAVY, ORANGE, apply_style, legend_line, shade_context


def baseline_example(p):
    """R2. 참가자 A 84일: 걸음 + 기준선 띠 → 이탈 점수 → 누적합과 경보."""
    d = p['days']
    x = d['day']
    up = [m + s if m is not None and s is not None else None for m, s in zip(d['steps_med'], d['steps_sd'])]
    lo = [m - s if m is not None and s is not None else None for m, s in zip(d['steps_med'], d['steps_sd'])]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.45, 0.25, 0.30],
                        subplot_titles=['① 걸음 수와 개인 기준선 (직전 28일 중앙값 ± 흔들림 폭) · −28~−1일은 화면 앞 기준선 기간',
                                        '② 이탈 점수 (요일·같은 날 공통효과를 뺀 뒤, 클수록 평소보다 적게 걸음)',
                                        '③ 누적합 (CUSUM) — 경보선을 넘으면 경보'])
    fig.add_trace(go.Scatter(x=x, y=up, line=dict(width=0), hoverinfo='skip', showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=lo, fill='tonexty', fillcolor=BAND, line=dict(width=0), name='기준선 띠',
                             hoverinfo='skip'), 1, 1)
    fig.add_trace(go.Bar(x=x, y=d['steps'], marker_color='rgba(100,116,139,0.55)', name='걸음 수'), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=d['steps_med'], line=dict(color=BLUE, width=2.5), name='개인 기준선 (중앙값)'), 1, 1)
    fig.add_trace(go.Bar(x=x, y=d['b_final'], marker_color=[ORANGE if (v or 0) > 0 else '#cbd5e1' for v in d['b_final']],
                         name='이탈 점수', showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=d['S_final'], line=dict(color=NAVY, width=2.5), name='누적합'), 3, 1)
    fig.add_hline(y=p['h']['final'], line=dict(color=ORANGE, dash='dash'), row=3, col=1,
                  annotation_text='경보선 (하루 경보율 1/30)', annotation_position='top left')
    ax = [i for i, a in zip(x, d['a_final']) if a]
    fig.add_trace(go.Scatter(x=ax, y=[p['h']['final']] * len(ax), mode='markers',
                             marker=dict(symbol='triangle-down', size=16, color=ORANGE), name='경보'), 3, 1)
    shade_context(fig, d, rows=[1, 2, 3])
    fig.update_xaxes(title_text='일차 (0 = 화면 구간 시작)', row=3, col=1)
    return apply_style(fig, height=760, subplots=True)


def holiday_bars():
    """R3-①. 휴일 걸음 경보율: 개인 vs 집단."""
    h = RD.HOLIDAY_PERSONAL_VS_GROUP
    fig = go.Figure(go.Bar(x=h['labels'], y=h['values'], marker_color=[BLUE, GRAY],
                           text=[f'{v}%' for v in h['values']], textposition='outside', width=0.5))
    fig.update_yaxes(title_text='휴일 걸음 경보율 (%)', range=[0, max(h['values']) * 1.3])
    return apply_style(fig, height=360, title='휴일처럼 모두가 함께 덜 걷는 날')


def calendar_bars():
    """R3-②. 공통효과 제거 전후 달력 상태별 하루 경보율 (주입 없음)."""
    c = RD.CALENDAR_ALARM
    fig = go.Figure([
        go.Bar(x=c['rows'], y=c['base'], name='출발점 (개인+CUSUM)', marker_color=BLUE,
               text=[f'{v}%' for v in c['base']], textposition='outside'),
        go.Bar(x=c['rows'], y=c['final'], name='최종 (공통효과 제거 포함)', marker_color=NAVY,
               text=[f'{v}%' for v in c['final']], textposition='outside'),
    ])
    fig.add_hline(y=100 / 30, line=dict(color=GRAY, dash='dot'))
    legend_line(fig, '목표 하루 경보율 3.3%', GRAY, 'dot')
    fig.update_yaxes(title_text='하루 경보율 (%)', range=[0, 18])
    fig.update_layout(barmode='group')
    return apply_style(fig, height=360, title='주입 없는 실제 기록 · 달력 상태별 하루 경보율')


def person_methods(p):
    """R3-③. 참가자 A: 방학·휴일에 방법별로 경보가 어디서 울렸나."""
    d = p['days']
    v = slice(28, None)
    x = d['day'][v]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=d['steps'][v], marker_color='rgba(100,116,139,0.45)', name='걸음 수'))
    fig.add_trace(go.Scatter(x=x, y=d['steps_med'][v], line=dict(color=BLUE, width=2), name='개인 기준선'))
    top = max(s for s in d['steps'][v] if s) * 1.08
    for i, (key, name, color, sym) in enumerate([('a_base', '출발점 (개인+CUSUM)', BLUE, 'x'),
                                                  ('a_group', '집단 기준선', GRAY, 'diamond'),
                                                  ('a_final', '최종', ORANGE, 'triangle-down')]):
        xs = [xx for xx, a in zip(x, d[key][v]) if a]
        fig.add_trace(go.Scatter(x=xs, y=[top * (1 + 0.07 * i)] * len(xs), mode='markers', name=f'경보: {name}',
                                 marker=dict(symbol=sym, size=15, color=color, line=dict(width=1, color='white'))))
    shade_context(fig, {k: d[k][v] for k in ['day', 'term', 'event']})
    fig.update_xaxes(title_text='일차')
    fig.update_yaxes(title_text='걸음 수')
    return apply_style(fig, height=400, title='참가자 A · 방학 중 걸음 감소에 출발점 방법만 경보를 울림')


def ladder():
    """R4-①. 단계별 개선 (탐지율 − 오경보율)."""
    j = RD.INTEGRATED_J
    vals = [j[k][0] for k in RD.LADDER]
    fig = go.Figure(go.Bar(x=[s.replace('\n', '<br>') for s in RD.LADDER_LABELS], y=vals,
                           marker_color=['#93c5fd', '#60a5fa', '#2563eb', NAVY],
                           text=[f'{v:.2f}' for v in vals], textposition='outside', width=0.6))
    fig.add_hline(y=j['집단 기준선'][0], line=dict(color=GRAY, dash='dash'))
    legend_line(fig, f"집단 기준선 {j['집단 기준선'][0]:.2f}", GRAY)
    fig.update_traces(selector=dict(type='bar'), showlegend=False)
    fig.update_yaxes(title_text='탐지율 − 오경보율 (무작위 = 0)', range=[0, 0.62])
    return apply_style(fig, height=380, title='걸음 −30% 갑자기 · 개선 단계별')


def det_fa():
    """R4-②. 탐지율과 오경보율을 함께."""
    s, f = RD.DET_FA['start'], RD.DET_FA['final']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', line=dict(color=GRAY, dash='dot'),
                             name='무작위 (탐지율 = 오경보율)'))
    fig.add_trace(go.Scatter(x=[s['fa'], f['fa']], y=[s['det'], f['det']], mode='markers+text',
                             marker=dict(size=16, color=[BLUE, NAVY]), showlegend=False,
                             text=[f"{s['name']}<br>탐지 {s['det']}% · 오경보 {s['fa']}%",
                                   f"{f['name']}<br>탐지 {f['det']}% · 오경보 {f['fa']}%"],
                             textposition=['bottom right', 'top left']))
    fig.add_annotation(x=f['fa'], y=f['det'], ax=s['fa'], ay=s['det'], xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=NAVY)
    fig.update_xaxes(title_text='오경보율 (%) — 주입 안 한 같은 구간에서 울린 비율', range=[0, 60])
    fig.update_yaxes(title_text='탐지율 (%)', range=[30, 100])
    return apply_style(fig, height=380, title='탐지율은 크게 오르고 오경보율은 조금 늘어남')


def heatmap():
    """R4-③. 10개 점수 × 5개 시나리오."""
    names = list(RD.INTEGRATED_J)[::-1]
    z = [RD.INTEGRATED_J[n] for n in names]
    fig = go.Figure(go.Heatmap(z=z, x=[s.replace('\n', '<br>') for s in RD.SCENARIOS], y=names, zmin=-0.15, zmax=0.6,
                               colorscale=[[0, '#fee2e2'], [0.2, '#ffffff'], [1, NAVY]],
                               text=[[f'{v:.2f}' for v in r] for r in z], texttemplate='%{text}',
                               colorbar=dict(title='탐지율<br>− 오경보율')))
    return apply_style(fig, height=520, title='같은 조건에서 다시 잰 10개 점수 · 수면 시각 흔들림에서는 순위가 뒤집힘')


def label_table():
    """R5. 의학 기준 위험 신호 라벨의 발생 빈도."""
    L = RD.LABELS
    cols = list(zip(*L['rows']))
    fill = [['#eff6ff' if r[-1] == '사용' else '#f8fafc' for r in L['rows']]] * len(L['cols'])
    fig = go.Figure(go.Table(header=dict(values=L['cols'], fill_color=NAVY, font=dict(color='white', size=15),
                                         align='left', height=34),
                             cells=dict(values=cols, fill_color=fill, align='left', height=32, font=dict(size=14)),
                             columnwidth=[3.2, 1.8, 1.2, 1.8, 1.3]))
    return apply_style(fig, height=420, title='NetHealth에서 실제로 쓸 수 있는 기준 (발생 = 직전 4주 정상 → 이번 주 위반)')
