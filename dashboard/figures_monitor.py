# 작성 2026-10-20 10:40
"""모니터링 데모 탭 그림. 보호자가 한 사람을 볼 때의 화면 (원 제안서 산출물)."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from style import BAND, BLUE, ORANGE, PURPLE, apply_style, shade_context


def _clock(m):
    """정오 기준 분 → 시:분 문자열 (690 → 23:30)."""
    if m is None:
        return None
    t = (int(round(m)) + 720) % 1440
    return f'{t // 60:02d}:{t % 60:02d}'


def timeline(p):
    """D2. 최근 8주: 걸음 · 수면 · 잠든 시각 · 수면 규칙성, 개인 기준선과 경보."""
    d = p['days']
    v = slice(28, None)
    x = d['day'][v]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=['걸음 수', '수면 시간 (시간)', '잠든 시각 (위로 갈수록 늦게 잠)', '수면 규칙성 (수면 중앙 시각의 7일 흔들림, 분)'])

    up = [m + s if m is not None and s is not None else None for m, s in zip(d['steps_med'][v], d['steps_sd'][v])]
    lo = [m - s if m is not None and s is not None else None for m, s in zip(d['steps_med'][v], d['steps_sd'][v])]
    fig.add_trace(go.Scatter(x=x, y=up, line=dict(width=0), hoverinfo='skip', showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=lo, fill='tonexty', fillcolor=BAND, line=dict(width=0), name='평소 범위',
                             hoverinfo='skip'), 1, 1)
    fig.add_trace(go.Bar(x=x, y=d['steps'][v], marker_color='rgba(100,116,139,0.55)', name='기록'), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=d['steps_med'][v], line=dict(color=BLUE, width=2), name='평소 (직전 28일)'), 1, 1)

    hrs = [s / 60 if s else None for s in d['sleep_min'][v]]
    fig.add_trace(go.Scatter(x=x, y=hrs, mode='lines+markers', line=dict(color='#475569'), marker=dict(size=5),
                             showlegend=False, name='수면'), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=[s / 60 if s else None for s in d['sleep_med'][v]], line=dict(color=BLUE, width=2),
                             showlegend=False, name='평소'), 2, 1)

    fig.add_trace(go.Scatter(x=x, y=d['bed_min'][v], mode='lines+markers', line=dict(color='#475569'),
                             marker=dict(size=5), showlegend=False, name='잠든 시각',
                             text=[_clock(m) for m in d['bed_min'][v]], hovertemplate='%{text}'), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=d['bed_med'][v], line=dict(color=BLUE, width=2), showlegend=False, name='평소',
                             text=[_clock(m) for m in d['bed_med'][v]], hovertemplate='%{text}'), 3, 1)
    ticks = list(range(540, 1081, 120))
    fig.update_yaxes(tickvals=ticks, ticktext=[_clock(t) for t in ticks], row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=d['mid_sd'][v], mode='lines', line=dict(color=PURPLE, width=2), showlegend=False,
                             name='규칙성'), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=d['mid_sd_med'][v], line=dict(color=BLUE, width=2, dash='dot'), showlegend=False,
                             name='평소'), 4, 1)

    for key, row, ycol, name in [('a_final', 1, 'steps', '평소와 다름 (걸음)'), ('a_reg', 4, 'mid_sd', '평소와 다름 (규칙성)')]:
        xs = [xx for xx, a in zip(x, d[key][v]) if a]
        ys = [yy for yy, a in zip(d[ycol][v], d[key][v]) if a]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name=name,
                                 marker=dict(symbol='triangle-down', size=16, color=ORANGE)), row, 1)
    shade_context(fig, {k: d[k][v] for k in ['day', 'term', 'event']}, rows=[1, 2, 3, 4])
    fig.update_xaxes(title_text='일차', row=4, col=1)
    return apply_style(fig, height=900, subplots=True)

