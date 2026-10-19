# 작성 2026-10-20 10:20
"""대시보드 공통 색과 그림 모양. 색의 뜻은 docs/DASHBOARD_DESIGN.md 6.3절."""

BLUE = '#2563eb'      # 개인 기준선
GRAY = '#94a3b8'      # 집단 기준선, 기준점
NAVY = '#1e3a8a'      # 개선 방법 (최종)
PURPLE = '#7c3aed'    # 수면 리듬 지표
ORANGE = '#ea580c'    # 경보 / 평소와 다름 (빨강은 쓰지 않음)
BAND = 'rgba(37,99,235,0.15)'
FONT = 'Pretendard, "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif'

LABEL_POS = {'방학': 'top right', '휴일': 'top left', '시험': 'top left'}
CONTEXT = {'방학': 'rgba(148,163,184,0.16)', '휴일': 'rgba(234,179,8,0.16)', '시험': 'rgba(124,58,237,0.12)'}


def apply_style(fig, height=400, title=None, subplots=False):
    top = 120 if subplots else (90 if title else 50)
    fig.update_layout(height=height, template='plotly_white', font=dict(family=FONT, size=15, color='#0f172a'),
                      margin=dict(l=70, r=30, t=top, b=60),
                      legend=dict(orientation='h', yanchor='bottom', y=1.07 if subplots else 1.0, x=0,
                                  font=dict(size=13)),
                      hoverlabel=dict(font_family=FONT), bargap=0.15)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=18), x=0, xanchor='left'))
    fig.update_annotations(font_size=14)
    return fig


def _runs(xs, flags):
    """연속 구간 [(시작, 끝)]."""
    out, start, prev = [], None, None
    for x, f in zip(xs, flags):
        if f and start is None:
            start = x
        if not f and start is not None:
            out.append((start, prev))
            start = None
        prev = x
    if start is not None:
        out.append((start, prev))
    return out


def shade_context(fig, d, rows=None):
    """방학·휴일·시험 구간을 옅은 음영으로. 날짜 대신 맥락 태그만 쓴다."""
    marks = [('방학', [t == '방학' for t in d['term']]), ('휴일', [e == '휴일' for e in d['event']]),
             ('시험', [e == '시험' for e in d['event']])]
    for name, flags in marks:
        for i, (a, b) in enumerate(_runs(d['day'], flags)):
            kw = dict(x0=a - 0.5, x1=b + 0.5, fillcolor=CONTEXT[name], line_width=0, layer='below')
            if rows:
                for r in rows:
                    fig.add_vrect(row=r, col=1, **kw)
                fig.add_vrect(row=rows[0], col=1, x0=a - 0.5, x1=b + 0.5, fillcolor='rgba(0,0,0,0)', line_width=0,
                              annotation_text=name, annotation_position=LABEL_POS[name])
            else:
                fig.add_vrect(annotation_text=name, annotation_position=LABEL_POS[name], **kw)


def legend_line(fig, name, color, dash='dash'):
    """add_hline 기준선을 글자 대신 범례로 설명 (막대 숫자와 겹치지 않게)."""
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=color, dash=dash), name=name))
