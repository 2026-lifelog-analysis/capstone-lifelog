# 작성 2026-10-20 13:30 · 개정 2026-10-20 16:10 (문구·줄바꿈 검수, 강조 표시, 디자인 변주)
"""
모니터링 데모: 일반 사용자(본인·가족)가 휴대폰으로 보는 화면.

설계 원칙 (docs/DASHBOARD_DESIGN.md 4.2절)
  - 첫 화면은 한 문장 + 카드 4장. 자세한 내용은 카드를 눌러서 본다 (점진적 공개)
  - 전문 용어 대신 일상어. "경보선", "CUSUM", "기준선", 퍼센트 대신 "평소 범위", "평소보다 적어요"
  - 날짜를 고르지 않는다. 항상 '오늘' 기준, 지난날은 '어제', 'n일 전'
  - 큰 글씨 전환, 넉넉한 터치 영역, 빨강 대신 부드러운 호박색, 진단처럼 들리는 말을 쓰지 않음
  - 핵심 숫자는 굵게. 문구에 **...**로 표시하면 화면에서 굵게 바뀐다

상태 판정과 문장은 이 파일(Python)에서 정하고, 화면 그리기만 JavaScript가 한다.
"""
import json

import numpy as np

WEEKDAY = ['월', '화', '수', '목', '금', '토', '일']
REG_OK, REG_WARN = 65, 90          # 수면 규칙성 경계 (분): Brindle 2019 양호 < 65, Huang 2020 위험 > 90


# ---------------------------------------------------------------------------
# 표시 형식
# ---------------------------------------------------------------------------
def fmt_steps(v):
    return '기록 없음' if v is None else f'{int(round(v)):,}보'


def usual_steps(v):
    """평소 값은 추정치라 100보 단위로."""
    return '기록 없음' if v is None else f'약\u00a0{int(round(v, -2)):,}보'


def fmt_dur(m):
    if m is None:
        return '기록 없음'
    m = int(round(m))
    return f'{m // 60}시간\u00a0{m % 60}분' if m % 60 else f'{m // 60}시간'


def usual_dur(m):
    return '기록 없음' if m is None else fmt_dur(round(m / 10) * 10)


def fmt_clock(m):
    """정오 기준 분 → '새벽 2:32' 같은 말."""
    if m is None:
        return '기록 없음'
    t = (int(round(m)) + 720) % 1440
    h, mi = t // 60, t % 60
    part = '새벽' if h < 5 else '오전' if h < 12 else '오후' if h < 18 else '저녁' if h < 21 else '밤'
    return f'{part}\u00a0{h % 12 or 12}:{mi:02d}'


def usual_clock(m):
    return '기록 없음' if m is None else fmt_clock(round(m / 5) * 5)


def ago(n):
    return '오늘' if n == 0 else '어제' if n == 1 else f'{n}일 전'


# ---------------------------------------------------------------------------
# 상태 판정. 칩 문구는 카드 폭에 맞게 짧게 (한 줄, 8자 안팎)
# ---------------------------------------------------------------------------
def level(v, med, sd, low_word, high_word):
    """평소 범위(직전 28일 중앙값 ± 흔들림 폭)와 비교."""
    if v is None or med is None or sd is None:
        return 'none', '기록이 부족해요'
    if v < med - sd:
        return 'low', low_word
    if v > med + sd:
        return 'high', high_word
    return 'ok', '평소와 비슷해요'


def view_for(p, day):
    """오늘 = day(화면 구간 일차)로 보는 화면 하나의 내용."""
    d, B = p['days'], 28
    k = B + day

    def g(key, i=k):
        return d[key][i]

    steps_lv = level(g('steps'), g('steps_med'), g('steps_sd'), '평소보다 적어요', '평소보다 많아요')
    sleep_lv = level(g('sleep_min'), g('sleep_med'), g('sleep_sd'), '평소보다 짧아요', '평소보다 길어요')
    bed, bmed = g('bed_min'), g('bed_med')
    if bed is None or bmed is None:
        bed_lv = ('none', '기록이 부족해요')
    elif bed - bmed >= 60:
        bed_lv = ('low', '평소보다 늦어요')
    elif bmed - bed >= 60:
        bed_lv = ('high', '평소보다 일러요')
    else:
        bed_lv = ('ok', '평소와 비슷해요')
    reg = g('mid_sd')
    reg_lv = (('none', '기록 없음') if reg is None else ('ok', '일정해요') if reg < REG_OK
              else ('mid', '조금 들쭉날쭉해요') if reg <= REG_WARN else ('low', '들쭉날쭉해요'))

    recent = range(max(0, k - 2), k + 1)
    steps_alert = any(d['a_final'][i] for i in recent)
    reg_alert = any(d['a_reg'][i] for i in recent)
    # 알림은 며칠의 누적을 본다. 오늘 하루만 본 카드 문구가 알림과 엇갈리지 않게 맞춘다
    if steps_alert:
        steps_lv = ('low', '며칠째 적어요')
    if reg_alert:
        reg_lv = ('low', '며칠째 들쭉날쭉해요')
    build = (g('S_final') or 0) / p['h']['final']
    term, event = g('term'), g('event')
    common = term == '방학' or event in ('휴일', '시험')
    if steps_alert:
        hero = ('attention', '최근 며칠, 평소보다 덜 움직이셨어요',
                '지난 **4주**의 나와 비교하면 걸음이 적은 날이 이어지고 있어요.')
    elif reg_alert:
        hero = ('attention', '요즘 잠드는 시간이 들쭉날쭉해요',
                '지난 **4주**의 나와 비교하면 잠드는 시간이 고르지 않아요.')
    elif build >= 0.5:
        hero = ('watch', '조금 덜 움직이는 날이 이어지고 있어요', '아직 평소 범위 안이에요. 며칠 더 지켜볼게요.')
    else:
        hero = ('calm', '평소와 비슷한 하루를 보내고 계세요', '지난 **4주**의 나와 비교했어요.')
    note = ('방학·휴일처럼 모두의 생활이 함께 바뀌는 시기예요. 이런 변화는 감안해서 알려 드려요.'
            if common else None)

    def series(key, n):
        base = key.replace('_min', '') if key in ('sleep_min', 'bed_min') else key
        idx = list(range(max(0, k - n + 1), k + 1))
        return {'v': [d[key][i] for i in idx], 'med': [d[base + '_med'][i] for i in idx],
                'sd': [d[base + '_sd'][i] for i in idx], 'wd': [WEEKDAY[d['dow'][i]] for i in idx]}

    alerts = []
    for i in range(k, max(-1, k - 28), -1):
        if d['a_final'][i]:
            alerts.append({'when': ago(k - i), 'text': '걸음이 적은 날이 이어졌어요'})
        if d['a_reg'][i]:
            alerts.append({'when': ago(k - i), 'text': '잠드는 시간이 들쭉날쭉했어요'})

    def mean(key, a, b):
        v = [d[key][i] for i in range(max(0, a), b + 1) if d[key][i] is not None]
        return float(np.mean(v)) if v else None

    this_w, last_w = mean('steps', k - 6, k), mean('steps', k - 13, k - 7)
    if this_w is not None and last_w is not None:
        diff = this_w - last_w
        week_steps = (fmt_steps(this_w), f"하루 평균 · 지난주보다 **{fmt_steps(abs(diff))}**\u00a0"
                                         f"{'적어요' if diff < 0 else '많아요'}")
    else:
        week_steps = ('기록 없음', '')
    ws = mean('sleep_min', k - 6, k)
    reg_txt = f'**{int(round(reg))}분**' if reg is not None else ''

    metrics = [
        {'id': 'steps', 'name': '걸음', 'value': fmt_steps(g('steps')),
         'usual': f"평소 **{usual_steps(g('steps_med'))}**", 'lv': steps_lv, 'chart': 'bar',
         'w': series('steps', 7), 'm': series('steps', 28),
         'tip': '하루 걸음이 평소보다 **꾸준히** 줄어드는 것은 컨디션 변화의 신호일 수 있어요. 하루 이틀 적은 것은 흔한 일이에요.',
         'how': '지난 4주 동안의 내 걸음으로 "평소 범위"를 만들어요. 요일마다 다른 생활과, 휴일처럼 모두가 함께 덜 걷는 날은 감안해요. '
                '평소 범위보다 적은 날이 며칠 이어지면 알려 드려요.'},
        {'id': 'sleep', 'name': '수면 시간', 'value': fmt_dur(g('sleep_min')),
         'usual': f"평소 **{usual_dur(g('sleep_med'))}**", 'lv': sleep_lv, 'chart': 'bar',
         'w': series('sleep_min', 7), 'm': series('sleep_min', 28),
         'tip': '성인은 하루 **7시간 이상** 자는 것이 좋아요. (미국수면의학회 권고)',
         'how': '밤사이 잠든 시간을 모두 더해요. 지난 4주의 내 수면과 비교해요.'},
        {'id': 'bed', 'name': '잠든 시각', 'value': fmt_clock(bed),
         'usual': f'평소 **{usual_clock(bmed)}**', 'lv': bed_lv, 'chart': 'dot',
         'w': series('bed_min', 7), 'm': series('bed_min', 28),
         'tip': '매일 **비슷한 시각**에 잠드는 것이 좋아요.',
         'how': '하룻밤 중 가장 길게 잔 잠이 시작된 시각이에요. 지난 4주의 내 시각과 비교해요.'},
        {'id': 'reg', 'name': '수면 규칙성', 'value': reg_lv[1] if reg is not None else '기록 없음',
         'usual': f'최근 **7일**, 잠드는 시간 차이\u00a0{reg_txt}' if reg is not None else '',
         'lv': reg_lv, 'chart': 'line', 'w': series('mid_sd', 7), 'm': series('mid_sd', 28),
         'tip': f'잠드는 시간이 매일 비슷할수록 좋아요. 일주일 동안의 차이가 **{REG_OK}분**보다 작으면 일정한 편, '
                f'**{REG_WARN}분**보다 크면 들쭉날쭉한 편이에요.',
         'how': '최근 7일 동안 잠의 한가운데 시각이 날마다 얼마나 달랐는지 계산해요.'},
    ]
    for m in metrics:
        m['lv'] = list(m['lv'])
    return {'day': day, 'weekday': WEEKDAY[g('dow')], 'hero': list(hero), 'note': note, 'metrics': metrics,
            'alerts': alerts[:5],
            'week': [['걸음', week_steps[0], week_steps[1]],
                     ['수면', fmt_dur(ws) if ws else '기록 없음', '하루 평균'],
                     ['잠드는 시간', reg_lv[1], f'최근 7일 차이\u00a0{reg_txt}' if reg is not None else '']]}


def presets(p):
    """발표자가 고르는 데모 장면 (사용자 화면에는 없음). 규칙으로 정한다."""
    d, B = p['days'], 28
    alert = p['alarm_day']
    calm = next(t for t in range(alert - 1, -1, -1)
                if (d['S_final'][B + t] or 0) / p['h']['final'] < 0.2 and d['term'][B + t] == '학기'
                and d['steps'][B + t] is not None)
    vac = next((t for t in range(0, alert) if d['a_base'][B + t] and not d['a_final'][B + t]), None)
    out = [('알림이 뜬 날', alert), ('평소 날', calm)]
    if vac is not None:
        out.append(('방학 중 덜 걸은 날', vac))
    return out


def payload(p):
    return {'label': p['label'], 'scenes': [{'name': n, 'view': view_for(p, t)} for n, t in presets(p)]}


# ---------------------------------------------------------------------------
# 화면 (따뜻한 종이색 바탕, 테두리 카드, 지표별 선 아이콘, 왼쪽 띠가 있는 상태 카드)
# ---------------------------------------------------------------------------
CSS = """
.app{--bg:#f6f4f0;--card:#fff;--ink:#23211d;--sub:#78726a;--line:#ebe6de;
--steps:#2f9e6e;--sleep:#6b62c8;--bed:#3a7bd5;--reg:#1d97a6;
--warn:#d98a1c;--warnbg:#fdf3e2;--calm:#2f9e6e;--calmbg:#e8f5ee;--watch:#8a847b;
font-family:"Pretendard","Apple SD Gothic Neo",-apple-system,BlinkMacSystemFont,"Malgun Gothic","Noto Sans KR",sans-serif;
background:var(--bg);color:var(--ink);font-size:17px;line-height:1.5;-webkit-font-smoothing:antialiased;
word-break:keep-all;overflow-wrap:break-word}
.app.big{font-size:20px}
.app *{box-sizing:border-box}.app b{font-weight:800}
.app h3,.app .t,.app .v{text-wrap:balance}.app p,.app .u{text-wrap:pretty}
.scr{display:none;padding:14px 18px 30px}.scr.on{display:block}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;padding:6px 2px 14px}.top>div{flex:1;min-width:0}
.top .s{color:var(--sub);font-size:.85em;font-weight:600}.top .t{font-size:1.6em;font-weight:800;letter-spacing:-.03em;line-height:1.25}
.pill{flex:none;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:12px;padding:8px 12px;font-size:.82em;font-weight:700;
min-height:44px;cursor:pointer;font-family:inherit;white-space:nowrap}
.pill[aria-pressed=true]{background:var(--ink);color:#fff;border-color:var(--ink)}
.hero{position:relative;background:var(--card);border:1px solid var(--line);border-radius:20px;padding:18px 18px 18px 22px;margin:4px 0 12px;
display:flex;gap:14px;align-items:flex-start;overflow:hidden}
.hero::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px}
.hero.attention::before{background:var(--warn)}.hero.calm::before{background:var(--calm)}.hero.watch::before{background:var(--watch)}
.hero .ic{flex:none;width:44px;height:44px;border-radius:14px;display:grid;place-items:center}
.hero.attention .ic{background:var(--warnbg);color:var(--warn)}.hero.calm .ic{background:var(--calmbg);color:var(--calm)}.hero.watch .ic{background:#f1efea;color:var(--watch)}
.hero .ic svg{width:24px;height:24px}
.hero h3{margin:0 0 6px;font-size:1.2em;line-height:1.35;letter-spacing:-.02em}.hero p{margin:0;color:#4a463f;font-size:.93em}
.note{border:1px dashed #d9d2c6;border-radius:14px;padding:11px 14px;color:var(--sub);font-size:.86em;margin:0 0 12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 14px 10px;text-align:left;cursor:pointer;font-family:inherit;
color:inherit;min-height:44px;font-size:1em;display:flex;flex-direction:column}
.tile:active{transform:scale(.98)}
.mi{display:flex;align-items:center;gap:8px;font-size:.84em;font-weight:700}
.mi .sq{width:26px;height:26px;border-radius:8px;display:grid;place-items:center}.mi .sq svg{width:16px;height:16px}
.tile .v{font-size:1.4em;font-weight:800;margin-top:10px;letter-spacing:-.03em;line-height:1.2}
.tile .u{font-size:.8em;color:var(--sub);margin-top:2px}
.chip{align-self:flex-start;display:inline-block;margin-top:8px;border-radius:8px;padding:2px 8px;font-size:.76em;font-weight:700;border:1px solid;white-space:nowrap}
.chip.ok{border-color:#ddd7cd;color:#5f5a52}.chip.low{border-color:#f0c88c;color:#a8600a;background:var(--warnbg)}
.chip.mid{border-color:#eedcaa;color:#8a6a10}.chip.high{border-color:#bcd3f2;color:#2b62b0}.chip.none{border-color:#e5e0d8;color:var(--sub)}
.sp{margin-top:auto;padding-top:8px}
.sec{font-size:1.05em;font-weight:800;margin:22px 2px 10px;letter-spacing:-.02em}
.list{background:var(--card);border:1px solid var(--line);border-radius:18px;overflow:hidden}
.row{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:13px 16px;border-bottom:1px solid var(--line)}.row:last-child{border-bottom:0}
.row .k{font-weight:700;white-space:nowrap;flex:none}.row .r{text-align:right}.row .r b{font-size:1.05em}
.row .r small{display:block;color:var(--sub);font-size:.8em}.row .w{color:var(--sub);font-size:.88em;white-space:nowrap}
.empty{padding:16px;color:var(--sub)}
.foot{color:var(--sub);font-size:.8em;text-align:center;margin-top:24px;text-wrap:balance}
.back{border:0;background:none;color:var(--ink);font-size:.95em;font-weight:700;padding:10px 0;min-height:44px;cursor:pointer;font-family:inherit}
.dhead .v{font-size:2.1em;font-weight:800;letter-spacing:-.03em;margin-top:8px;line-height:1.2}.dhead .u{color:var(--sub)}
.seg{display:flex;gap:18px;border-bottom:1px solid var(--line);margin:16px 0 10px}
.seg button{border:0;background:none;padding:10px 2px;font-size:.92em;font-weight:700;min-height:44px;cursor:pointer;font-family:inherit;
color:var(--sub);border-bottom:3px solid transparent;margin-bottom:-1px}
.seg button.on{color:var(--ink);border-bottom-color:var(--ink)}
.chart{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:12px 10px 6px}.chart svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--sub);font-size:.78em;padding:6px 6px 2px}
.legend i{display:inline-block;width:14px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 16px;margin-top:12px}.card .h{font-weight:800;margin-bottom:4px}
details.card summary{font-weight:700;cursor:pointer;min-height:28px}details.card p{margin:8px 0 0;color:#4a463f}
"""

HTML = """
<div class="app" id="app">
 <div class="scr on" id="home">
  <div class="top"><div><div class="s" id="hDay"></div><div class="t">오늘의 생활 리듬</div></div>
   <button class="pill" id="bigBtn" aria-pressed="false">글씨 크게</button></div>
  <div id="hero"></div><div id="note"></div>
  <div class="grid" id="tiles"></div>
  <div class="sec">이번 주 한눈에</div><div class="list" id="week"></div>
  <div class="sec">최근 4주 알림</div><div class="list" id="alerts"></div>
  <div class="foot">연구용 시연 화면이에요. 의학적 진단이 아니에요.<br>불편한 곳이 있다면 가족이나 의료진과 이야기해 보세요.</div>
 </div>
 <div class="scr" id="detail">
  <button class="back" id="backBtn">‹ 요약으로</button>
  <div class="dhead" id="dHead"></div>
  <div class="seg"><button class="on" data-r="w">최근 1주</button><button data-r="m">최근 4주</button></div>
  <div class="chart"><div id="dChart"></div><div class="legend" id="dLegend"></div></div>
  <div class="card" id="dTip"></div>
  <details class="card"><summary>이 정보는 어떻게 계산하나요?</summary><p id="dHow"></p></details>
 </div>
</div>
"""

JS = r"""
(function(){
const DATA = __DATA__;
const app=document.getElementById('app');
let scene=0, cur=null, range='w';
const HEX={steps:'#2f9e6e',sleep:'#6b62c8',bed:'#3a7bd5',reg:'#1d97a6'};
const TINT={steps:'#e6f4ed',sleep:'#eeecfa',bed:'#e8f0fb',reg:'#e3f3f5'};
const SVG=(p)=>'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+p+'</svg>';
const ICON={steps:SVG('<path d="M4 16v-2.4C4 11.5 3 10.5 3 8c0-2.7 1.5-6 4.5-6C9.4 2 10 3.8 10 5.5c0 3.1-2 5.7-2 8.7V16a2 2 0 1 1-4 0z"/><path d="M20 20v-2.4c0-2.1 1-3.1 1-5.6 0-2.7-1.5-6-4.5-6-1.9 0-2.5 1.8-2.5 3.5 0 3.1 2 5.7 2 8.7V20a2 2 0 1 0 4 0z"/><path d="M16 17h4M4 13h4"/>'),
  sleep:SVG('<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>'),
  bed:SVG('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>'),
  reg:SVG('<path d="M3 12c2-5 4-5 6 0s4 5 6 0 4-5 6 0"/>'),
  attention:SVG('<path d="M12 7v6"/><circle cx="12" cy="17" r=".6" fill="currentColor"/><circle cx="12" cy="12" r="9"/>'),
  calm:SVG('<path d="M7.5 12.5l3 3 6-6.5"/><circle cx="12" cy="12" r="9"/>'),
  watch:SVG('<circle cx="12" cy="12" r="9"/><path d="M8 12h.01M12 12h.01M16 12h.01"/>')};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const rich=s=>esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');
const plain=s=>String(s??'').replace(/\*\*/g,'');
function V(){return DATA.scenes[scene].view;}
function mi(m){return '<div class="mi"><span class="sq" style="background:'+TINT[m.id]+';color:'+HEX[m.id]+'">'+ICON[m.id]+'</span>'+esc(m.name)+'</div>';}
function chip(m){return m.value===m.lv[1]?'':'<span class="chip '+m.lv[0]+'">'+esc(m.lv[1])+'</span>';}
function home(){
  const v=V();
  document.getElementById('hDay').textContent='오늘 · '+v.weekday+'요일';
  const h=v.hero;
  document.getElementById('hero').innerHTML='<div class="hero '+h[0]+'" role="status"><div class="ic">'+ICON[h[0]]+'</div><div><h3>'+esc(h[1])+'</h3><p>'+rich(h[2])+'</p></div></div>';
  document.getElementById('note').innerHTML=v.note?'<div class="note">'+rich(v.note)+'</div>':'';
  document.getElementById('tiles').innerHTML=v.metrics.map((m,i)=>
    '<button class="tile" data-i="'+i+'" aria-label="'+esc(m.name)+' 자세히 보기">'+mi(m)+
    '<div class="v">'+esc(m.value)+'</div><div class="u">'+rich(m.usual)+'</div>'+chip(m)+'<div class="sp">'+spark(m)+'</div></button>').join('');
  document.querySelectorAll('.tile').forEach(t=>t.onclick=()=>open(+t.dataset.i));
  document.getElementById('week').innerHTML=v.week.map(r=>'<div class="row"><span class="k">'+esc(r[0])+'</span><span class="r"><b>'+esc(r[1])+'</b>'+(r[2]?'<small>'+rich(r[2])+'</small>':'')+'</span></div>').join('');
  document.getElementById('alerts').innerHTML=v.alerts.length?v.alerts.map(a=>'<div class="row"><span>'+esc(a.text)+'</span><span class="w">'+esc(a.when)+'</span></div>').join(''):'<div class="empty">최근 4주 동안 알림이 없었어요.</div>';
}
function spark(m){
  const s=m.w, vals=s.v, ok=vals.filter(x=>x!=null), meds=s.med.filter(x=>x!=null);
  if(!ok.length) return '';
  const W=140,H=34,n=vals.length,slot=W/n,bw=Math.min(12,slot-6);
  const mx=Math.max(...ok,...meds)*1.08, mn=m.chart==='bar'?0:Math.min(...ok,...meds)*0.92;
  const y=x=>H-(x-mn)/(mx-mn||1)*H;
  let out='';
  vals.forEach((x,i)=>{if(x==null)return;const last=i===n-1,cx=i*slot+slot/2;
    if(m.chart==='bar') out+='<rect x="'+(cx-bw/2)+'" y="'+y(x)+'" width="'+bw+'" height="'+Math.max(2,H-y(x))+'" rx="'+bw/2+'" fill="'+HEX[m.id]+'" opacity="'+(last?1:.28)+'"/>';
    else out+='<circle cx="'+cx+'" cy="'+y(x)+'" r="'+(last?4.5:3)+'" fill="'+HEX[m.id]+'" opacity="'+(last?1:.35)+'"/>';});
  const md=s.med[n-1];
  if(md!=null) out='<line x1="0" x2="'+W+'" y1="'+y(md)+'" y2="'+y(md)+'" stroke="#b7afa3" stroke-dasharray="3 3"/>'+out;
  return '<svg viewBox="0 -3 '+W+' '+(H+6)+'" width="100%" height="38" aria-hidden="true">'+out+'</svg>';
}
function open(i){cur=i;range='w';document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on',b.dataset.r==='w'));
  document.getElementById('home').classList.remove('on');document.getElementById('detail').classList.add('on');detail();app.scrollIntoView({block:'start'});}
function detail(){
  const m=V().metrics[cur];
  document.getElementById('dHead').innerHTML=mi(m)+'<div class="v">'+esc(m.value)+'</div><div class="u">'+rich(m.usual)+'</div>'+chip(m);
  document.getElementById('dChart').innerHTML=chart(m, range==='w'?m.w:m.m);
  document.getElementById('dLegend').innerHTML='<span><i style="background:'+HEX[m.id]+'"></i>내 기록</span><span><i style="background:#e4ded3"></i>평소 범위 (지난 4주)</span>';
  document.getElementById('dTip').innerHTML='<div class="h">알아두면 좋아요</div>'+rich(m.tip);
  document.getElementById('dHow').textContent=plain(m.how);
}
function fmtY(m,x){if(m.id==='steps')return x===0?'0':(x>=10000?(x/10000).toFixed(x%10000?1:0)+'만':(x/1000).toFixed(x%1000?1:0)+'천');if(m.id==='sleep')return Math.round(x/60)+'시간';
  if(m.id==='bed'){let t=(Math.round(x)+720)%1440;return Math.floor(t/60)+'시';}return Math.round(x)+'분';}
function chart(m,s){
  const W=340,H=190,L=40,R=8,T=10,Bm=26,n=s.v.length,cw=(W-L-R)/n;
  const all=[].concat(s.v,s.med.map((x,i)=>x!=null&&s.sd[i]!=null?x+s.sd[i]:null),s.med.map((x,i)=>x!=null&&s.sd[i]!=null?x-s.sd[i]:null)).filter(x=>x!=null);
  const raw=(Math.max(...all)-(m.chart==='bar'?0:Math.min(...all)))/3||1, p10=Math.pow(10,Math.floor(Math.log10(raw)));
  const stp=[1,2,2.5,5,10].map(f=>f*p10).find(v=>v>=raw);
  let mn=m.chart==='bar'?0:Math.floor(Math.min(...all)/stp)*stp, mx=Math.ceil(Math.max(...all)/stp)*stp;
  const y=x=>T+(H-T-Bm)*(1-(x-mn)/(mx-mn||1)), cx=i=>L+cw*i+cw/2;
  let g='';
  for(let val=mn;val<=mx+1e-9;val+=stp){const yy=y(val);g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+yy+'" y2="'+yy+'" stroke="#efebe4"/><text x="'+(L-6)+'" y="'+(yy+4)+'" text-anchor="end" font-size="11" fill="#9a938a">'+fmtY(m,val)+'</text>';}
  let up='',lo='';s.med.forEach((x,i)=>{if(x==null||s.sd[i]==null)return;up+=(up?' L':'M')+cx(i)+' '+y(x+s.sd[i]);});
  for(let i=n-1;i>=0;i--){const x=s.med[i];if(x==null||s.sd[i]==null)continue;lo+=' L'+cx(i)+' '+y(Math.max(mn,x-s.sd[i]));}
  if(up) g+='<path d="'+up+lo+' Z" fill="#e4ded3" opacity=".7"/>';
  s.v.forEach((x,i)=>{if(x==null)return;const last=i===n-1;
    if(m.chart==='bar'){const bw=Math.max(3,Math.min(18,cw*.6));g+='<rect x="'+(cx(i)-bw/2)+'" y="'+y(x)+'" width="'+bw+'" height="'+(y(mn)-y(x))+'" rx="'+Math.min(5,bw/2)+'" fill="'+HEX[m.id]+'" opacity="'+(last?1:.5)+'"/>';}
    else g+='<circle cx="'+cx(i)+'" cy="'+y(x)+'" r="'+(last?5.5:4)+'" fill="'+HEX[m.id]+'" opacity="'+(last?1:.55)+'"/>';});
  if(m.chart!=='bar'){let p='';s.v.forEach((x,i)=>{if(x==null)return;p+=(p?' L':'M')+cx(i)+' '+y(x);});g+='<path d="'+p+'" fill="none" stroke="'+HEX[m.id]+'" stroke-width="2" opacity=".45"/>';}
  const step=n>10?7:1;
  s.wd.forEach((w,i)=>{const back=n-1-i;if(back%step)return;const lab=back===0?'오늘':(n>10?back+'일 전':w);
    g+='<text x="'+cx(i)+'" y="'+(H-6)+'" text-anchor="middle" font-size="11" fill="'+(back===0?'#23211d':'#9a938a')+'" font-weight="'+(back===0?700:400)+'">'+lab+'</text>';});
  return '<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(m.name)+' 기록과 평소 범위">'+g+'</svg>';
}
document.getElementById('backBtn').onclick=()=>{document.getElementById('detail').classList.remove('on');document.getElementById('home').classList.add('on');};
document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>{range=b.dataset.r;document.querySelectorAll('.seg button').forEach(x=>x.classList.toggle('on',x===b));detail();});
const big=document.getElementById('bigBtn');
big.onclick=()=>{const on=app.classList.toggle('big');big.setAttribute('aria-pressed',on);big.textContent=on?'기본 글씨':'글씨 크게';};
window.demoScene=i=>{scene=i;document.getElementById('detail').classList.remove('on');document.getElementById('home').classList.add('on');home();
  document.querySelectorAll('.scene-btn').forEach((b,j)=>b.classList.toggle('on',j===i));};
home();
})();
"""


def render(p):
    """데모 앱 조각 (CSS, 본문, 스크립트). build.py가 휴대폰 틀 안에 넣는다."""
    data = payload(p)
    return CSS, HTML, JS.replace('__DATA__', json.dumps(data, ensure_ascii=False)), data
