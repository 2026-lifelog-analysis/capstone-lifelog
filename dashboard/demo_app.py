# 작성 2026-10-20 13:30
"""
모니터링 데모: 일반 사용자(본인·가족)가 휴대폰으로 보는 화면.

설계 원칙 (docs/DASHBOARD_DESIGN.md 4.2절)
  - 첫 화면은 한 문장 + 카드 4장. 자세한 내용은 카드를 눌러서 본다 (점진적 공개)
  - 전문 용어 대신 일상어. "경보선", "CUSUM", "기준선", 퍼센트 대신 "평소 범위", "평소보다 적어요"
  - 날짜를 고르지 않는다. 항상 '오늘' 기준, 지난날은 '어제', 'n일 전'
  - 큰 글씨 전환, 넉넉한 터치 영역, 빨강 대신 부드러운 주황, 진단처럼 들리는 말을 쓰지 않음

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
    return '기록 없음' if v is None else f'약 {int(round(v, -2)):,}보'


def usual_dur(m):
    return '기록 없음' if m is None else fmt_dur(round(m / 10) * 10)


def usual_clock(m):
    return '기록 없음' if m is None else fmt_clock(round(m / 5) * 5)


def fmt_dur(m):
    if m is None:
        return '기록 없음'
    m = int(round(m))
    return f'{m // 60}시간 {m % 60}분' if m % 60 else f'{m // 60}시간'


def fmt_clock(m):
    """정오 기준 분 → '새벽 2:32' 같은 말."""
    if m is None:
        return '기록 없음'
    t = (int(round(m)) + 720) % 1440
    h, mi = t // 60, t % 60
    part = '새벽' if h < 5 else '오전' if h < 12 else '오후' if h < 18 else '저녁' if h < 21 else '밤'
    h12 = h % 12 or 12
    return f'{part} {h12}:{mi:02d}'


def ago(n):
    return '오늘' if n == 0 else '어제' if n == 1 else f'{n}일 전'


# ---------------------------------------------------------------------------
# 상태 판정
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
    sleep_lv = level(g('sleep_min'), g('sleep_med'), g('sleep_sd'), '평소보다 적게 주무셨어요', '평소보다 오래 주무셨어요')
    bed, bmed = g('bed_min'), g('bed_med')
    if bed is None or bmed is None:
        bed_lv = ('none', '기록이 부족해요')
    elif bed - bmed >= 60:
        bed_lv = ('low', '평소보다 늦게 주무셨어요')
    elif bmed - bed >= 60:
        bed_lv = ('high', '평소보다 일찍 주무셨어요')
    else:
        bed_lv = ('ok', '평소와 비슷해요')
    reg = g('mid_sd')
    reg_lv = (('none', '기록이 부족해요') if reg is None else ('ok', '일정해요') if reg < REG_OK
              else ('mid', '조금 들쭉날쭉해요') if reg <= REG_WARN else ('low', '들쭉날쭉해요'))

    recent = range(max(0, k - 2), k + 1)
    steps_alert = any(d['a_final'][i] for i in recent)
    reg_alert = any(d['a_reg'][i] for i in recent)
    # 알림은 며칠의 누적을 본다. 오늘 하루만 본 카드 문구가 알림과 엇갈리지 않게 맞춘다
    if steps_alert:
        steps_lv = ('low', '며칠째 평소보다 적어요')
    if reg_alert:
        reg_lv = ('low', '며칠째 들쭉날쭉해요')
    build = (g('S_final') or 0) / p['h']['final']
    term, event = g('term'), g('event')
    common = term == '방학' or event in ('휴일', '시험')
    if steps_alert:
        hero = ('attention', '최근 며칠, 평소보다 덜 움직이셨어요',
                '지난 4주의 나와 비교했을 때 걸음이 적은 날이 이어지고 있어요.')
    elif reg_alert:
        hero = ('attention', '요즘 잠드는 시간이 들쭉날쭉해요', '지난 4주의 나와 비교했을 때 잠드는 시간이 고르지 않아요.')
    elif build >= 0.5:
        hero = ('watch', '조금 덜 움직이는 날이 이어지고 있어요', '아직 평소 범위 안이에요. 며칠 더 지켜볼게요.')
    else:
        hero = ('calm', '평소와 비슷한 하루를 보내고 계세요', '지난 4주의 나와 비교했어요.')
    note = ('방학·휴일처럼 모두가 함께 생활이 바뀌는 시기예요. 이런 변화는 따로 감안해서 알려드려요.'
            if common else None)

    def series(key, n):
        lo = max(0, k - n + 1)
        idx = list(range(lo, k + 1))
        return {'v': [d[key][i] for i in idx],
                'med': [d.get(key + '_med', d.get(key.replace('_min', '') + '_med'))[i] for i in idx],
                'sd': [d.get(key + '_sd', d.get(key.replace('_min', '') + '_sd'))[i] for i in idx],
                'wd': [WEEKDAY[d['dow'][i]] for i in idx]}

    alerts = []
    for i in range(k, max(-1, k - 28), -1):
        if d['a_final'][i]:
            alerts.append({'when': ago(k - i), 'text': '걸음이 평소보다 적은 날이 이어졌어요', 'kind': 'steps'})
        if d['a_reg'][i]:
            alerts.append({'when': ago(k - i), 'text': '잠드는 시간이 들쭉날쭉했어요', 'kind': 'reg'})

    def mean(key, a, b):
        v = [d[key][i] for i in range(max(0, a), b + 1) if d[key][i] is not None]
        return float(np.mean(v)) if v else None

    this_w, last_w = mean('steps', k - 6, k), mean('steps', k - 13, k - 7)
    if this_w is not None and last_w is not None:
        diff = this_w - last_w
        week_steps = f"하루 평균 {fmt_steps(this_w)} · 지난주보다 {fmt_steps(abs(diff))} {'적어요' if diff < 0 else '많아요'}"
    else:
        week_steps = '기록이 부족해요'
    ws = mean('sleep_min', k - 6, k)

    metrics = [
        {'id': 'steps', 'name': '걸음', 'color': 'steps', 'value': fmt_steps(g('steps')),
         'usual': f"평소 {usual_steps(g('steps_med'))}", 'lv': steps_lv, 'chart': 'bar',
         'w': series('steps', 7), 'm': series('steps', 28),
         'tip': '하루 걸음이 평소보다 꾸준히 줄어드는 것은 컨디션 변화의 신호일 수 있어요. 하루 이틀 적은 것은 흔한 일이에요.',
         'how': '지난 4주 동안 내 걸음의 가운데 값과 흔들림 폭으로 "평소 범위"를 만들어요. 요일마다 다른 생활과, 휴일처럼 모두가 함께 덜 걷는 날은 감안해요. 평소 범위 아래인 날이 며칠 이어지면 알려드려요.'},
        {'id': 'sleep', 'name': '수면 시간', 'color': 'sleep', 'value': fmt_dur(g('sleep_min')),
         'usual': f"평소 {usual_dur(g('sleep_med'))}", 'lv': sleep_lv, 'chart': 'bar',
         'w': series('sleep_min', 7), 'm': series('sleep_min', 28),
         'tip': '성인은 하루 7시간 이상 자는 것이 권장돼요 (미국수면의학회).',
         'how': '밤사이 잠든 시간을 모두 더해요. 지난 4주의 내 수면과 비교해요.'},
        {'id': 'bed', 'name': '잠든 시각', 'color': 'bed', 'value': fmt_clock(bed),
         'usual': f'평소 {usual_clock(bmed)}', 'lv': bed_lv, 'chart': 'dot',
         'w': series('bed_min', 7), 'm': series('bed_min', 28),
         'tip': '매일 비슷한 시각에 잠드는 것이 좋아요.',
         'how': '하룻밤 중 가장 길게 잔 잠의 시작 시각이에요. 지난 4주의 내 시각과 비교해요.'},
        {'id': 'reg', 'name': '수면 규칙성', 'color': 'reg', 'value': reg_lv[1] if reg is not None else '기록 없음',
         'usual': f'최근 7일 잠드는 시간 차이 약 {int(round(reg))}분' if reg is not None else '',
         'lv': reg_lv, 'chart': 'line', 'w': series('mid_sd', 7), 'm': series('mid_sd', 28),
         'tip': f'잠드는 시간이 매일 비슷할수록 좋아요. 일주일 동안 차이가 {REG_OK}분보다 작으면 일정한 편, '
                f'{REG_WARN}분보다 크면 들쭉날쭉한 편이에요.',
         'how': '최근 7일 동안 잠의 한가운데 시각이 날마다 얼마나 달랐는지 계산해요.'},
    ]
    for m in metrics:
        m['lv'] = list(m['lv'])
    return {'day': day, 'weekday': WEEKDAY[g('dow')], 'hero': list(hero), 'note': note, 'metrics': metrics,
            'alerts': alerts[:5], 'week': {'steps': week_steps,
                                           'sleep': f'하루 평균 {fmt_dur(ws)}' if ws else '기록이 부족해요',
                                           'reg': reg_lv[1]}}


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


CSS = """
.app{--bg:#f2f2f7;--card:#fff;--ink:#1c1c1e;--sub:#6e6e73;--line:#e5e5ea;--steps:#34c759;--sleep:#5e5ce6;--bed:#0a84ff;
--reg:#30b0c7;--warn:#ff9f0a;--warnbg:#fff4e0;--calm:#34c759;--calmbg:#eaf8ee;--watchbg:#f2f2f7;
font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Apple SD Gothic Neo","Pretendard","Malgun Gothic","Noto Sans KR",sans-serif;
background:var(--bg);color:var(--ink);font-size:17px;line-height:1.45;-webkit-font-smoothing:antialiased}
.app.big{font-size:20px}
.app *{box-sizing:border-box}
.scr{display:none;padding:12px 16px 28px}.scr.on{display:block}
.top{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px 2px 10px}.top>div{flex:1;min-width:0}.top .pill{flex:none}
.top .t{font-size:1.75em;font-weight:800;letter-spacing:-.02em}.top .s{color:var(--sub);font-size:.9em}
.pill{border:0;background:var(--card);color:var(--ink);border-radius:999px;padding:10px 14px;font-size:.85em;font-weight:600;min-height:44px;cursor:pointer;font-family:inherit;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.hero{border-radius:22px;padding:20px;margin:6px 0 14px;display:flex;gap:14px;align-items:flex-start}
.hero.attention{background:var(--warnbg)}.hero.calm{background:var(--calmbg)}.hero.watch{background:var(--card)}
.hero .ic{flex:none;width:48px;height:48px;border-radius:50%;display:grid;place-items:center;font-size:1.4em;color:#fff}
.hero.attention .ic{background:var(--warn)}.hero.calm .ic{background:var(--calm)}.hero.watch .ic{background:#8e8e93}
.hero h3{margin:0 0 4px;font-size:1.25em;line-height:1.3;letter-spacing:-.01em}.hero p{margin:0;color:#3a3a3c;font-size:.95em}
.note{background:var(--card);border-radius:16px;padding:12px 14px;color:var(--sub);font-size:.88em;margin:-4px 0 14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.tile{background:var(--card);border-radius:20px;padding:14px 14px 10px;border:0;text-align:left;cursor:pointer;font-family:inherit;color:inherit;min-height:44px;box-shadow:0 1px 2px rgba(0,0,0,.05);font-size:1em}
.tile .n{font-size:.82em;font-weight:700;display:flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.tile .v{font-size:1.45em;font-weight:800;margin-top:6px;letter-spacing:-.02em;line-height:1.2}.tile .u{font-size:.78em;color:var(--sub)}
.chip{display:inline-block;margin-top:8px;border-radius:999px;padding:3px 10px;font-size:.78em;font-weight:700}
.chip.ok{background:#f2f2f7;color:#3a3a3c}.chip.low{background:var(--warnbg);color:#b25000}.chip.mid{background:#fff8e6;color:#8a5a00}
.chip.high{background:#e8f1ff;color:#0a5bd6}.chip.none{background:#f2f2f7;color:var(--sub)}
.sec{font-size:1.1em;font-weight:800;margin:22px 2px 10px}
.list{background:var(--card);border-radius:20px;overflow:hidden}
.row{display:flex;justify-content:space-between;gap:12px;padding:14px 16px;border-bottom:1px solid var(--line)}.row:last-child{border-bottom:0}
.row .k{font-weight:600;white-space:nowrap}.row .w{color:var(--sub);font-size:.9em;text-align:right}
.empty{padding:16px;color:var(--sub)}
.foot{color:var(--sub);font-size:.8em;text-align:center;margin-top:22px}
.back{border:0;background:none;color:#0a84ff;font-size:1em;font-weight:600;padding:10px 0;min-height:44px;cursor:pointer;font-family:inherit}
.dhead .n{font-size:1em;font-weight:700;display:flex;gap:8px;align-items:center}.dhead .v{font-size:2.1em;font-weight:800;letter-spacing:-.02em}
.seg{display:flex;background:#e3e3e8;border-radius:10px;padding:3px;margin:14px 0 8px}
.seg button{flex:1;border:0;background:none;padding:8px;border-radius:8px;font-size:.9em;font-weight:600;min-height:40px;cursor:pointer;font-family:inherit;color:var(--ink)}
.seg button.on{background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.12)}
.chart{background:var(--card);border-radius:20px;padding:12px 10px 6px}.chart svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:14px;color:var(--sub);font-size:.78em;padding:6px 6px 2px}.legend i{display:inline-block;width:14px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.card{background:var(--card);border-radius:20px;padding:14px 16px;margin-top:12px}.card b{display:block;margin-bottom:4px}
details.card summary{font-weight:700;cursor:pointer;min-height:28px}details.card p{margin:8px 0 0;color:#3a3a3c}
"""

HTML = """
<div class="app" id="app">
 <div class="scr on" id="home">
  <div class="top"><div><div class="s" id="hDay"></div><div class="t">오늘의 생활 리듬</div></div>
   <button class="pill" id="bigBtn" aria-pressed="false">가 큰 글씨</button></div>
  <div id="hero"></div><div id="note"></div>
  <div class="grid" id="tiles"></div>
  <div class="sec">이번 주 한눈에</div><div class="list" id="week"></div>
  <div class="sec">최근 4주 알림</div><div class="list" id="alerts"></div>
  <div class="foot">연구용 시연 화면이에요. 의학적 진단이 아니에요.<br>불편한 곳이 있다면 가족이나 의료진과 이야기해 보세요.</div>
 </div>
 <div class="scr" id="detail">
  <button class="back" id="backBtn">‹ 요약</button>
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
const COLOR={steps:'var(--steps)',sleep:'var(--sleep)',bed:'var(--bed)',reg:'var(--reg)'};
const HEX={steps:'#34c759',sleep:'#5e5ce6',bed:'#0a84ff',reg:'#30b0c7'};
const ICON={attention:'!',watch:'…',calm:'✓'};
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function V(){return DATA.scenes[scene].view;}
function home(){
  const v=V();
  document.getElementById('hDay').textContent='오늘 · '+v.weekday+'요일';
  const h=v.hero;
  document.getElementById('hero').innerHTML='<div class="hero '+h[0]+'" role="status"><div class="ic">'+ICON[h[0]]+'</div><div><h3>'+esc(h[1])+'</h3><p>'+esc(h[2])+'</p></div></div>';
  document.getElementById('note').innerHTML=v.note?'<div class="note">'+esc(v.note)+'</div>':'';
  document.getElementById('tiles').innerHTML=v.metrics.map((m,i)=>
    '<button class="tile" data-i="'+i+'" aria-label="'+esc(m.name)+' 자세히 보기"><div class="n"><span class="dot" style="background:'+COLOR[m.color]+'"></span>'+esc(m.name)+'</div>'+
    '<div class="v">'+esc(m.value)+'</div><div class="u">'+esc(m.usual)+'</div>'+(m.value===m.lv[1]?'':'<span class="chip '+m.lv[0]+'">'+esc(m.lv[1])+'</span>')+spark(m)+'</button>').join('');
  document.querySelectorAll('.tile').forEach(t=>t.onclick=()=>open(+t.dataset.i));
  const w=v.week;
  document.getElementById('week').innerHTML=[['걸음',w.steps],['수면',w.sleep],['잠드는 시간',w.reg]].map(r=>'<div class="row"><span class="k">'+r[0]+'</span><span class="w">'+esc(r[1])+'</span></div>').join('');
  document.getElementById('alerts').innerHTML=v.alerts.length?v.alerts.map(a=>'<div class="row"><span>'+esc(a.text)+'</span><span class="w">'+esc(a.when)+'</span></div>').join(''):'<div class="empty">최근 4주 동안 알림이 없었어요.</div>';
}
function spark(m){
  const s=m.w, vals=s.v.map(x=>x??null), ok=vals.filter(x=>x!=null);
  if(!ok.length) return '';
  const W=140,H=34,n=vals.length,bw=W/n-4;
  const mx=Math.max(...ok,...s.med.filter(x=>x!=null))*1.05, mn=m.chart==='bar'?0:Math.min(...ok,...s.med.filter(x=>x!=null))*0.95;
  const y=x=>H-(x-mn)/(mx-mn||1)*H;
  let out='';
  vals.forEach((x,i)=>{if(x==null)return;const last=i===n-1;
    if(m.chart==='bar') out+='<rect x="'+(i*(W/n)+2)+'" y="'+y(x)+'" width="'+bw+'" height="'+(H-y(x))+'" rx="2" fill="'+HEX[m.color]+'" opacity="'+(last?1:.35)+'"/>';
    else out+='<circle cx="'+(i*(W/n)+W/n/2)+'" cy="'+y(x)+'" r="'+(last?4:3)+'" fill="'+HEX[m.color]+'" opacity="'+(last?1:.4)+'"/>';});
  return '<svg viewBox="0 -2 '+W+' '+(H+4)+'" width="100%" height="36" aria-hidden="true" style="margin-top:8px">'+out+'</svg>';
}
function open(i){cur=i;range='w';document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on',b.dataset.r==='w'));
  document.getElementById('home').classList.remove('on');document.getElementById('detail').classList.add('on');detail();app.scrollIntoView({block:'start'});}
function detail(){
  const m=V().metrics[cur];
  document.getElementById('dHead').innerHTML='<div class="n"><span class="dot" style="background:'+COLOR[m.color]+'"></span>'+esc(m.name)+' · 오늘</div><div class="v">'+esc(m.value)+'</div><div class="u" style="color:var(--sub)">'+esc(m.usual)+'</div><span class="chip '+m.lv[0]+'">'+esc(m.lv[1])+'</span>';
  document.getElementById('dChart').innerHTML=chart(m, range==='w'?m.w:m.m);
  document.getElementById('dLegend').innerHTML='<span><i style="background:'+HEX[m.color]+'"></i>내 기록</span><span><i style="background:#d1d1d6"></i>평소 범위 (지난 4주)</span>';
  document.getElementById('dTip').innerHTML='<b>알아두면 좋아요</b>'+esc(m.tip);
  document.getElementById('dHow').textContent=m.how;
}
function fmtY(m,x){if(m.id==='steps')return Math.round(x/1000)+'천';if(m.id==='sleep')return Math.round(x/60)+'시간';
  if(m.id==='bed'){let t=(Math.round(x)+720)%1440;return Math.floor(t/60)+'시';}return Math.round(x)+'분';}
function chart(m,s){
  const W=340,H=190,L=40,R=8,T=10,Bm=26,n=s.v.length,cw=(W-L-R)/n;
  const all=[].concat(s.v,s.med.map((x,i)=>x!=null&&s.sd[i]!=null?x+s.sd[i]:null),s.med.map((x,i)=>x!=null&&s.sd[i]!=null?x-s.sd[i]:null)).filter(x=>x!=null);
  let mx=Math.max(...all)*1.05, mn=m.chart==='bar'?0:Math.min(...all)*0.97;
  const y=x=>T+(H-T-Bm)*(1-(x-mn)/(mx-mn||1)), cx=i=>L+cw*i+cw/2;
  let g='';
  for(let k=0;k<=3;k++){const val=mn+(mx-mn)*k/3,yy=y(val);g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+yy+'" y2="'+yy+'" stroke="#e5e5ea"/><text x="'+(L-6)+'" y="'+(yy+4)+'" text-anchor="end" font-size="11" fill="#8e8e93">'+fmtY(m,val)+'</text>';}
  let up='',lo='';s.med.forEach((x,i)=>{if(x==null||s.sd[i]==null)return;up+=(up?' L':'M')+cx(i)+' '+y(x+s.sd[i]);});
  for(let i=n-1;i>=0;i--){const x=s.med[i];if(x==null||s.sd[i]==null)continue;lo+=' L'+cx(i)+' '+y(Math.max(mn,x-s.sd[i]));}
  if(up) g+='<path d="'+up+lo+' Z" fill="#d1d1d6" opacity=".55"/>';
  s.v.forEach((x,i)=>{if(x==null)return;const last=i===n-1;
    if(m.chart==='bar'){const bw=Math.max(3,cw*.62);g+='<rect x="'+(cx(i)-bw/2)+'" y="'+y(x)+'" width="'+bw+'" height="'+(y(mn)-y(x))+'" rx="3" fill="'+HEX[m.color]+'" opacity="'+(last?1:.55)+'"/>';}
    else g+='<circle cx="'+cx(i)+'" cy="'+y(x)+'" r="'+(last?5.5:4)+'" fill="'+HEX[m.color]+'" opacity="'+(last?1:.6)+'"/>';});
  if(m.chart!=='bar'){let p='';s.v.forEach((x,i)=>{if(x==null)return;p+=(p?' L':'M')+cx(i)+' '+y(x);});g+='<path d="'+p+'" fill="none" stroke="'+HEX[m.color]+'" stroke-width="2" opacity=".5"/>';}
  const step=n>10?7:1;
  s.wd.forEach((w,i)=>{const back=n-1-i;if(back%step)return;const lab=back===0?'오늘':(n>10?back+'일 전':w);
    g+='<text x="'+cx(i)+'" y="'+(H-6)+'" text-anchor="middle" font-size="11" fill="'+(back===0?'#1c1c1e':'#8e8e93')+'" font-weight="'+(back===0?700:400)+'">'+lab+'</text>';});
  return '<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(m.name)+' 기록과 평소 범위">'+g+'</svg>';
}
document.getElementById('backBtn').onclick=()=>{document.getElementById('detail').classList.remove('on');document.getElementById('home').classList.add('on');};
document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>{range=b.dataset.r;document.querySelectorAll('.seg button').forEach(x=>x.classList.toggle('on',x===b));detail();});
const big=document.getElementById('bigBtn');
big.onclick=()=>{const on=app.classList.toggle('big');big.setAttribute('aria-pressed',on);};
window.demoScene=i=>{scene=i;document.getElementById('detail').classList.remove('on');document.getElementById('home').classList.add('on');home();
  document.querySelectorAll('.scene-btn').forEach((b,j)=>b.classList.toggle('on',j===i));};
home();
})();
"""


def render(p):
    """데모 앱 조각 (CSS, 본문, 스크립트). build.py가 휴대폰 틀 안에 넣는다."""
    data = payload(p)
    return CSS, HTML, JS.replace('__DATA__', json.dumps(data, ensure_ascii=False)), data
