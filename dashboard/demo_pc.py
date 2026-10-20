# 작성 2026-10-21 14:00
"""
모니터링 데모 PC판: 보호자·관리자가 모니터(발표 화면)에서 한 번에 보는 대시보드.

모바일판(demo_app.py)과 같은 사람·같은 날의 같은 판정을 쓰고, 화면만 다르다.
  - 16:9 화면 한 장에 스크롤 없이 모두 들어간다. 1600×900 캔버스를 그려 화면 크기에 맞게 줄이고 늘린다
  - 원 제안서 산출물 세 가지를 한 화면에: 개인 기준선 대비 이탈률 · 주간 리듬 변화 · 조기 알림 기록
  - 보호자·관리자용이라 모바일보다 숫자를 더 보여주되, "위험"·"이상" 대신 "평소와 다름"을 쓰고 진단이 아님을 고정 표시

상태 문장·카드 판정은 demo_app.view_for를 그대로 부르고, 여기서는 8주 타임라인·주간 집계·이탈률만 더 만든다.
"""
import json

import numpy as np

import demo_app as DA

B = 28          # 화면 구간 0일차 앞에 붙어 있는 기준선용 기록 일수
SPAN = 56       # 타임라인 길이 (8주)
WEEKS = 8


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _out_of_range(d, key, base, i):
    """평소 범위(중앙값 ± 흔들림 폭) 밖이면 1, 안이면 0, 기록이 없으면 None."""
    v, med, sd = d[key][i], d[base + '_med'][i], d[base + '_sd'][i]
    if v is None or med is None or sd is None:
        return None
    return int(abs(v - med) > sd)


def deviation_rates(d, k):
    """개인 기준선 대비 이탈률: 최근 7일 중 평소 범위를 벗어난 날의 비율, 그 전 3주와 비교."""
    rows = []
    for name, key, base in [('걸음', 'steps', 'steps'), ('수면 시간', 'sleep_min', 'sleep'),
                            ('잠든 시각', 'bed_min', 'bed'), ('수면 규칙성', 'mid_sd', 'mid_sd')]:
        now = [_out_of_range(d, key, base, i) for i in range(max(0, k - 6), k + 1)]
        before = [_out_of_range(d, key, base, i) for i in range(max(0, k - 27), max(0, k - 6))]
        now, before = [x for x in now if x is not None], [x for x in before if x is not None]
        rows.append({'name': name,
                     'now': round(100 * sum(now) / len(now)) if now else None,
                     'before': round(100 * sum(before) / len(before)) if before else None,
                     'n': len(now)})
    return rows


def weekly(d, k):
    """주간 리듬 변화: 최근 8주의 주별 평균과, 그 주 마지막 날의 평소 값."""
    out = []
    for w in range(WEEKS - 1, -1, -1):
        hi = k - 7 * w
        lo = hi - 6
        if hi < 0:
            continue
        idx = range(max(0, lo), hi + 1)
        out.append({'label': '이번 주' if w == 0 else f'-{w}주',
                    'steps': _mean(d['steps'][i] for i in idx), 'steps_med': d['steps_med'][hi],
                    'sleep': _mean(d['sleep_min'][i] for i in idx), 'sleep_med': d['sleep_med'][hi],
                    'reg': d['mid_sd'][hi],
                    'alerts': sum(bool(d['a_final'][i]) + bool(d['a_reg'][i]) for i in idx),
                    'context': next((t for t in ('시험', '휴일', '개강주') if any(d['event'][i] == t for i in idx)),
                                    '방학' if all(d['term'][i] == '방학' for i in idx) else '')})
    return out


def timeline(p, k):
    """최근 8주 하루 단위 기록. 오늘이 맨 오른쪽, 기록이 없는 앞쪽은 비운다."""
    d, h = p['days'], p['h']
    idx = list(range(k - SPAN + 1, k + 1))

    def col(key, nd=1):
        return [None if i < 0 or d[key][i] is None else round(d[key][i], nd) for i in idx]

    def flag(key):
        return [0 if i < 0 else int(bool(d[key][i])) for i in idx]

    return {'back': [k - i for i in idx],
            'wd': [None if i < 0 else DA.WEEKDAY[d['dow'][i]] for i in idx],
            'common': [0 if i < 0 else int(d['term'][i] == '방학' or d['event'][i] in ('휴일', '시험')) for i in idx],
            'tag': [None if i < 0 else (d['event'][i] if d['event'][i] != '없음' else d['term'][i]) for i in idx],
            'steps': col('steps', 0), 'steps_med': col('steps_med', 0), 'steps_sd': col('steps_sd', 0),
            'sleep': col('sleep_min'), 'sleep_med': col('sleep_med'), 'sleep_sd': col('sleep_sd'),
            'bed': col('bed_min'), 'bed_med': col('bed_med'), 'bed_sd': col('bed_sd'),
            # 변화 누적: 누적합을 알림 기준(경보선)으로 나눈 값. 1에 닿으면 알림
            'build': [None if i < 0 or d['S_final'][i] is None else round(d['S_final'][i] / h['final'], 3) for i in idx],
            'build_reg': [None if i < 0 or d['S_reg'][i] is None else round(d['S_reg'][i] / h['reg'], 3) for i in idx],
            'a_steps': flag('a_final'), 'a_reg': flag('a_reg')}


def alert_log(d, k):
    out = []
    for i in range(k, max(-1, k - SPAN), -1):
        tag = d['event'][i] if d['event'][i] != '없음' else d['term'][i]
        for key, cause, text in [('a_final', '걸음', '걸음이 평소보다 적은 날이 이어졌어요'),
                                 ('a_reg', '수면 규칙성', '잠드는 시간이 들쭉날쭉했어요')]:
            if d[key][i]:
                out.append({'when': DA.ago(k - i), 'wd': DA.WEEKDAY[d['dow'][i]], 'cause': cause, 'text': text, 'tag': tag})
    return out


def view_pc(p, day):
    d, k = p['days'], B + day
    v = DA.view_for(p, day)
    v.update({'tl': timeline(p, k), 'weeks': weekly(d, k), 'dev': deviation_rates(d, k), 'log': alert_log(d, k),
              'tag': d['event'][k] if d['event'][k] != '없음' else d['term'][k]})
    return v


def payload(p):
    return {'label': p['label'], 'sample': bool(p.get('sample')),
            'scenes': [{'name': n, 'view': view_pc(p, t)} for n, t in DA.presets(p)]}


# ---------------------------------------------------------------------------
# 화면. 모바일판과 같은 색·아이콘 체계 (따뜻한 종이색, 지표별 색, 주황 = 평소와 다름)
# ---------------------------------------------------------------------------
CSS = """
body.pc-mode header,body.pc-mode .sample-banner{display:none}body.pc-mode main{max-width:none;padding:0}body.pc-mode footer{display:none}
.pc-fit{width:100%;height:calc(100vh - var(--navh,52px));overflow:hidden;background:#e9e5de;display:flex;flex-direction:column}
.pc-fit:fullscreen{height:100vh}.pc-fit:fullscreen .pc-bar{display:none}
.pc-bar{flex:none;display:flex;align-items:center;gap:10px;padding:8px 16px;font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif}
.pc-bar .lab{font-size:13px;color:#78726a;font-weight:700}.pc-bar .sp{flex:1}.pc-bar .hint{font-size:12px;color:#9a938a}
.pc-stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
.pc{--bg:#f6f4f0;--card:#fff;--ink:#23211d;--sub:#78726a;--line:#e6e0d6;--steps:#2f9e6e;--sleep:#6b62c8;--bed:#3a7bd5;--reg:#1d97a6;
--warn:#d98a1c;--warnbg:#fdf3e2;--calm:#2f9e6e;--calmbg:#e8f5ee;--watch:#8a847b;--shade:#efebe4;
width:1600px;height:900px;flex:none;transform-origin:center center;background:var(--bg);color:var(--ink);
font-family:"Pretendard","Apple SD Gothic Neo",-apple-system,"Malgun Gothic","Noto Sans KR",sans-serif;font-size:16px;line-height:1.45;
word-break:keep-all;padding:18px 22px;display:grid;gap:14px;grid-template-rows:56px 1fr 30px;box-shadow:0 10px 40px rgba(0,0,0,.12)}
.pc *{box-sizing:border-box}.pc b{font-weight:800}
.pc-top{display:flex;align-items:center;gap:18px}
.pc-top .nm{font-size:24px;font-weight:800;letter-spacing:-.03em;line-height:1.2}
.pc-top .ttl{padding-left:16px;border-left:1px solid var(--line);font-size:15px;color:var(--sub);font-weight:700}
.pc-top .p-sp{flex:1}.pc-top .ctx{display:flex;gap:8px;font-size:14px;color:#4a463f}
.pc-top .ctx span{border:1px solid var(--line);background:#fff;border-radius:10px;padding:7px 12px}.pc-top .ctx b{margin-left:6px}
.pc-scenes{display:flex;gap:4px;background:#fff;border:1px solid #e6e0d6;border-radius:10px;padding:3px}
.pc-scenes button{border:0;background:none;border-radius:8px;padding:6px 12px;font-size:13px;font-weight:700;font-family:inherit;color:#78726a;cursor:pointer}
.pc-scenes button.on{background:#23211d;color:#fff}
.pc-fs{border:1px solid #e6e0d6;background:#fff;border-radius:10px;padding:7px 12px;font-size:13px;font-weight:700;font-family:inherit;cursor:pointer;color:#23211d}
.pc-body{display:grid;grid-template-columns:360px 1fr 380px;gap:14px;min-height:0}
.col{display:flex;flex-direction:column;gap:14px;min-height:0}
.pnl{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 16px;min-height:0}
.pnl h4{margin:0 0 8px;font-size:15px;font-weight:800;letter-spacing:-.01em;display:flex;align-items:baseline;gap:8px}
.pnl h4 small{font-weight:600;color:var(--sub);font-size:12.5px}
.p-hero{position:relative;padding:16px 16px 16px 22px;overflow:hidden;display:flex;gap:12px}
.p-hero::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px}
.p-hero.attention::before{background:var(--warn)}.p-hero.calm::before{background:var(--calm)}.p-hero.watch::before{background:var(--watch)}
.p-hero .p-ic{flex:none;width:42px;height:42px;border-radius:13px;display:grid;place-items:center}
.p-hero.attention .p-ic{background:var(--warnbg);color:var(--warn)}.p-hero.calm .p-ic{background:var(--calmbg);color:var(--calm)}.p-hero.watch .p-ic{background:#f1efea;color:var(--watch)}
.p-hero .p-ic svg{width:24px;height:24px}.p-hero h3{margin:0 0 4px;font-size:19px;line-height:1.3;letter-spacing:-.02em}.p-hero p{margin:0;color:#4a463f;font-size:14px}
.p-hero .p-ctx{margin-top:8px;font-size:12.5px;color:var(--sub);border:1px dashed #d9d2c6;border-radius:10px;padding:6px 10px}
.p-tiles{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.p-tile{border:1px solid var(--line);border-radius:14px;padding:11px 12px;background:var(--card)}
.p-mi{display:flex;align-items:center;gap:7px;font-size:13.5px;font-weight:700}.p-mi .p-sq{width:24px;height:24px;border-radius:7px;display:grid;place-items:center}.p-mi .p-sq svg{width:15px;height:15px}
.p-tile .p-v{font-size:21px;font-weight:800;margin-top:6px;letter-spacing:-.03em;line-height:1.2}.p-tile .p-u{font-size:12.5px;color:var(--sub)}
.p-chip{display:inline-block;margin-top:6px;border-radius:7px;padding:1px 7px;font-size:12px;font-weight:700;border:1px solid;white-space:nowrap}
.p-chip.ok{border-color:#ddd7cd;color:#5f5a52}.p-chip.low{border-color:#f0c88c;color:#a8600a;background:var(--warnbg)}
.p-chip.mid{border-color:#eedcaa;color:#8a6a10}.p-chip.high{border-color:#bcd3f2;color:#2b62b0}.p-chip.none{border-color:#e5e0d8;color:var(--sub)}
.dev{display:grid;grid-template-columns:86px 1fr 46px;gap:6px 10px;align-items:center;font-size:13.5px}
.dev .bar{position:relative;height:14px;background:#f1eee8;border-radius:7px}
.dev .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:7px}.dev .bar s{position:absolute;top:-3px;bottom:-3px;width:2px;background:#23211d;opacity:.55}
.dev .pct{text-align:right;font-weight:800;font-variant-numeric:tabular-nums}
.dev-leg{display:flex;gap:14px;color:var(--sub);font-size:12px;margin-top:8px}
.dev-leg i{display:inline-block;width:12px;height:9px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.tl{flex:1;display:flex;flex-direction:column}.tl svg{flex:1;width:100%;height:100%;display:block}
.leg{display:flex;gap:16px;flex-wrap:wrap;color:var(--sub);font-size:12.5px;margin-top:4px}
.leg i{display:inline-block;width:14px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.wk svg{width:100%;display:block}
.log{display:flex;flex-direction:column;overflow:hidden;flex:1}
.log .p-row{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid var(--line);align-items:flex-start}.log .p-row:last-child{border-bottom:0}
.log .p-dot{flex:none;width:10px;height:10px;border-radius:50%;margin-top:6px}
.log .p-tx{flex:1;font-size:14px;line-height:1.35}.log .p-tx small{display:block;color:var(--sub);font-size:12px}
.log .p-w{flex:none;color:var(--sub);font-size:13px;white-space:nowrap;text-align:right}
.log .p-empty{color:var(--sub);font-size:14px;padding:6px 0}
.pc-foot{display:flex;justify-content:space-between;align-items:center;color:var(--sub);font-size:12.5px}
.pc-foot b{color:#4a463f}
"""

HTML = """
<div class="pc-fit" id="pcFit">
 <div class="pc-bar"><span class="lab">발표용 장면</span><div class="pc-scenes" id="pcScenes"></div><span class="sp"></span>
  <span class="hint">전체 화면에서는 이 줄이 숨고, 1·2·3 키로 장면을 바꿉니다</span><button class="pc-fs" id="pcFs">전체 화면</button></div>
 <div class="pc-stage" id="pcStage"><div class="pc" id="pc">
 <div class="pc-top">
  <div class="nm">생활 리듬 모니터링</div><div class="ttl">보호자·관리자 화면</div>
  <div class="p-sp"></div>
  <div class="ctx" id="pcCtx"></div>
 </div>
 <div class="pc-body">
  <div class="col">
   <div class="pnl p-hero" id="pcHero"></div>
   <div class="p-tiles" id="pcTiles"></div>
   <div class="pnl" style="flex:1"><h4>개인 기준선 대비 이탈률 <small>평소 범위를 벗어난 날의 비율</small></h4>
    <div class="dev" id="pcDev"></div>
    <div class="dev-leg"><span><i style="background:#d98a1c"></i>최근 7일</span><span><i style="background:#23211d;width:2px;opacity:.55"></i>그 전 3주</span></div>
    <p style="margin:10px 0 0;color:var(--sub);font-size:12.5px">평소 범위 = 지난 4주 중앙값 ± 흔들림 폭. 알림은 하루가 아니라 며칠의 누적으로 판단해서, 이탈률이 낮아도 알림이 뜰 수 있습니다.</p></div>
  </div>
  <div class="col">
   <div class="pnl tl"><h4>최근 8주 기록 <small>지난 4주의 나로 만든 평소 범위와 비교 · 회색 음영은 방학·휴일·시험</small></h4>
    <svg id="pcTl" viewBox="0 0 800 640" preserveAspectRatio="none"></svg>
    <div class="leg"><span><i style="background:#2f9e6e"></i>걸음</span><span><i style="background:#6b62c8"></i>수면 시간</span><span><i style="background:#3a7bd5"></i>잠든 시각</span>
     <span><i style="background:#e4ded3"></i>평소 범위</span><span><i style="background:#d98a1c"></i>알림</span><span><i style="background:repeating-linear-gradient(90deg,#23211d 0 3px,transparent 3px 6px);height:2px;vertical-align:3px"></i>알림 기준</span></div></div>
  </div>
  <div class="col">
   <div class="pnl wk"><h4>주간 리듬 변화 <small>주별 하루 평균</small></h4><svg id="pcWk" viewBox="0 0 350 290"></svg></div>
   <div class="pnl log"><h4>알림 기록 <small>최근 8주</small></h4><div id="pcLog"></div></div>
  </div>
 </div>
 <div class="pc-foot"><span>연구용 시연 화면입니다. <b>의학적 진단이 아니며</b>, "평소와 다름"은 지난 4주의 본인과 비교한 결과입니다.</span><span id="pcSrc"></span></div>
</div></div></div>
"""

JS = r"""
(function(){
const DATA=__DATA__;
let scene=0;
const HEX={steps:'#2f9e6e',sleep:'#6b62c8',bed:'#3a7bd5',reg:'#1d97a6'},TINT={steps:'#e6f4ed',sleep:'#eeecfa',bed:'#e8f0fb',reg:'#e3f3f5'};
const WARN='#d98a1c',BAND='#e4ded3',SHADE='#efebe4',GRID='#efebe4',SUB='#9a938a',INK='#23211d';
const SVG=p=>'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+p+'</svg>';
const ICON={steps:SVG('<path d="M4 16v-2.4C4 11.5 3 10.5 3 8c0-2.7 1.5-6 4.5-6C9.4 2 10 3.8 10 5.5c0 3.1-2 5.7-2 8.7V16a2 2 0 1 1-4 0z"/><path d="M20 20v-2.4c0-2.1 1-3.1 1-5.6 0-2.7-1.5-6-4.5-6-1.9 0-2.5 1.8-2.5 3.5 0 3.1 2 5.7 2 8.7V20a2 2 0 1 0 4 0z"/>'),
 sleep:SVG('<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>'),bed:SVG('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>'),
 reg:SVG('<path d="M3 12c2-5 4-5 6 0s4 5 6 0 4-5 6 0"/>'),
 attention:SVG('<path d="M12 7v6"/><circle cx="12" cy="17" r=".6" fill="currentColor"/><circle cx="12" cy="12" r="9"/>'),
 calm:SVG('<path d="M7.5 12.5l3 3 6-6.5"/><circle cx="12" cy="12" r="9"/>'),watch:SVG('<circle cx="12" cy="12" r="9"/><path d="M8 12h.01M12 12h.01M16 12h.01"/>')};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const rich=s=>esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');
const $=id=>document.getElementById(id);
const V=()=>DATA.scenes[scene].view;
const T=(x,y,s,o)=>'<text x="'+x+'" y="'+y+'" '+(o||'')+'>'+s+'</text>';
const clock=x=>{const t=(Math.round(x)+720)%1440;return Math.floor(t/60)+'시';};
const kfmt=x=>x>=10000?(x/10000).toFixed(x%10000?1:0)+'만':x===0?'0':(x/1000).toFixed(x%1000?1:0)+'천';

function fit(){const f=$('pcStage'),pc=$('pc');if(!document.body.classList.contains('pc-mode'))return;
  const nav=document.querySelector('nav');document.documentElement.style.setProperty('--navh',(nav&&!document.fullscreenElement?nav.offsetHeight:0)+'px');
  const s=Math.min(f.clientWidth/1600,f.clientHeight/900)*0.985;pc.style.transform='scale('+s+')';}

function top(){const v=V();
  const tag=v.tag==='학기'?'학기 중':v.tag==='방학'?'방학 중':v.tag;
  $('pcCtx').innerHTML='<span>대상<b>'+esc(DATA.label)+'</b></span><span>오늘<b>'+esc(v.weekday)+'요일 · '+esc(tag)+'</b></span>';
  $('pcScenes').innerHTML=DATA.scenes.map((s,i)=>'<button class="'+(i===scene?'on':'')+'" data-i="'+i+'">'+esc(s.name)+'</button>').join('');
  $('pcScenes').querySelectorAll('button').forEach(b=>b.onclick=()=>{scene=+b.dataset.i;render();});
  $('pcSrc').textContent=DATA.sample?'가상 예시 인물 · 실제 사람의 기록이 아닙니다':'NetHealth '+DATA.label+' · 날짜는 지우고 요일과 n일 전으로만 표시';}

function left(){const v=V(),h=v.hero;
  $('pcHero').className='pnl p-hero '+h[0];
  $('pcHero').innerHTML='<div class="p-ic">'+ICON[h[0]]+'</div><div><h3>'+esc(h[1])+'</h3><p>'+rich(h[2])+'</p>'+(v.note?'<div class="p-ctx">'+rich(v.note)+'</div>':'')+'</div>';
  $('pcTiles').innerHTML=v.metrics.map(m=>'<div class="p-tile"><div class="p-mi"><span class="p-sq" style="background:'+TINT[m.id]+';color:'+HEX[m.id]+'">'+ICON[m.id]+'</span>'+esc(m.name)+'</div>'+
    '<div class="p-v">'+esc(m.value)+'</div><div class="p-u">'+rich(m.usual)+'</div>'+(m.value===m.lv[1]?'':'<span class="p-chip '+m.lv[0]+'">'+esc(m.lv[1])+'</span>')+'</div>').join('');
  $('pcDev').innerHTML=v.dev.map(r=>{const now=r.now??0;
    return '<span>'+esc(r.name)+'</span><div class="bar"><i style="width:'+now+'%;background:'+(now>=50?WARN:'#e8b766')+'"></i>'+
      (r.before!=null?'<s style="left:calc('+r.before+'% - 1px)"></s>':'')+'</div><span class="pct">'+(r.now==null?'–':r.now+'%')+'</span>';}).join('');}

/* 최근 8주: 걸음 · 변화 누적 · 수면 시간 · 잠든 시각을 같은 날짜 축에 위아래로 */
function timeline(){const t=V().tl,n=t.back.length,W=800,L=58,R=12,cw=(W-L-R)/n,cx=i=>L+cw*i+cw/2;
  const lanes=[{k:'steps',h:200,title:'걸음'},{k:'build',h:100,title:'변화 누적'},{k:'sleep',h:135,title:'수면 시간'},{k:'bed',h:135,title:'잠든 시각'}];
  let g='',y0=8;
  // 방학·휴일·시험 음영 (전 구간 공통)
  const H=lanes.reduce((a,l)=>a+l.h+18,0);
  t.common.forEach((c,i)=>{if(c)g+='<rect x="'+(L+cw*i)+'" y="0" width="'+cw+'" height="'+(H-4)+'" fill="'+SHADE+'"/>';});
  lanes.forEach(l=>{const top=y0+24,bot=y0+l.h;
    g+=T(4,y0+12,l.title,'font-size="13" font-weight="800" fill="'+INK+'"');
    let vals,lo,hi,fmt;
    if(l.k==='build'){vals=t.build;lo=0;hi=Math.max(1.2,...vals.filter(x=>x!=null),...t.build_reg.filter(x=>x!=null));fmt=x=>'';}
    else{const v=t[l.k],md=t[l.k+'_med'],sd=t[l.k+'_sd'];vals=v;
      const all=[].concat(v,md.map((x,i)=>x!=null&&sd[i]!=null?x+sd[i]:null),md.map((x,i)=>x!=null&&sd[i]!=null?x-sd[i]:null)).filter(x=>x!=null);
      lo=l.k==='bed'?Math.min(...all):0;hi=Math.max(...all);if(l.k!=='bed')hi*=1.05;
      fmt=l.k==='steps'?kfmt:l.k==='sleep'?(x=>Math.round(x/60)+'시간'):clock;}
    const y=x=>bot-(x-lo)/((hi-lo)||1)*(bot-top);
    // 눈금 3개
    if(l.k!=='build')[lo,(lo+hi)/2,hi].forEach(val=>{g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(val)+'" y2="'+y(val)+'" stroke="'+GRID+'"/>'+T(L-6,y(val)+4,fmt(val),'font-size="11" text-anchor="end" fill="'+SUB+'"');});
    if(l.k==='build'){
      g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(1)+'" y2="'+y(1)+'" stroke="'+INK+'" stroke-dasharray="4 4" opacity=".6"/>'+T(L-6,y(1)+4,'알림','font-size="11" text-anchor="end" fill="'+INK+'"');
      g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+bot+'" y2="'+bot+'" stroke="'+GRID+'"/>';
      [['build',HEX.steps],['build_reg',HEX.reg]].forEach(([k,c])=>{let p='';t[k].forEach((x,i)=>{if(x==null){p+=' ';return;}p+=(p&&!p.endsWith(' ')?' L':' M')+cx(i)+' '+y(Math.min(x,hi));});
        g+='<path d="'+p.trim()+'" fill="none" stroke="'+c+'" stroke-width="2.2"/>';});
      g+=T(70,y0+12,'<tspan fill="'+HEX.steps+'">━ 걸음</tspan>   <tspan fill="'+HEX.reg+'">━ 수면 규칙성</tspan>   <tspan fill="'+SUB+'" font-weight="400">평소와 다른 날이 쌓인 정도. 알림 기준에 닿으면 알림</tspan>','font-size="11" font-weight="700"');
    }else{const md=t[l.k+'_med'],sd=t[l.k+'_sd'];let up='',dn='';
      md.forEach((x,i)=>{if(x==null||sd[i]==null)return;up+=(up?' L':'M')+cx(i)+' '+y(Math.min(hi,x+sd[i]));});
      for(let i=n-1;i>=0;i--){if(md[i]==null||sd[i]==null)continue;dn+=' L'+cx(i)+' '+y(Math.max(lo,md[i]-sd[i]));}
      if(up)g+='<path d="'+up+dn+' Z" fill="'+BAND+'" opacity=".75"/>';
      const c=HEX[l.k];
      vals.forEach((x,i)=>{if(x==null)return;const last=i===n-1;
        if(l.k==='bed')g+='<circle cx="'+cx(i)+'" cy="'+y(x)+'" r="'+(last?4.5:3)+'" fill="'+c+'" opacity="'+(last?1:.7)+'"/>';
        else{const bw=Math.max(3,cw*.62);g+='<rect x="'+(cx(i)-bw/2)+'" y="'+y(x)+'" width="'+bw+'" height="'+Math.max(1,bot-y(x))+'" rx="2" fill="'+c+'" opacity="'+(last?1:.62)+'"/>';}});
    }
    // 알림 표시: 걸음 칸과 변화 누적 칸 위에
    if(l.k==='steps'||l.k==='build')t.back.forEach((b,i)=>{const a=(l.k==='steps'&&t.a_steps[i])||(l.k==='build'&&(t.a_steps[i]||t.a_reg[i]));
      if(a)g+='<path d="M'+cx(i)+' '+(top-1)+' l-6 -9 h12 z" fill="'+WARN+'"/>';});
    y0+=l.h+18;});
  // 날짜 축: 주 단위 + 오늘
  t.back.forEach((b,i)=>{if(b%7&&b!==0)return;const lab=b===0?'오늘':b+'일 전';
    g+='<line x1="'+cx(i)+'" x2="'+cx(i)+'" y1="0" y2="'+(y0-8)+'" stroke="'+(b===0?INK:'#ddd6cb')+'" stroke-dasharray="'+(b===0?'':'2 3')+'" opacity="'+(b===0?.35:1)+'"/>'+
    T(cx(i),y0+6,lab,'font-size="12" text-anchor="middle" fill="'+(b===0?INK:SUB)+'" font-weight="'+(b===0?800:400)+'"');});
  const svg=$('pcTl');svg.setAttribute('viewBox','0 0 '+W+' '+(y0+12));svg.innerHTML=g;}

/* 주간 리듬: 주별 평균 걸음 막대 + 평소 값 점선 + 주별 수면 시간 점, 아래에 알림·맥락 */
function weeks(){const w=V().weeks,n=w.length,W=350,L=40,R=40,top=36,bot=206,cw=(W-L-R)/Math.max(n,1),cx=i=>L+cw*i+cw/2;
  const sv=w.flatMap(x=>[x.steps,x.steps_med]).filter(x=>x!=null),hi=Math.max(...sv)*1.1;
  const sl=w.flatMap(x=>[x.sleep,x.sleep_med]).filter(x=>x!=null),slo=Math.min(...sl)-30,shi=Math.max(...sl)+30;
  const y=x=>bot-x/hi*(bot-top),ys=x=>bot-(x-slo)/(shi-slo)*(bot-top);
  let g='';
  [0,hi/2,hi].forEach(val=>{g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(val)+'" y2="'+y(val)+'" stroke="'+GRID+'"/>'+T(L-5,y(val)+4,kfmt(Math.round(val/100)*100),'font-size="10.5" text-anchor="end" fill="'+SUB+'"');});
  [slo+30,shi-30].forEach(val=>{g+=T(W-R+5,ys(val)+4,(val/60).toFixed(1)+'h','font-size="10.5" fill="'+HEX.sleep+'"');});
  w.forEach((r,i)=>{const bw=cw*.56,last=i===n-1;
    if(r.steps!=null)g+='<rect x="'+(cx(i)-bw/2)+'" y="'+y(r.steps)+'" width="'+bw+'" height="'+(bot-y(r.steps))+'" rx="3" fill="'+HEX.steps+'" opacity="'+(last?1:.55)+'"/>';
    if(r.steps_med!=null)g+='<line x1="'+(cx(i)-cw/2+2)+'" x2="'+(cx(i)+cw/2-2)+'" y1="'+y(r.steps_med)+'" y2="'+y(r.steps_med)+'" stroke="'+INK+'" stroke-width="1.6" stroke-dasharray="3 2" opacity=".55"/>';
    if(r.alerts)g+='<path d="M'+cx(i)+' '+(top-4)+' l-6 -9 h12 z" fill="'+WARN+'"/>';
    g+=T(cx(i),bot+16,r.label,'font-size="10.5" text-anchor="middle" fill="'+(last?INK:SUB)+'" font-weight="'+(last?800:400)+'"');
    if(r.context)g+=T(cx(i),bot+30,r.context,'font-size="10" text-anchor="middle" fill="'+SUB+'"');});
  let p='';w.forEach((r,i)=>{if(r.sleep==null)return;p+=(p?' L':'M')+cx(i)+' '+ys(r.sleep);});
  g+='<path d="'+p+'" fill="none" stroke="'+HEX.sleep+'" stroke-width="2"/>';
  w.forEach((r,i)=>{if(r.sleep!=null)g+='<circle cx="'+cx(i)+'" cy="'+ys(r.sleep)+'" r="3.5" fill="#fff" stroke="'+HEX.sleep+'" stroke-width="2"/>';});
  g+=T(L,12,'<tspan fill="'+HEX.steps+'">■ 걸음</tspan>  <tspan fill="'+INK+'" opacity=".6">┄ 평소 걸음</tspan>  <tspan fill="'+HEX.sleep+'">● 수면 시간</tspan>  <tspan fill="'+WARN+'">▼ 알림</tspan>','font-size="11" font-weight="700"');
  const last=w[n-1],prev=w[n-2];
  if(last&&prev&&last.steps!=null&&prev.steps!=null){const d=Math.round((last.steps-prev.steps)/prev.steps*100);
    g+=T(4,bot+56,'이번 주 걸음은 지난주보다 <tspan font-weight="800" fill="'+(d<0?WARN:INK)+'">'+(d>0?'+':'')+d+'%</tspan>','font-size="13.5" fill="'+INK+'"');
    if(last.reg!=null)g+=T(4,bot+76,'최근 7일 잠드는 시간 차이 <tspan font-weight="800">'+Math.round(last.reg)+'분</tspan> <tspan fill="'+SUB+'">(65분 미만이면 일정한 편)</tspan>','font-size="13.5" fill="'+INK+'"');}
  $('pcWk').innerHTML=g;}

function log(){const l=V().log;
  $('pcLog').innerHTML=l.length?l.slice(0,6).map(a=>'<div class="p-row"><span class="p-dot" style="background:'+(a.cause==='걸음'?WARN:HEX.reg)+'"></span>'+
    '<div class="p-tx">'+esc(a.text)+'<small>원인 지표: '+esc(a.cause)+' · 그때: '+esc(a.tag)+'</small></div><div class="p-w">'+esc(a.when)+'<br>'+esc(a.wd)+'요일</div></div>').join('')
    :'<div class="p-empty">최근 8주 동안 알림이 없었어요.</div>';}

function render(){top();left();timeline();weeks();log();fit();}
$('pcFs').onclick=()=>{const f=$('pcFit');document.fullscreenElement?document.exitFullscreen():f.requestFullscreen&&f.requestFullscreen();};
document.addEventListener('fullscreenchange',()=>{$('pcFs').textContent=document.fullscreenElement?'전체 화면 끝내기':'전체 화면';setTimeout(fit,50);});
window.addEventListener('resize',fit);
document.addEventListener('keydown',e=>{const i=+e.key-1;if(!document.body.classList.contains('pc-mode')||!(i>=0&&i<DATA.scenes.length))return;scene=i;render();});
window.pcFit=fit;
render();
})();
"""


def render(p):
    """PC 데모 조각 (CSS, 본문, 스크립트). build.py가 'PC 대시보드' 탭에 넣는다."""
    data = payload(p)
    return CSS, HTML, JS.replace('__DATA__', json.dumps(data, ensure_ascii=False)), data
