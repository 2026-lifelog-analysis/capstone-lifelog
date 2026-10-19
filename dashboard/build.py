# 작성 2026-10-20 11:00
"""
대시보드 HTML 한 장 만들기. 서버 없이 더블클릭으로 열리고, 그림 라이브러리를 파일 안에 넣어 오프라인에서도 작동한다.

  발표용 : 참가자 A의 개인 기록 포함 (발표장 화면 전용. 깃·공개 링크에 올리지 않음)
  공유용 : 개인 기록 없이 집계 결과만

실행 (레포 맨 위에서)
  python3 -I dashboard/export_person.py <person.json>                     # 참가자 A 기록 (발표용일 때만 필요)
  python3 -I dashboard/build.py 발표용 <person.json> <출력.html>
  python3 -I dashboard/build.py 공유용 - <출력.html>

절을 추가하려면: figures_*.py에 그림 함수를 만들고 아래 research_sections() / monitor_sections()에 한 줄 넣는다.
"""
import datetime as dt
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.offline                   # noqa: E402

import figures_monitor as FM            # noqa: E402
import figures_research as FR           # noqa: E402
import results_data as RD               # noqa: E402

VERSION = RD.CHANGELOG[-1][0]
STATUS = {'done': ('완료', 'st-done'), 'doing': ('진행 중', 'st-doing'), 'final': ('기말 예정', 'st-final')}


def fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displaylogo': False, 'responsive': True})


def section(sid, title, status, body, note='', source='', badge=''):
    label, cls = STATUS[status]
    b = f'<span class="badge inj">{badge}</span>' if badge else ''
    src = f'<p class="source">출처: {html.escape(source)}</p>' if source else ''
    nt = f'<p class="note">{note}</p>' if note else ''
    return (f'<section id="{sid}"><div class="sec-head"><h2>{title}</h2>'
            f'<span class="badge {cls}">{label}</span>{b}</div>{body}{nt}{src}</section>')


def cards(items):
    return '<div class="cards">' + ''.join(
        f'<div class="card"><div class="card-k">{k}</div><div class="card-v">{v}</div></div>' for k, v in items) + '</div>'


def private_note():
    return '<div class="placeholder">개인 기록은 발표용 빌드에서만 보입니다.</div>'


def research_sections(p):
    out = []
    out.append(section('r1', 'R1. 한눈에', 'done', cards([
        ('목표', '웨어러블 걸음·수면 기록에서 <b>그 사람의 평소와 달라지는 변화</b>를 일찍 알아차린다'),
        ('정의: 일상생활 이탈', '개인 평소(직전 28일) 대비 활동·수면 패턴의 지속적 변화. 건강 악화 진단이 아님'),
        ('관찰 축', '걸음 수 감소 · 수면 시간 감소 · 취침 시각 지연 · 수면 규칙성 저하'),
        ('범위 선정 기준', '① 본인이 알아차리기 어려운가 ② 수일~수주 안에 신호가 변하는가 ③ 변화가 악화에 앞선다는 개인 내 근거가 있는가'),
    ]) + '<div class="flow"><div class="step done">모듈 1<br><small>기준선 이탈 탐지 · NetHealth 579명</small></div>'
         '<div class="arrow">→</div><div class="step doing">모듈 2<br><small>설문·의학 기준으로 상태와의 관련 검증</small></div>'
         '<div class="arrow">→</div><div class="step final">대시보드<br><small>보호자·관리자용 화면</small></div></div>'))

    body = fig_html(FR.baseline_example(p)) if p else private_note()
    out.append(section('r2', 'R2. 어떻게 탐지하나', 'done', body,
                       note=('개인 기준선 = 그 사람의 직전 28일 중앙값과 흔들림 폭(MAD). 집단 기준선 = 같은 날 다른 학생 30명 이상. '
                             '매일의 이탈 점수를 누적합(CUSUM)으로 쌓아, 하루 경보율이 1/30이 되는 경보선을 넘으면 경보. '
                             f"예시는 {p['label']} (선정 규칙: {p['rule']})" if p else ''),
                       source='레포 함수 experiments/final.py personal_bad, src/baseline.py cusum · dashboard/export_person.py'))

    body = (f'<div class="grid2"><div>{fig_html(FR.holiday_bars())}</div><div>{fig_html(FR.calendar_bars())}</div></div>'
            + (fig_html(FR.person_methods(p)) if p else ''))
    c = RD.CALENDAR_ALARM
    out.append(section('r3', 'R3. 개인 기준선 vs 집단 기준선', 'done', body,
                       note=('휴일처럼 <b>모두가 함께</b> 덜 걷는 날, 개인 기준선은 경보를 많이 울리지만 대부분 일정 변화다. '
                             '집단 기준선은 같은 날 다른 학생과 비교해 이를 상쇄한다. 그래서 개인 기준선에서 같은 날 공통 변화를 빼는 보정을 더했다. '
                             f"휴일 경보율이 {c['base'][0]}% → {c['final'][0]}%로 줄고, 상위 10% 사람의 경보 몫도 "
                             f"{c['concentration']['base']}% → {c['concentration']['final']}%로 줄었다. "
                             f"대신 개강주 경보율은 {c['base'][3]}% → {c['final'][3]}%로 늘었다."),
                       source=f"{RD.HOLIDAY_PERSONAL_VS_GROUP['source']} · {c['source']}"))

    body = (f'<div class="grid2"><div>{fig_html(FR.ladder())}</div><div>{fig_html(FR.det_fa())}</div></div>'
            + fig_html(FR.heatmap()))
    out.append(section('r4', 'R4. 방법 개선과 통합 비교', 'done', body, badge='합성 주입 · 동작 점검',
                       note=('실제 기록에 인위적인 변화(걸음 −30%, 잠든 시각 +120분 등)를 14일 넣고, 그 안에 경보가 한 번이라도 울리면 탐지, '
                             '주입하지 않은 같은 구간에서 울리면 오경보로 셌다. 30%·120분은 의학 기준이 아니라 알고리즘의 반응을 시험하기 위한 변화량이다. '
                             '평균이 움직이는 변화에는 개선 방법이, 평균은 그대로이고 흔들림만 커지는 변화에는 수면 리듬 지표가 가장 높다.'),
                       source=f"{RD.INTEGRATED_SOURCE} · {RD.DET_FA['source']}"))

    body = ('<div class="grid2"><div><h3>현재 결과의 한계</h3><ol>'
            '<li>특정 방향의 변화를 주입해, 그 방향을 누적하는 방법에 유리했을 수 있다. 걸음을 줄여 넣고 걸음으로 잡은 순환 구조다.</li>'
            '<li>28일·30% 같은 설정값은 의학적으로 검증된 이탈 기준이 아니다. 실제 우울 진단 직전 걸음 감소는 약 5%로 보고된다 (Zhang 2026).</li>'
            '<li>주입 탐지 성능만으로 실제 상태와의 관련을 입증할 수 없고, 같은 자료의 재분석이라 독립 검증도 아니다.</li></ol></div>'
            '<div><h3>기말 검증 구조</h3><div class="vflow">'
            '<div class="vbox">숨은 상태 (예: 컨디션 저하)</div><div class="vsplit">'
            '<div class="vbox small">행동 변화<br><small>선행연구의 크기·시간 경과</small><br>↓<br><b>탐지기 입력</b></div>'
            '<div class="vbox small">설문 · 의학 기준<br><small>탐지에 쓰지 않은 지표</small><br>↓<br><b>정답</b></div>'
            '</div><p class="note">탐지에 사용한 지표로는 정답을 만들지 않는다.</p></div></div></div>'
            + fig_html(FR.label_table()))
    out.append(section('r5', 'R5. 한계와 다음 단계', 'doing', body, source=RD.LABELS['source']))

    body = '<div class="cards">' + ''.join(
        f'<div class="card"><div class="card-k">{s["tag"]}</div><div class="card-v"><b>{s["cite"]}</b> · {s["n"]}'
        f'<br>{s["finding"]}</div></div>' for s in RD.STUDIES) + '</div>'
    out.append(section('r6', 'R6. 근거가 된 선행연구', 'done', body,
                       note='특정 센서 값으로 질병을 진단할 수 있다는 뜻이 아니라, 평소 대비 변화를 관찰할 필요성을 뒷받침한다.'))
    out.append(section('r7', 'R7. 검증 결과', 'final',
                       '<div class="placeholder">문헌 기반 시나리오 검증 · 설문 라벨과 경보 비교 (기말)</div>'))
    return out


def monitor_sections(p):
    out = ['<div class="disclaimer">연구용 시연 화면입니다. 의학적 진단이 아니며, "평소와 다름"은 그 사람의 최근 28일과 비교한 결과입니다.</div>']
    if not p:
        return out + [private_note()]
    out.append(section('d1', f"D1. 오늘 상태 · {p['label']}", 'done',
                       '<div class="day-pick"><label>날짜 선택 <input type="range" id="daySlider" min="0" max="55"></label>'
                       '<span id="dayLabel"></span></div><div id="statusLine" class="status"></div>'
                       '<div class="cards four" id="dayCards"></div>'))
    out.append(section('d2', 'D2. 최근 8주 타임라인', 'done', fig_html(FM.timeline(p)),
                       note='파란 선 = 평소(직전 28일 중앙값), 주황 삼각형 = 평소와 다름. 음영은 방학·휴일·시험 기간.'))
    for sid, title, desc in [('d3', 'D3. 주간 리듬 변화', '주별 평균과 수면 규칙성의 추세 (원 제안서의 주간 생활패턴 리듬 변화)'),
                             ('d4', 'D4. 경보 기록', '경보 날짜 · 원인 지표 · 그날의 맥락'),
                             ('d5', 'D5. 설문 겹쳐 보기', '팀원 파일럿의 피로·기분·수면 만족도를 타임라인에 겹침'),
                             ('d6', 'D6. 장기 추세', '월 단위 활동 리듬 진폭·낮잠 추세 (인지 저하 연결, 경보 없음)')]:
        out.append(section(sid, title, 'final', f'<div class="placeholder">{desc}</div>'))
    return out


CSS = """
:root{--bg:#f8fafc;--panel:#fff;--ink:#0f172a;--muted:#475569;--line:#e2e8f0;--blue:#2563eb;--navy:#1e3a8a;--orange:#ea580c;--purple:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Pretendard,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;font-size:16px;line-height:1.6}
header{background:var(--navy);color:#fff;padding:20px 32px}header h1{margin:0;font-size:24px}header .meta{opacity:.8;font-size:14px}
nav{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid var(--line);padding:0 32px;display:flex;gap:4px}
nav button{border:0;background:none;padding:14px 20px;font-size:17px;font-weight:600;color:var(--muted);cursor:pointer;border-bottom:3px solid transparent;font-family:inherit}
nav button.on{color:var(--navy);border-bottom-color:var(--navy)}
main{max-width:1280px;margin:0 auto;padding:24px 32px 60px}.tab{display:none}.tab.on{display:block}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin:0 0 22px}
.sec-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}h2{margin:0;font-size:21px}h3{font-size:17px;margin:8px 0}
.badge{font-size:13px;padding:2px 10px;border-radius:999px;font-weight:600}.st-done{background:#dcfce7;color:#166534}
.st-doing{background:#fef9c3;color:#854d0e}.st-final{background:#e2e8f0;color:#334155}.inj{background:#ffedd5;color:#9a3412}
.note{color:var(--muted);margin:10px 0 0}.source{color:#94a3b8;font-size:13px;margin:6px 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:14px}.cards.four{grid-template-columns:repeat(4,1fr)}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:#fbfdff}.card-k{font-size:14px;color:var(--muted);font-weight:600}
.card-v{font-size:16px;margin-top:4px}.big{font-size:30px;font-weight:700}.up{color:var(--orange)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.flow{display:flex;align-items:stretch;gap:10px;margin-top:16px}.step{flex:1;text-align:center;padding:14px;border-radius:10px;font-weight:700;border:2px solid var(--line)}
.step.done{border-color:#86efac}.step.doing{border-color:#fde047}.step.final{border-color:#cbd5e1}.step small{font-weight:400;color:var(--muted)}.arrow{align-self:center;font-size:24px;color:var(--muted)}
.vflow{text-align:center}.vbox{border:2px solid var(--line);border-radius:10px;padding:10px;margin:6px auto;font-weight:600;background:#fbfdff}
.vsplit{display:flex;gap:10px}.vsplit .vbox{flex:1;font-weight:400}
.placeholder{border:2px dashed var(--line);border-radius:10px;padding:26px;text-align:center;color:#94a3b8;margin-top:12px}
.disclaimer{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:10px;padding:12px 16px;margin-bottom:18px;font-weight:600}
.day-pick{display:flex;align-items:center;gap:14px;margin-top:12px}.day-pick input{width:360px}
.status{margin-top:12px;font-size:18px;font-weight:700}.status.alert{color:var(--orange)}.status.calm{color:#166534}
footer{color:var(--muted);font-size:14px;max-width:1280px;margin:0 auto;padding:0 32px 40px}
@media (max-width:900px){.grid2,.cards.four{grid-template-columns:1fr}.flow{flex-direction:column}}
"""

JS = """
function showTab(id){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id===id));
document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
window.dispatchEvent(new Event('resize'));}
const P = %s;
function clock(m){if(m==null)return '-';let t=(Math.round(m)+720)%%1440;return String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%%60).padStart(2,'0');}
function pct(v,m){if(v==null||m==null||m===0)return null;return Math.round((v/m-1)*100);}
function fmt(n,u){if(n==null)return '기록 없음';return (n>0?'+':'')+n+u;}
function renderDay(i){if(!P)return;const d=P.days,k=i+28;
const term=d.term[k]||'-',ev=d.event[k]&&d.event[k]!=='없음'?' · '+d.event[k]:'';
document.getElementById('dayLabel').textContent=i+'일차 · 연구 '+(d.study_week[k]??'-')+'주차 · '+term+ev;
const st=pct(d.steps[k],d.steps_med[k]),sl=pct(d.sleep_min[k],d.sleep_med[k]);
const bd=(d.bed_min[k]!=null&&d.bed_med[k]!=null)?Math.round(d.bed_min[k]-d.bed_med[k]):null;
const rg=d.mid_sd[k]!=null?Math.round(d.mid_sd[k]):null,rgm=d.mid_sd_med[k]!=null?Math.round(d.mid_sd_med[k]):null;
const C=[['걸음','평소 대비',fmt(st,'%%'),st!=null&&st<=-20],['수면 시간','평소 대비',fmt(sl,'%%'),sl!=null&&sl<=-15],
['잠든 시각',clock(d.bed_min[k])+' (평소 '+clock(d.bed_med[k])+')',fmt(bd,'분'),bd!=null&&bd>=60],
['수면 규칙성','7일 흔들림 (평소 '+(rgm??'-')+'분)',rg==null?'기록 없음':rg+'분',rg!=null&&rgm!=null&&rg>rgm*1.5]];
document.getElementById('dayCards').innerHTML=C.map(c=>'<div class="card"><div class="card-k">'+c[0]+'</div><div class="big'+(c[3]?' up':'')+'">'+c[2]+'</div><div class="card-v">'+c[1]+'</div></div>').join('');
const s=document.getElementById('statusLine'),why=[];
if(d.a_final[k])why.push('걸음 감소 누적');if(d.a_reg[k])why.push('수면 규칙성 저하');
const acc=d.S_final[k]!=null?d.S_final[k]/P.h.final:0;
if(why.length){s.className='status alert';s.textContent='● 평소와 다름 — 주된 원인: '+why.join(', ');}
else if(acc>=0.5){s.className='status alert';s.textContent='◐ 걸음 감소가 누적되는 중 (경보선의 '+Math.round(acc*100)+'%%)';}
else{s.className='status calm';s.textContent='○ 평소 범위';}}
document.addEventListener('DOMContentLoaded',()=>{const sl=document.getElementById('daySlider');
if(sl&&P){sl.value=P.alarm_day;renderDay(P.alarm_day);sl.addEventListener('input',e=>renderDay(+e.target.value));}});
"""


def main():
    mode, person, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if mode not in ('발표용', '공유용'):
        raise SystemExit('모드는 발표용 또는 공유용')
    p = json.loads(Path(person).read_text()) if mode == '발표용' else None
    today = dt.date.today().isoformat()
    research = ''.join(research_sections(p))
    monitor = ''.join(monitor_sections(p))
    log = ''.join(f'<li>{v} ({d}) {t}</li>' for v, d, t in RD.CHANGELOG)
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>일상생활 이탈 탐지</title>
<style>{CSS}</style><script>{plotly.offline.get_plotlyjs()}</script></head><body>
<header><h1>라이프로그 기반 개인별 일상생활 이탈 탐지</h1>
<div class="meta">동국대 컴퓨터·AI학부 종합설계 · {VERSION} · {today} 생성 · {mode}</div></header>
<nav><button data-t="tab-r" class="on" onclick="showTab('tab-r')">연구</button>
<button data-t="tab-m" onclick="showTab('tab-m')">모니터링 데모</button></nav>
<main><div id="tab-r" class="tab on">{research}</div><div id="tab-m" class="tab">{monitor}</div></main>
<footer><b>변경 이력</b><ul>{log}</ul></footer>
<script>{JS % json.dumps(p, ensure_ascii=False) if p else JS % 'null'}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding='utf-8')
    print(f'{mode} 대시보드 저장: {out} ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == "__main__":
    main()
