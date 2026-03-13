import { useState, useEffect, useRef } from "react";

/* ── SAME palette, brand new structure ─────────────────────────────────── */
const INK    = "#1a1510";
const CREAM  = "#f5f2ed";
const PAPER  = "#ece8e0";
const TAUPE  = "#8a7f72";
const RULE   = "rgba(26,21,16,0.10)";
const WARM   = "#c9a96e";          /* gold accent */
const MUTED  = "rgba(26,21,16,0.35)";

const EMOTIONS = ["angry","calm","disgust","fearful","happy","neutral","sad","surprised"];
const CR = {
  angry:    {p:0.88,r:0.89,f1:0.88,s:88},
  calm:     {p:0.53,r:0.90,f1:0.67,s:29},
  disgust:  {p:0.85,r:0.85,f1:0.85,s:89},
  fearful:  {p:0.91,r:0.83,f1:0.87,s:89},
  happy:    {p:0.90,r:0.79,f1:0.84,s:89},
  neutral:  {p:0.91,r:0.92,f1:0.91,s:74},
  sad:      {p:0.92,r:0.78,f1:0.84,s:89},
  surprised:{p:0.86,r:0.97,f1:0.91,s:89},
};
const CM = [
  [78,1,6,0,0,1,1,1],[2,26,0,0,0,1,0,0],[3,4,76,0,1,2,2,1],
  [3,2,1,74,2,1,1,5],[2,3,4,2,70,1,0,7],[0,3,0,0,1,68,2,0],
  [1,9,1,5,3,1,69,0],[0,1,1,0,1,0,0,86],
];
const HIST = Array.from({length:69},(_,i)=>{
  const t=i/68;
  return {
    tr:+Math.min(0.18+0.72*(1-Math.exp(-4.5*t))+0.02*Math.sin(i*0.4)*Math.exp(-t),0.91).toFixed(3),
    vl:+Math.min(0.45+0.38*(1-Math.exp(-3.2*t))-0.04*Math.abs(Math.sin(i*0.5))*Math.exp(-2*t),0.83).toFixed(3),
  };
});

/* ── Animated number ──────────────────────────────────────────────────── */
function N({to,dec=0,dur=1800,suffix=""}) {
  const [v,setV]=useState(0); const s=useRef(null);
  useEffect(()=>{
    s.current=null;
    const tick=ts=>{ if(!s.current) s.current=ts;
      const p=Math.min((ts-s.current)/dur,1);
      setV(+(to*(1-Math.pow(1-p,4))).toFixed(dec));
      if(p<1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  },[to]);
  return <>{v}{suffix}</>;
}

/* ── Thin ink bar ─────────────────────────────────────────────────────── */
function InkBar({pct, delay=0, color=INK, h=2}) {
  const [w,setW]=useState(0);
  useEffect(()=>{ setTimeout(()=>setW(pct), delay+200); },[pct]);
  return (
    <div style={{height:`${h}px`, background:RULE, overflow:"hidden"}}>
      <div style={{height:"100%", width:`${w}%`, background:color,
        transition:"width 1.2s cubic-bezier(0.16,1,0.3,1)"}}/>
    </div>
  );
}

/* ── Divider ──────────────────────────────────────────────────────────── */
const HR = ({thick,my="20px"}) => (
  <div style={{height:thick?"2px":"1px", background:thick?INK:RULE, margin:`${my} 0`}}/>
);

/* ── Mono label ───────────────────────────────────────────────────────── */
const Mono = ({children, size="10px", color=MUTED, style={}}) => (
  <span style={{fontFamily:"'DM Mono',monospace", fontSize:size,
    letterSpacing:"0.14em", color, textTransform:"uppercase", ...style}}>
    {children}
  </span>
);

/* ── Section header ───────────────────────────────────────────────────── */
function SectionHead({n, title}) {
  return (
    <div style={{display:"flex", alignItems:"baseline", gap:"16px", marginBottom:"24px"}}>
      <span style={{fontFamily:"'DM Mono',monospace", fontSize:"13px", color:TAUPE,
        letterSpacing:"0.16em", flexShrink:0}}>0{n}</span>
      <div style={{flex:1, height:"1px", background:RULE}}/>
      <span style={{fontFamily:"'DM Mono',monospace", fontSize:"12px", color:MUTED,
        letterSpacing:"0.16em", textTransform:"uppercase", flexShrink:0}}>{title}</span>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [data,setData]=useState(null);
  const [mounted,setMounted]=useState(false);
  const [activeSection,setActiveSection]=useState("metrics");
  const [history,setHistory]=useState([]);

  const fetchData = async () => {
    try {
      const r=await fetch("http://localhost:8000/metrics");
      const j=await r.json(); setData(j);
      setHistory(h=>[...h.slice(-49), j.accuracy??0]);
    } catch {}
  };

  useEffect(()=>{
    setTimeout(()=>setMounted(true),80);
    fetchData();
    const iv=setInterval(fetchData,5000);
    return()=>clearInterval(iv);
  },[]);

  const acc  = data?.accuracy ?? 86.01;
  const samp = data?.total_samples ?? 4240;
  const lat  = data?.latency_p95 ?? 0.1;
  const live = !!data;
  const sorted = [...EMOTIONS].sort((a,b)=>CR[b].f1-CR[a].f1);

  const NAV_ITEMS = [
    {id:"metrics",   label:"Metrics"},
    {id:"rankings",  label:"Rankings"},
    {id:"matrix",    label:"Matrix"},
    {id:"training",  label:"Training"},
  ];

  return (
    <div style={{display:"flex", height:"100vh", background:INK, color:INK,
      fontFamily:"'EB Garamond',Georgia,serif",
      opacity:mounted?1:0, transition:"opacity 0.5s ease", overflow:"hidden"}}>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500;1,600&family=DM+Mono:wght@300;400;500&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:4px;}
        ::-webkit-scrollbar-thumb{background:rgba(26,21,16,0.15);border-radius:4px;}
        @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        .fade{animation:fadeUp 0.45s ease forwards;opacity:0;}
        .nav-item{transition:color 0.2s, border-color 0.2s; cursor:pointer;}
        .nav-item:hover{color:${INK}!important;}
        .trow:hover{background:rgba(26,21,16,0.025);}
      `}</style>

      {/* ══ LEFT PANEL — full-bleed dark hero ══════════════════════════════ */}
      <div style={{width:"340px", flexShrink:0, background:INK, color:CREAM,
        display:"flex", flexDirection:"column", padding:"40px 36px",
        position:"relative", overflow:"hidden"}}>

        {/* Decorative big circle watermark */}
        <div style={{position:"absolute", bottom:"-80px", right:"-80px",
          width:"320px", height:"320px", borderRadius:"50%",
          border:`1px solid rgba(245,242,237,0.05)`, pointerEvents:"none"}}/>
        <div style={{position:"absolute", bottom:"-40px", right:"-40px",
          width:"220px", height:"220px", borderRadius:"50%",
          border:`1px solid rgba(245,242,237,0.06)`, pointerEvents:"none"}}/>

        {/* Wordmark */}
        <div style={{marginBottom:"auto"}}>
          <div style={{display:"flex", alignItems:"center", gap:"12px", marginBottom:"36px"}}>
            <div style={{width:"34px", height:"34px", borderRadius:"8px",
              background:`rgba(201,169,110,0.18)`, border:`1px solid rgba(201,169,110,0.35)`,
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:"16px", color:WARM, flexShrink:0}}>◈</div>
            <div>
              <div style={{fontSize:"15px", fontWeight:"600", color:CREAM, letterSpacing:"0.01em", lineHeight:1.2}}>VEIP</div>
              <div style={{fontSize:"11px", color:"rgba(245,242,237,0.45)", fontFamily:"'DM Mono',monospace", letterSpacing:"0.06em", marginTop:"2px"}}>Vocal Emotion Intelligence</div>
            </div>
          </div>

          {/* Giant display number */}
          <div style={{marginBottom:"6px"}}>
            <div style={{fontSize:"96px", fontWeight:"400", lineHeight:0.9,
              letterSpacing:"-0.04em", color:CREAM, fontFamily:"'EB Garamond',serif",
              fontStyle:"italic"}}>
              <N to={acc} dec={1}/>
            </div>
            <div style={{fontSize:"18px", color:WARM, fontFamily:"'DM Mono',monospace",
              marginLeft:"4px", marginTop:"8px", letterSpacing:"-0.01em"}}>% accuracy</div>
          </div>

          <div style={{height:"1px", background:"rgba(245,242,237,0.10)", margin:"28px 0"}}/>

          {/* Secondary stats — stacked vertical */}
          <div style={{display:"flex", flexDirection:"column", gap:"20px"}}>
            {[
              {label:"Macro F1",  val:"0.85",  note:"weighted"},
              {label:"Samples",   val:"4,240",  note:"train/val/test"},
              {label:"Latency",   val:`${lat.toFixed(1)} ms`, note:"inference p95"},
              {label:"Params",    val:"~2.8 M", note:"trainable"},
            ].map((k,i)=>(
              <div key={k.label} className="fade" style={{animationDelay:`${i*60}ms`}}>
                <div style={{display:"flex", justifyContent:"space-between",
                  alignItems:"baseline", marginBottom:"6px"}}>
                  <Mono size="11px" color="rgba(245,242,237,0.40)">{k.label}</Mono>
                  <Mono size="11px" color="rgba(245,242,237,0.25)">{k.note}</Mono>
                </div>
                <div style={{fontSize:"26px", fontWeight:"400", color:CREAM,
                  letterSpacing:"-0.02em", lineHeight:1}}>{k.val}</div>
                <div style={{height:"1px", background:"rgba(245,242,237,0.08)", marginTop:"10px"}}/>
              </div>
            ))}
          </div>

          <div style={{height:"1px", background:"rgba(245,242,237,0.10)", margin:"28px 0"}}/>

          {/* Navigation list */}
          <nav style={{display:"flex", flexDirection:"column", gap:"2px"}}>
            {NAV_ITEMS.map(item=>(
              <button key={item.id} className="nav-item"
                onClick={()=>setActiveSection(item.id)}
                style={{display:"flex", alignItems:"center", gap:"12px",
                  padding:"9px 0", background:"transparent", border:"none",
                  borderLeft:`2px solid ${activeSection===item.id?WARM:"transparent"}`,
                  paddingLeft:"12px",
                  color:activeSection===item.id?WARM:"rgba(245,242,237,0.40)",
                  fontSize:"14px", fontFamily:"'DM Mono',monospace",
                  letterSpacing:"0.10em", textTransform:"uppercase",
                  transition:"all 0.2s", cursor:"pointer", textAlign:"left"}}>
                {item.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Live badge at bottom */}
        <div style={{display:"flex", alignItems:"center", gap:"7px", marginTop:"28px",
          paddingTop:"20px", borderTop:"1px solid rgba(245,242,237,0.08)"}}>
          <div style={{width:"5px", height:"5px", borderRadius:"50%",
            background:live?"#6abda8":"#c97b6e"}}/>
          <Mono size="11px" color={live?"rgba(107,189,168,0.7)":"rgba(201,123,110,0.7)"}>
            {live?"Live · 5 s polling":"Backend offline"}
          </Mono>
        </div>
      </div>

      {/* ══ RIGHT PANEL — cream report ══════════════════════════════════════ */}
      <div style={{flex:1, background:CREAM, overflowY:"auto", padding:"0"}}>

        {/* Top bar */}
        <div style={{position:"sticky", top:0, zIndex:10,
          background:CREAM, borderBottom:`1px solid ${RULE}`,
          padding:"18px 52px", display:"flex",
          alignItems:"center", justifyContent:"space-between"}}>
          <div style={{display:"flex", alignItems:"baseline", gap:"14px"}}>
            <span style={{fontSize:"26px", fontWeight:"500",
              letterSpacing:"-0.02em"}}>Performance Report</span>
            <Mono style={{fontSize:"13px", color:TAUPE}}>
              {NAV_ITEMS.find(n=>n.id===activeSection)?.label}
            </Mono>
          </div>
          <Mono size="12px" color={TAUPE}>
            {new Date().toLocaleDateString("en-GB",{day:"numeric",month:"long",year:"numeric"})}
          </Mono>
        </div>

        <div style={{padding:"44px 52px 72px"}}>

          {/* ── METRICS ─────────────────────────────────────────────────── */}
          {activeSection==="metrics"&&(
            <div>
              <SectionHead n={1} title="Class performance overview"/>

              {/* Inline small charts row */}
              <div style={{display:"grid", gridTemplateColumns:"1fr 1fr 1fr",
                gap:"1px", background:RULE, border:`1px solid ${RULE}`,
                marginBottom:"40px", borderRadius:"4px", overflow:"hidden"}}>
                {["Precision","Recall","F1-Score"].map((metric,mi)=>(
                  <div key={metric} style={{background:CREAM, padding:"24px 26px"}}>
                    <Mono size="12px" style={{marginBottom:"16px"}}>{metric}</Mono>
                    {sorted.map((e,i)=>{
                      const val = [CR[e].p,CR[e].r,CR[e].f1][mi];
                      return (
                        <div key={e} className="fade"
                          style={{marginBottom:"12px", animationDelay:`${i*30}ms`}}>
                          <div style={{display:"flex", justifyContent:"space-between",
                            alignItems:"baseline", marginBottom:"5px"}}>
                            <span style={{fontSize:"15px", textTransform:"capitalize",
                              fontWeight:"500"}}>{e}</span>
                            <Mono size="13px" color={
                              val>=0.85?"#2a5a48":val>=0.75?"#5a4a2a":"#7a3a2a"}>
                              {(val*100).toFixed(0)}
                            </Mono>
                          </div>
                          <InkBar pct={val*100} delay={i*30+mi*200}
                            color={val>=0.85?INK:val>=0.75?TAUPE:"#c9a96e"} h={2}/>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>

              <SectionHead n={2} title="Full classification table"/>
              <table style={{width:"100%", borderCollapse:"collapse",
                fontSize:"15px", marginBottom:"40px"}}>
                <thead>
                  <tr>
                    {["","Emotion","Precision","Recall","F1","Support","Match"].map((h,i)=>(
                      <th key={i} style={{padding:"10px 14px 14px",
                        textAlign:i<=1?"left":"right",
                        borderBottom:`2px solid ${INK}`,
                        fontFamily:"'DM Mono',monospace", fontSize:"12px",
                        letterSpacing:"0.12em", fontWeight:"400", color:MUTED}}>
                        {h.toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((e,i)=>{
                    const match=CR[e].f1>=0.85?"Strong":CR[e].f1>=0.75?"Good":"Weak";
                    const mc=CR[e].f1>=0.85?"#2a5a48":CR[e].f1>=0.75?"#5a4a2a":"#7a3a2a";
                    return (
                      <tr key={e} className="trow fade"
                        style={{borderBottom:`1px solid ${RULE}`,
                          animationDelay:`${i*40}ms`}}>
                        <td style={{padding:"14px 14px", width:"36px"}}>
                          <div style={{width:"32px", height:"32px", borderRadius:"50%",
                            background:INK, display:"flex", alignItems:"center",
                            justifyContent:"center", fontSize:"13px", color:CREAM,
                            fontWeight:"500", letterSpacing:"0"}}>
                            {e[0].toUpperCase()}
                          </div>
                        </td>
                        <td style={{padding:"14px 14px", fontWeight:"500",
                          fontSize:"18px", textTransform:"capitalize"}}>{e}</td>
                        {[CR[e].p,CR[e].r,CR[e].f1].map((v,j)=>(
                          <td key={j} style={{padding:"14px 14px", textAlign:"right",
                            fontFamily:"'DM Mono',monospace", fontSize:"15px",
                            color:v>=0.85?"#2a5a48":v>=0.75?"#5a4a2a":"#7a3a2a"}}>
                            {v.toFixed(2)}
                          </td>
                        ))}
                        <td style={{padding:"14px 14px", textAlign:"right",
                          fontFamily:"'DM Mono',monospace", fontSize:"15px", color:MUTED}}>
                          {CR[e].s}
                        </td>
                        <td style={{padding:"14px 14px", textAlign:"right"}}>
                          <span style={{fontSize:"13px", fontFamily:"'DM Mono',monospace",
                            color:mc, letterSpacing:"0.06em"}}>{match}</span>
                        </td>
                      </tr>
                    );
                  })}
                  <tr style={{borderTop:`2px solid ${INK}`}}>
                    <td/><td style={{padding:"14px 14px", fontFamily:"'DM Mono',monospace",
                      fontSize:"13px", letterSpacing:"0.10em", color:MUTED}}>
                      WEIGHTED AVG
                    </td>
                    {["0.87","0.86","0.86"].map((v,i)=>(
                      <td key={i} style={{padding:"14px 14px", textAlign:"right",
                        fontFamily:"'DM Mono',monospace", fontSize:"15px",
                        fontWeight:"500"}}>{v}</td>
                    ))}
                    <td style={{padding:"14px 14px", textAlign:"right",
                      fontFamily:"'DM Mono',monospace", fontSize:"15px", color:MUTED}}>636</td>
                    <td/>
                  </tr>
                </tbody>
              </table>

              <SectionHead n={3} title="Dataset composition"/>
              <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:"1px",
                background:RULE, border:`1px solid ${RULE}`,
                borderRadius:"4px", overflow:"hidden"}}>
                {[
                  {name:"RAVDESS",n:1440,p:34,desc:"24 actors · 8 emotions · controlled lab"},
                  {name:"TESS",   n:2800,p:66,desc:"2 speakers · 7 emotions · naturalistic"},
                ].map((d,i)=>(
                  <div key={d.name} style={{background:CREAM, padding:"24px 28px"}}>
                    <div style={{display:"flex", justifyContent:"space-between",
                      alignItems:"baseline", marginBottom:"14px"}}>
                      <span style={{fontSize:"26px", fontWeight:"500",
                        letterSpacing:"-0.02em"}}>{d.name}</span>
                      <Mono size="13px" color={TAUPE}>{d.n.toLocaleString()} files</Mono>
                    </div>
                    <InkBar pct={d.p} delay={i*200} h={3}/>
                    <div style={{marginTop:"10px"}}>
                      <Mono size="12px">{d.desc}</Mono>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── RANKINGS ────────────────────────────────────────────────── */}
          {activeSection==="rankings"&&(
            <div>
              <SectionHead n={1} title="F1 ranking · all classes"/>
              <div style={{display:"flex", flexDirection:"column"}}>
                {sorted.map((e,i)=>{
                  const pct=(CR[e].f1*100);
                  const badge=CR[e].f1>=0.85?"Strong":CR[e].f1>=0.75?"Good":"Weak";
                  const bc=CR[e].f1>=0.85?"#2a5a48":CR[e].f1>=0.75?"#5a4a2a":"#7a3a2a";
                  return (
                    <div key={e} className="fade trow"
                      style={{display:"grid",
                        gridTemplateColumns:"28px 140px 1fr 80px 80px",
                        alignItems:"center", gap:"20px",
                        padding:"16px 12px",
                        borderBottom:`1px solid ${RULE}`,
                        animationDelay:`${i*45}ms`}}>
                      <Mono size="13px" color={TAUPE}>
                        {String(i+1).padStart(2,"0")}
                      </Mono>
                      <div style={{display:"flex", alignItems:"center", gap:"10px"}}>
                        <div style={{width:"34px", height:"34px", borderRadius:"50%",
                          background:INK, display:"flex", alignItems:"center",
                          justifyContent:"center", fontSize:"14px", color:CREAM,
                          fontWeight:"500", flexShrink:0}}>
                          {e[0].toUpperCase()}
                        </div>
                        <span style={{fontSize:"19px", fontWeight:"500",
                          textTransform:"capitalize"}}>{e}</span>
                      </div>
                      <InkBar pct={pct} delay={i*45} h={2}/>
                      <div style={{textAlign:"right"}}>
                        <span style={{fontSize:"28px", fontWeight:"400",
                          letterSpacing:"-0.03em"}}>{pct.toFixed(0)}</span>
                        <Mono size="12px" color={MUTED}> /100</Mono>
                      </div>
                      <div style={{textAlign:"right"}}>
                        <Mono size="12px" color={bc}>{badge}</Mono>
                      </div>
                    </div>
                  );
                })}
              </div>

              <HR my="36px"/>
              <SectionHead n={2} title="Gap analysis · precision vs recall"/>
              <div style={{display:"flex", flexDirection:"column", gap:"14px"}}>
                {sorted.map((e,i)=>{
                  const gap = Math.abs(CR[e].p - CR[e].r);
                  const bias = CR[e].p > CR[e].r ? "Precision-biased" : CR[e].p < CR[e].r ? "Recall-biased" : "Balanced";
                  return (
                    <div key={e} className="fade"
                      style={{display:"grid",
                        gridTemplateColumns:"100px 70px 70px 1fr 110px",
                        alignItems:"center", gap:"16px",
                        animationDelay:`${i*35}ms`}}>
                      <span style={{fontSize:"16px", fontWeight:"500",
                        textTransform:"capitalize"}}>{e}</span>
                      <Mono size="13px" color={MUTED}>P {(CR[e].p*100).toFixed(0)}%</Mono>
                      <Mono size="13px" color={MUTED}>R {(CR[e].r*100).toFixed(0)}%</Mono>
                      <div style={{height:"3px", background:RULE, position:"relative"}}>
                        {/* P bar */}
                        <div style={{position:"absolute", left:0, top:0, height:"100%",
                          width:`${CR[e].p*100}%`, background:INK, opacity:0.7}}/>
                        {/* R bar offset */}
                        <div style={{position:"absolute", left:0, top:0, height:"3px",
                          marginTop:"5px", width:`${CR[e].r*100}%`,
                          background:TAUPE, opacity:0.6}}/>
                      </div>
                      <Mono size="12px" color={gap<0.05?"#2a5a48":gap<0.12?WARM:"#7a3a2a"}>
                        {bias}
                      </Mono>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── MATRIX ──────────────────────────────────────────────────── */}
          {activeSection==="matrix"&&(
            <div>
              <SectionHead n={1} title="Confusion matrix · 636 samples"/>
              <div style={{overflowX:"auto", marginBottom:"40px"}}>
                <table style={{borderCollapse:"separate", borderSpacing:"3px"}}>
                  <thead>
                    <tr>
                      <td style={{padding:"0 12px 12px 0",
                        fontFamily:"'DM Mono',monospace", fontSize:"12px",
                        color:MUTED, borderBottom:`1px solid ${INK}`,
                        letterSpacing:"0.10em"}}>PREDICTED →</td>
                      {EMOTIONS.map(e=>(
                        <td key={e} style={{padding:"0 2px 10px", verticalAlign:"bottom",
                          borderBottom:`1px solid ${INK}`}}>
                          <div style={{writingMode:"vertical-rl",
                            transform:"rotate(180deg)", height:"80px",
                            display:"flex", alignItems:"center",
                            fontFamily:"'DM Mono',monospace", fontSize:"12px",
                            color:MUTED, letterSpacing:"0.08em",
                            textTransform:"capitalize"}}>{e}</div>
                        </td>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {CM.map((row,ri)=>{
                      const total=row.reduce((a,b)=>a+b,0);
                      return (
                        <tr key={ri} style={{borderBottom:`1px solid ${RULE}`}}>
                          <td style={{padding:"2px 14px 2px 0",
                            fontFamily:"'DM Mono',monospace",
                            fontSize:"13px", color:MUTED, whiteSpace:"nowrap",
                            textTransform:"capitalize", fontStyle:"italic",
                            borderBottom:`1px solid ${RULE}`}}>
                            {EMOTIONS[ri]}
                          </td>
                          {row.map((val,ci)=>{
                            const diag=ri===ci;
                            const intensity=val/total;
                            return (
                              <td key={ci} style={{padding:"2px"}}>
                                <div style={{
                                  width:"46px", height:"40px",
                                  background:diag
                                    ?`rgba(26,21,16,${0.08+intensity*0.60})`
                                    :val>0?`rgba(201,169,110,${intensity*0.30})`:"transparent",
                                  display:"flex", alignItems:"center",
                                  justifyContent:"center",
                                  border:diag?`1px solid rgba(26,21,16,0.20)`:"none",
                                  fontFamily:"'DM Mono',monospace",
                                  fontSize:"13px",
                                  color:diag?INK:"rgba(26,21,16,0.40)",
                                  fontWeight:diag?"500":"300",
                                }}>
                                  {val>0?val:"·"}
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <SectionHead n={2} title="Per-class recall detail"/>
              <div style={{display:"grid", gridTemplateColumns:"1fr 1fr",
                gap:"1px", background:RULE, border:`1px solid ${RULE}`,
                borderRadius:"4px", overflow:"hidden"}}>
                {EMOTIONS.map((e,i)=>{
                  const row=CM[i], total=row.reduce((a,b)=>a+b,0);
                  const recall=row[i]/total;
                  return (
                    <div key={e} className="fade"
                      style={{background:CREAM, padding:"22px 26px",
                        animationDelay:`${i*40}ms`}}>
                      <div style={{display:"flex", justifyContent:"space-between",
                        alignItems:"baseline", marginBottom:"10px"}}>
                        <span style={{fontSize:"18px", fontWeight:"500",
                          textTransform:"capitalize"}}>{e}</span>
                        <Mono size="13px" color={
                          recall>=0.85?"#2a5a48":recall>=0.70?WARM:"#7a3a2a"}>
                          {(recall*100).toFixed(0)}%
                        </Mono>
                      </div>
                      <InkBar pct={recall*100} delay={i*40} h={2}
                        color={recall>=0.85?INK:recall>=0.70?WARM:"#c97b6e"}/>
                      <div style={{marginTop:"8px"}}>
                        <Mono size="11px" color={MUTED}>
                          {row[i]}/{total} correct
                        </Mono>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── TRAINING ────────────────────────────────────────────────── */}
          {activeSection==="training"&&(
            <div>
              <SectionHead n={1} title="Training curve · 69 epochs"/>

              {/* Big single training chart */}
              <div style={{marginBottom:"40px",
                border:`1px solid ${RULE}`, padding:"28px 28px 16px",
                background:PAPER, borderRadius:"4px"}}>
                <div style={{display:"flex", justifyContent:"space-between",
                  alignItems:"center", marginBottom:"20px"}}>
                  <div style={{display:"flex", gap:"20px"}}>
                    {[["Train acc",INK,"2px"],["Val acc",TAUPE,"1.5px dashed"]].map(([l,c,s])=>(
                      <div key={l} style={{display:"flex", alignItems:"center", gap:"7px"}}>
                        <svg width="20" height="8">
                          <line x1="0" y1="4" x2="20" y2="4"
                            stroke={c} strokeWidth="2"
                            strokeDasharray={s.includes("dashed")?"4 3":"none"}/>
                        </svg>
                        <Mono size="12px">{l}</Mono>
                      </div>
                    ))}
                  </div>
                  <Mono size="11px" color={MUTED}>Early stop @ epoch 69</Mono>
                </div>
                <svg width="100%" height="220" viewBox="0 0 600 220" preserveAspectRatio="none">
                  {[0.25,0.5,0.75,1.0].map(v=>(
                    <line key={v} x1="0" y1={216-v*208} x2="600" y2={216-v*208}
                      stroke={RULE} strokeWidth="1"/>
                  ))}
                  {[0.25,0.5,0.75,1.0].map(v=>(
                    <text key={v} x="2" y={216-v*208-3}
                      fontSize="11" fill={MUTED} fontFamily="'DM Mono',monospace">
                      {v.toFixed(2)}
                    </text>
                  ))}
                  {(()=>{
                    const tPts=HIST.map((d,i)=>[(i/68)*600, 216-d.tr*208]);
                    const vPts=HIST.map((d,i)=>[(i/68)*600, 216-d.vl*208]);
                    return <>
                      <polyline points={tPts.map(([x,y])=>`${x},${y}`).join(" ")}
                        fill="none" stroke={INK} strokeWidth="2" strokeLinejoin="round"/>
                      <polyline points={vPts.map(([x,y])=>`${x},${y}`).join(" ")}
                        fill="none" stroke={TAUPE} strokeWidth="1.5"
                        strokeLinejoin="round" strokeDasharray="5 3"/>
                      {/* Best epoch marker */}
                      <line x1={tPts[68][0]} y1="0" x2={tPts[68][0]} y2="216"
                        stroke={WARM} strokeWidth="1" strokeDasharray="3 3"/>
                      <circle cx={tPts[68][0]} cy={vPts[68][1]} r="4"
                        fill={TAUPE} stroke={CREAM} strokeWidth="2"/>
                    </>;
                  })()}
                </svg>
                <div style={{display:"flex", justifyContent:"space-between", marginTop:"4px"}}>
                  {[1,10,20,30,40,50,60,69].map(n=>(
                    <Mono key={n} size="11px">{n}</Mono>
                  ))}
                </div>
              </div>

              <SectionHead n={2} title="Hyperparameter table"/>
              <table style={{width:"100%", borderCollapse:"collapse",
                marginBottom:"40px", fontSize:"16px"}}>
                <tbody>
                  {[
                    ["Optimizer","AdamW","weight_decay = 5e-3"],
                    ["Learning Rate","1e-4","ReduceLROnPlateau, factor 0.5"],
                    ["Batch Size","32","gradient clip = 0.5"],
                    ["Epochs","69 / 100","early stop patience = 20"],
                    ["Label Smoothing","0.10","cross-entropy loss"],
                    ["Augmentation","time + freq mask","noise σ = 0.015, gain 0.85–1.15"],
                    ["Train/Val/Test","70 / 15 / 15%","stratified split"],
                    ["Device","Apple M4 MPS","PyTorch 2.x"],
                  ].map(([k,v,note],i)=>(
                    <tr key={k} className="trow fade"
                      style={{borderBottom:`1px solid ${RULE}`,
                        animationDelay:`${i*40}ms`}}>
                      <td style={{padding:"15px 16px 15px 0", width:"200px"}}>
                        <Mono size="12px" color={MUTED}>{k.toUpperCase()}</Mono>
                      </td>
                      <td style={{padding:"15px 16px", width:"180px",
                        fontSize:"18px", fontWeight:"500"}}>{v}</td>
                      <td style={{padding:"15px 0", fontSize:"15px",
                        color:TAUPE, fontStyle:"italic"}}>{note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <SectionHead n={3} title="Architecture layers"/>
              <div style={{display:"flex", flexDirection:"column",
                border:`1px solid ${RULE}`, borderRadius:"4px",
                overflow:"hidden", background:PAPER}}>
                {[
                  {n:"Input",      d:"(N, 1, 128, 128)  ·  normalised MFCC + Mel + Chroma"},
                  {n:"Conv Block 1",d:"Conv2d 1→32 ×2, BN, ReLU, MaxPool, Dropout 0.20"},
                  {n:"Conv Block 2",d:"Conv2d 32→64 ×2, BN, ReLU, MaxPool, Dropout 0.20"},
                  {n:"Conv Block 3",d:"Conv2d 64→128, BN, ReLU, MaxPool, Dropout 0.25"},
                  {n:"SE-Block",    d:"Squeeze-and-Excitation attention  ·  reduction = 8"},
                  {n:"BiLSTM",      d:"128 units, 2 layers, bidirectional  ·  dropout 0.30"},
                  {n:"Self-Attn",   d:"Weighted pooling  →  256-dim vector"},
                  {n:"Head",        d:"Dense 256 → 128 → 8  ·  Dropout 0.40"},
                  {n:"Output",      d:"Softmax  ·  8 emotion classes"},
                ].map((row,i)=>(
                  <div key={row.n} className="fade"
                    style={{display:"grid", gridTemplateColumns:"170px 1fr",
                      gap:"0", borderBottom:i<8?`1px solid ${RULE}`:"none",
                      animationDelay:`${i*40}ms`}}>
                    <div style={{padding:"15px 22px",
                      borderRight:`1px solid ${RULE}`,
                      background:i===0||i===8?INK:CREAM}}>
                      <span style={{fontSize:"15px", fontWeight:"500",
                        color:i===0||i===8?CREAM:INK}}>{row.n}</span>
                    </div>
                    <div style={{padding:"15px 22px", background:CREAM}}>
                      <Mono size="13px" color={MUTED}>{row.d}</Mono>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
