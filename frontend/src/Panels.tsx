import {useEffect,useState} from 'react'
const get=(u:string)=>fetch(u).then(r=>r.ok?r.json():null).catch(()=>null)
const Bars=({d}:{d:Record<string,number>})=>{const mx=Math.max(1,...Object.values(d));return <>{Object.entries(d).map(([k,v])=>
  <div key={k} style={{margin:'4px 0'}}>{k}: {v}<div style={{background:'#4aa3ff',height:8,width:`${100*v/mx}%`}}/></div>)}</>}
const mean=(a:number[])=>a.length?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(1):'–'

export default function Panels(){
  const [m,setM]=useState<any>(null),[d,setD]=useState<any>(null),[s,setS]=useState<any>(null)
  useEffect(()=>{get('/api/model/metrics').then(setM);get('/api/dataset/statistics').then(setD);get('/api/analytics/summary').then(setS)},[])
  return <div className="grid">
    <div className="card"><h2>Model performance (held-out test set)</h2>{!m?<p>No evaluation yet — run scripts/evaluate_model.py</p>:<>
      <div className="row"><span>accuracy</span><b>{(m.test_accuracy*100).toFixed(1)}%</b></div>
      <div className="row"><span>macro F1</span><b>{m.macro_f1.toFixed(3)}</b></div>
      <div className="row"><span>ROC-AUC</span><b>{m.roc_auc==null?'n/a':m.roc_auc.toFixed(3)}</b></div>
      <table style={{marginTop:8}}><tbody>{m.confusion_matrix.map((r:number[],i:number)=><tr key={i}><td>{m.classes[i]}</td>{r.map((v,j)=><td key={j} style={{padding:'2px 10px',background:i===j?'#1f6f43':'#5a2a2a'}}>{v}</td>)}</tr>)}</tbody></table>
      <p style={{color:'#8b98a5'}}>{m.limitations}</p></>}</div>
    <div className="card"><h2>Dataset</h2>{!d?<p>No dataset stats — run scripts/prepare_dataset.py</p>:<>
      <div className="row"><span>total / corrupt / duplicates</span><span>{d.total} / {d.bad_files} / {d.duplicates_removed}</span></div>
      <div className="row"><span>train / val / test</span><span>{d.splits.train.n} / {d.splits.val.n} / {d.splits.test.n}</span></div>
      <div className="row"><span>patient-level split</span><span>{String(d.patient_level_split)}</span></div>
      <div className="row"><span>imbalance ratio</span><span>{d.imbalance_ratio}</span></div><Bars d={d.class_distribution}/></>}</div>
    <div className="card"><h2>Inference history</h2>{!s||!s.count?<p>No analyses yet</p>:<>
      <div className="row"><span>analyses</span><span>{s.count}</span></div>
      <div className="row"><span>mean quality</span><span>{mean(s.quality)}</span></div>
      <div className="row"><span>mean confidence</span><span>{mean(s.confidence.map((c:number)=>c*100))}%</span></div>
      <div className="row"><span>mean time (ms)</span><span>{mean(s.total_ms)}</span></div><Bars d={s.predictions}/></>}</div>
  </div>}
