/* Apply dynamic experts to the geographically excluded Maryino district. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320;
const xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY],ll=p=>({lon:p[0]/MX+37.74,lat:p[1]/MY+55.654});
const sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2, sum=a=>a.reduce((s,v)=>s+v,0);
function run(raw,lib,expert,gate,owner){
 const d=Object.fromEntries(Object.keys(lib.paths).map(k=>[k,lib.csv(raw[k])])),measure=lib.featureFactory(d),es=lib.prepare(d,measure).map(expert.decorate),cut='2026-11-27';
 for(let k of Object.keys(owner.sources))d[k]=lib.csv(raw[k]); const assets=owner.makeAssets(d);
 const cells=d.district.map(r=>[+r.x_m,+r.y_m]),f=cells.map(measure),idx=d.district.map((r,i)=>r.allowed==='1'?i:-1).filter(i=>i>=0),sites=idx.map(i=>cells[i]),sf=idx.map(i=>f[i]);
 let rho=f.map(x=>Math.exp(.8*x[4]+.15*x[5]+.05*x[6])),total=sum(rho);rho=rho.map(x=>x/total);
 const market=d.market.map(r=>({...r,p:xy(r)})),all=d.reviewed.filter(r=>r.eligible==='yes').map(r=>({...r,p:xy(r)})),extra=all.filter(r=>!market.some(t=>t.network===r.network&&sq(t.p,r.p)<300**2));
 function event(rest,club){let allowed=sites.map(p=>[...rest,...extra].every(c=>sq(p,c.p)>=150**2)),g=lib.capture(cells,sites,rest,rho,800);let c=club||{name:'new DDX',opened:cut,p:[0,0]};let target=lib.nearest(sites,c.p);let e=expert.decorate({c,tile:'Maryino',p:sites,f:sf,allowed,target,gain:{800:g.gain}});return {e,g};}
 const current=event(market),loo=market.filter(c=>c.network==='DDX'&&c.scope==='inside').map(c=>event(market.filter(r=>r!==c),c)),out={};
 for(let cfg of owner.configs){
   let data=es.filter(e=>e.c.opened<cut).map(e=>owner.project(e,cfg,assets)),{m,g}=owner.fitModel(data,cfg,cut,expert,gate);
   m.cut=data.map(e=>e.c.opened).sort().at(-1);let ep=owner.project(current.e,cfg,assets),mm=gate.adapted(ep,m,g,cfg),p=expert.prob(ep,cfg,mm);
   const report=(e,g,prob,rule)=>expert.pick(e,prob,rule).map(i=>({...ll(e.p[i]),probability:prob[i],net_gain_pp:100*g.gain[i],gross_pp:100*g.gross[i],own_transfer_pp:100*(g.gross[i]-g.gain[i])}));
   out[cfg.name]={model:m,local_gate:mm.gate,posterior_total:sum(p),recommendations:{mode:report(current.e,current.g,p,'mode'),mass:report(current.e,current.g,p,'mass')},loo:loo.flatMap(({e})=>['mode','mass'].map(rule=>(()=>{let ep=owner.project({...e,c:{...e.c,opened:cut}},cfg,assets);return expert.evaluate(ep,cfg,gate.adapted(ep,m,g,cfg),rule);})()))};
 }
 return {cut,training_openings:es.length,eligible:current.e.allowed.filter(Boolean).length,results:out,limitations:'Geographically held-out Maryino; leave-one-out is reconstruction using current competitor set and all outside-district observations up to cutoff, not historical forecast. Current POIs, Euclidean distances, demand proxies, no lease or vacancy. Probabilities are conditional on the allowed local grid.'};
}
if(typeof module!=='undefined')module.exports={run};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),expert=require('./expert_flexible'),gate=require('./local_gate_experiment'),owner=require('./owner_experiment'),raw=Object.fromEntries(Object.entries({...lib.paths,...owner.sources}).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib,expert,gate,owner);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/owner_maryino.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify(r,null,2));}
