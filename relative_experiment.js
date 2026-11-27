/* Relative location features and decision-policy calibration on pre-2025 data. */
'use strict';
const sum=a=>a.reduce((s,v)=>s+v,0),sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320,ll=p=>({lon:p[0]/MX+37.74,lat:p[1]/MY+55.654});
const configs=[
{name:'absolute',relative:false,structural:false,half:1.5,mix:true,experts:true},
{name:'relative',relative:true,structural:false,half:1.5,mix:true,experts:true},
{name:'relative_structural',relative:true,structural:true,half:1.5,mix:true,experts:true}
];
function ranks(a){let order=a.map((v,i)=>i).sort((i,j)=>a[i]-a[j]),r=Array(a.length);for(let i=0;i<order.length;){let j=i+1;while(j<order.length&&a[order[j]]===a[order[i]])j++;for(let k=i;k<j;k++)r[order[k]]=(i+j-1)/(2*Math.max(1,a.length-1));i=j;}return r;}
function design(e,c){
 let a=e.a.map(v=>v.slice(0,c.structural?8:7));
 if(c.relative){let cols=a[0].map((_,h)=>ranks(a.map(v=>v[h])));a=a.map((v,i)=>[...v,...cols.map(col=>col[i])]);}
 return {...e,a};
}
function mass(e,p){let lookup=new Map(e.p.map((z,i)=>[z.join(','),i])),off=[];for(let x=-300;x<=300;x+=100)for(let y=-300;y<=300;y+=100)if(x*x+y*y<=90000)off.push([x,y]);return e.p.map(z=>sum(off.map(([x,y])=>p[lookup.get([z[0]+x,z[1]+y].join(','))]||0)));}
function policies(e,p){
 const m=mass(e,p),g=e.group,near=p.map((v,i)=>g[i]===1?v:0),far=p.map((v,i)=>g[i]===0?v:0),mn=mass(e,near),mf=mass(e,far);
 const take=(scores,seen=[])=>{let best=-1;for(let i=0;i<scores.length;i++)if(e.allowed[i]&&seen.every(j=>sq(e.p[i],e.p[j])>=600**2)&&(best<0||scores[i]>scores[best]))best=i;return best;};
 const pick=seq=>{let out=[];for(let s of seq){let v=take(s,out);if(v>=0)out.push(v);}return out;};
 const median=[sum(e.p.map((z,i)=>z[0]*p[i])),sum(e.p.map((z,i)=>z[1]*p[i]))],center=e.p.map(z=>-sq(z,median));
 return {mode:pick([p,p,p]),mass:pick([m,m,m]),regime_near:pick([mn,mf,m]),regime_far:pick([mf,mn,m]),hybrid:pick([p,mf,m]),mse_hybrid:pick([center,p,m])};
}
function evaluate(e,c,m,expert){
 let p=expert.prob(e,c,m),near=sum(p.filter((_,i)=>sq(e.p[i],e.c.p)<=90000));
 return Object.entries(policies(e,p)).map(([rule,idx])=>{let ds=idx.map(i=>Math.sqrt(sq(e.p[i],e.c.p)));return{name:e.c.name,date:e.c.opened,tile:e.tile,model:c.name,rule,train_n:m.n,train_max_date:m.cut,mall_group:e.group[e.target],top1_m:ds[0],top3_m:Math.min(...ds),hit1:+(ds[0]<=300),hit3:+(Math.min(...ds)<=300),logloss:-Math.log(Math.max(p[e.target],1e-300)),radius_logloss:-Math.log(Math.max(near,1e-300)),masked:!e.allowed[e.target],predictions:idx.map(i=>ll(e.p[i]))};});
}
function run(raw,lib,expert){
 const d=Object.fromEntries(Object.keys(lib.paths).map(k=>[k,lib.csv(raw[k])])),es=lib.prepare(d,lib.featureFactory(d)).map(expert.decorate),base=es.filter(e=>e.split==='train'),test=es.filter(e=>e.split==='time_test'),cache=Object.fromEntries(configs.map(c=>[c.name,es.map(e=>design(e,c))]));
 let cv=[],rows=[],models={};function step(cut,end,isCV){let targets=isCV?base:test;for(let c of configs){let a=cache[c.name],train=a.filter(e=>e.c.opened<cut),val=a.filter(e=>e.c.opened>=cut&&e.c.opened<end&&targets.some(t=>t.c.id===e.c.id));if(!val.length||train.length<4)continue;let m=expert.fit(train,c,cut);m.cut=train.map(e=>e.c.opened).sort().at(-1);if(!isCV){models[cut]??={};models[cut][c.name]=m;}for(let e of val){if(m.cut>=e.c.opened)throw Error('Future labels');(isCV?cv:rows).push(...evaluate(e,c,m,expert));}}}
 step('2023-01-01','2024-01-01',true);step('2024-01-01','2025-01-01',true);
 let dates=['2025-01-01','2025-07-01','2026-01-01','2026-07-01','2027-01-01'];for(let i=0;i<4;i++)step(dates[i],dates[i+1],false);
 const summarize=a=>configs.flatMap(c=>['mode','mass','regime_near','regime_far','hybrid','mse_hybrid'].map(rule=>({name:c.name,rule,...expert.stats(a.filter(r=>r.model===c.name&&r.rule===rule))})));
 let validation=summarize(cv),selected=[...validation].sort((a,b)=>b.hit3_300-a.hit3_300||b.hit1_300-a.hit1_300||a.median_top1_m-b.median_top1_m)[0];
 return {train:base.length,test:test.length,validation,selected,results:summarize(rows),models,rows,limitations:'Moscow only, known target tile, modern POIs, historical dated competitors only. Relative features computed across each candidate set without labels. Selection via 2023–24 forward validation top3 then top1 then median top1. Previously examined 2025+ labels are not a fresh test.'};
}
if(typeof module!=='undefined')module.exports={run,design,configs,policies,evaluate};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),expert=require('./expert_flexible'),raw=Object.fromEntries(Object.entries(lib.paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib,expert);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/relative_benchmark.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({selected:r.selected,results:r.results},null,2));}
