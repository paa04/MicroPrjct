/* Conditional regime gate: account for local availability of the two regimes. */
'use strict';
const sum=a=>a.reduce((s,v)=>s+v,0),sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
const logit=p=>Math.log(p/(1-p)),sig=x=>1/(1+Math.exp(-Math.max(-30,Math.min(30,x))));
const configs=[
{name:'fixed_gate',mode:'fixed',structural:false,half:1.5,mix:true,experts:true},
{name:'local_gate',mode:'local',structural:false,half:1.5,mix:true,experts:true},
{name:'local_structural',mode:'local',structural:true,half:1.5,mix:true,experts:true}
];
function gateFeatures(e){
 let n=e.allowed.filter(Boolean).length,k=sum(e.group.map((g,i)=>e.allowed[i]?g:0));
 let frac=(k+1)/(n+2),g=e.a.filter((_,i)=>e.allowed[i]&&e.group[i]===1),h=e.a.filter((_,i)=>e.allowed[i]&&e.group[i]===0);
 let av=(a,j)=>a.length?sum(a.map(v=>v[j]))/a.length:0;
 return [logit(frac),av(g,3)-av(h,3),av(g,4)-av(h,4)];
}
function gateFit(es,m,cut,c){
 const d=es.filter(e=>e.allowed[e.target]),w=d.map(e=>2**(-(Date.parse(cut)-Date.parse(e.c.opened))/(365.25*86400000*c.half))),W=sum(w),x=d.map(gateFeatures),mu=x[0].map((_,j)=>sum(x.map((r,i)=>r[j]*w[i]))/W),sd=mu.map((u,j)=>Math.max(.1,Math.sqrt(sum(x.map((r,i)=>w[i]*(r[j]-u)**2))/W))),z=x.map(r=>r.map((v,j)=>(v-mu[j])/sd[j]));
 let b=[0,0,0,0];for(let iter=0;iter<100;iter++){let g=Array(4).fill(0);for(let i=0;i<d.length;i++){let p=sig(logit(m.gate)+b[0]+sum(z[i].map((v,j)=>v*b[j+1]))),err=(p-d[i].group[d[i].target])*w[i]/W;g[0]+=err;for(let j=0;j<3;j++)g[j+1]+=err*z[i][j];}b=b.map((v,j)=>v-.25*(g[j]+.5*v));}
 return {mu,sd,b};
}
function adapted(e,m,g,c){if(c.mode==='fixed')return m;let z=gateFeatures(e).map((v,j)=>(v-g.mu[j])/g.sd[j]),p=sig(logit(m.gate)+g.b[0]+sum(z.map((v,j)=>v*g.b[j+1])));return {...m,gate:p};}
function run(raw,lib,expert){
 let d=Object.fromEntries(Object.keys(lib.paths).map(k=>[k,lib.csv(raw[k])])),es=lib.prepare(d,lib.featureFactory(d)).map(expert.decorate),base=es.filter(e=>e.split==='train'),test=es.filter(e=>e.split==='time_test'),cv=[],rows=[],models={};
 function step(cut,end,isCV){let train=es.filter(e=>e.c.opened<cut),val=es.filter(e=>e.split===(isCV?'train':'time_test')&&e.c.opened>=cut&&e.c.opened<end);if(train.length<4||!val.length)return;
 for(let c of configs){let m=expert.fit(train.map(e=>({...e,a:e.a.map(a=>a.slice(0,c.structural?8:7))})),c,cut);m.cut=train.map(e=>e.c.opened).sort().at(-1);let g=gateFit(train,m,cut,c);if(!isCV){models[cut]??={};models[cut][c.name]={m,g};}for(let e of val){if(m.cut>=e.c.opened)throw Error('Future labels');for(let rule of ['mode','mass']){let mm=adapted(e,m,g,c);(isCV?cv:rows).push(expert.evaluate(e,c,mm,rule));}}}}
 step('2023-01-01','2024-01-01',true);step('2024-01-01','2025-01-01',true);
 let dates=['2025-01-01','2025-07-01','2026-01-01','2026-07-01','2027-01-01'];for(let i=0;i<4;i++)step(dates[i],dates[i+1],false);
 let summarize=a=>configs.flatMap(c=>['mode','mass'].map(rule=>({name:c.name,rule,...expert.stats(a.filter(r=>r.model===c.name&&r.rule===rule))})));
 let validation=summarize(cv),selected=[...validation].sort((a,b)=>a.radius_nll-b.radius_nll||b.hit3_300-a.hit3_300||b.hit1_300-a.hit1_300)[0];
 return {train:base.length,test:test.length,validation,selected,results:summarize(rows),models,rows,limitations:'Moscow only, known target tile, modern POIs, historical dated competitors only. Local gate uses three standardized covariates without target labels; fitted on past events. 2025+ outcomes previously examined.'};
}
if(typeof module!=='undefined')module.exports={run,gateFeatures,gateFit,adapted,configs};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),expert=require('./expert_flexible'),raw=Object.fromEntries(Object.entries(lib.paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib,expert);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/local_gate_benchmark.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({selected:r.selected,results:r.results},null,2));}
