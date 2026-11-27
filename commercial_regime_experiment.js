/* Repair the regime definition: a mapped mall point is not a building category. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320,xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY];
const sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2,sum=a=>a.reduce((s,v)=>s+v,0);
function index(rows){let b=new Map(),B=500;for(let r of rows){let p=xy(r);if(!p.every(Number.isFinite))continue;let key=Math.floor(p[0]/B)+','+Math.floor(p[1]/B);if(!b.has(key))b.set(key,[]);b.get(key).push([p,+r.footprint_m2]);}return(p,dist,area)=>{let x=Math.floor(p[0]/B),y=Math.floor(p[1]/B),n=Math.ceil(dist/B);for(let i=x-n;i<=x+n;i++)for(let j=y-n;j<=y+n;j++)for(let [z,v] of b.get(i+','+j)||[])if(v>=area&&sq(p,z)<=dist*dist)return true;return false;};}
function featureFactory(d,lib){let measure=lib.featureFactory(d),venue=index(d.venues),cache=new Map();return p=>{let k=p.join(',');if(cache.has(k))return cache.get(k);let f=measure(p),v=[...f,+venue(p,150,1000),+venue(p,250,1500),+venue(p,100,2000)];cache.set(k,v);return v;};}
const specs=[
{name:'mall_reference',group:'mall',structural:false,half:1.5,mix:true,experts:true},
{name:'commercial_150',group:'commercial150',structural:false,half:1.5,mix:true,experts:true},
{name:'commercial_250',group:'commercial250',structural:false,half:1.5,mix:true,experts:true},
{name:'commercial_structural',group:'commercial150',structural:true,half:1.5,mix:true,experts:true}
];
function project(e,c){let group=e.f.map(a=>+(a[0]>=Math.exp(-150/180)||(c.group==='commercial150'&&a[7])||(c.group==='commercial250'&&a[8]))),a=e.f.map((a,i)=>[...a.slice(0,7),...(c.structural?[Math.log(Math.max(e.gain[800][i],1e-14))]:[])]);return {...e,group,a};}
function run(raw,lib,expert){
 let d=Object.fromEntries(Object.keys(lib.paths).map(k=>[k,lib.csv(raw[k])])),es=lib.prepare(d,featureFactory(d,lib)),base=es.filter(e=>e.split==='train'),test=es.filter(e=>e.split==='time_test'),cv=[],rows=[],models={};
 let audit={};for(let c of specs){let a=es.map(e=>project(e,c));audit[c.name]={train:sum(a.filter(e=>e.split==='train').map(e=>e.group[e.target])),test:sum(a.filter(e=>e.split==='time_test').map(e=>e.group[e.target]))};}
 function step(cut,end,isCV){for(let c of specs){let a=es.map(e=>project(e,c)),train=a.filter(e=>e.c.opened<cut),val=a.filter(e=>e.split===(isCV?'train':'time_test')&&e.c.opened>=cut&&e.c.opened<end);if(!val.length||train.length<4)continue;let m=expert.fit(train,c,cut);m.cut=train.map(e=>e.c.opened).sort().at(-1);if(!isCV){models[cut]??={};models[cut][c.name]=m;}for(let e of val){if(m.cut>=e.c.opened)throw Error('Future labels');for(let rule of ['mode','mass'])(isCV?cv:rows).push(expert.evaluate(e,c,m,rule));}}}
 step('2023-01-01','2024-01-01',true);step('2024-01-01','2025-01-01',true);
 let dates=['2025-01-01','2025-07-01','2026-01-01','2026-07-01','2027-01-01'];for(let i=0;i<4;i++)step(dates[i],dates[i+1],false);
 let summarize=a=>specs.flatMap(c=>['mode','mass'].map(rule=>({name:c.name,rule,...expert.stats(a.filter(r=>r.model===c.name&&r.rule===rule))})));
 let validation=summarize(cv),selected=[...validation].sort((a,b)=>a.radius_nll-b.radius_nll||b.hit3_300-a.hit3_300||b.hit1_300-a.hit1_300)[0];
 return {train:base.length,test:test.length,audit,validation,selected,results:summarize(rows),models,rows,limitations:'Moscow only, known target tile, contemporary POIs and venue centroids. Commercial-proximity regime is a proxy, not verified building category. No rent/vacancy. Pre-2025 forward validation selects model; 2025+ labels have already been examined in earlier experiments.'};
}
if(typeof module!=='undefined')module.exports={run,project,featureFactory,specs};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),expert=require('./expert_flexible'),raw=Object.fromEntries(Object.entries(lib.paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib,expert);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/commercial_regime_benchmark.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({audit:r.audit,selected:r.selected,results:r.results},null,2));}
