/* Portfolio-aware Moscow opening model. No gym pins or DDX announcement lists in owner features. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320,xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY],ll=p=>({lon:p[0]/MX+37.74,lat:p[1]/MY+55.654});
const sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2,sum=a=>a.reduce((s,v)=>s+v,0);
const sources={geocodes:'data/real_estate/adg_geocodes.csv',areas:'data/real_estate/adg_planned_areas_2020.csv'};
const configs=[
{name:'matched_reference',owner:false,area:false,structural:false,broad:false},
{name:'owner_strict',owner:true,area:false,structural:false,broad:false},
{name:'owner_broad',owner:true,area:false,structural:false,broad:true},
{name:'owner_area',owner:true,area:true,structural:false,broad:false},
{name:'owner_structural',owner:true,area:true,structural:true,broad:false}
].map(c=>({...c,half:1.5,mix:true,experts:true,mode:'local'}));
function makeAssets(d){let a=new Map(d.areas.map(r=>[r.name,r]));return d.geocodes.map(r=>({...r,p:xy(r),area:a.get(r.name)}));}
function ownerFeatures(p,date,assets,c){
 let proximity=0,gla=0,postDeal=0;
 for(let a of assets){if(a.portfolio_known_by>date||(!c.broad&&a.confidence==='provisional_osm_match'))continue;
   let w=Math.exp(-sq(p,a.p)/(2*120**2));proximity=Math.max(proximity,w);
   if(a.area&&a.area.known_by<=date)gla=Math.max(gla,w*Math.log1p(+a.area.planned_gla_m2/1000));
 }
 if(date>='2021-10-19')postDeal=proximity;
 return c.owner?[proximity,postDeal,...(c.area?[gla]:[])]:[];
}
function project(e,c,assets){
 // Date-specific attributes, not the identity of the eventual tenant. Owner list is a historical portfolio, not vacancy.
 return {...e,a:e.a.map((a,i)=>[...a.slice(0,c.structural?8:7),...ownerFeatures(e.p[i],e.c.opened,assets,c)])};
}
function fitModel(es,c,cut,expert,gate){let m=expert.fit(es,c,cut);m.cut=es.map(e=>e.c.opened).sort().at(-1);let g=gate.gateFit(es,m,cut,c);return {m,g};}
function run(raw,lib,expert,gate){
 const d=Object.fromEntries(Object.keys(lib.paths).map(k=>[k,lib.csv(raw[k])]));for(let k of Object.keys(sources))d[k]=lib.csv(raw[k]);
 const assets=makeAssets(d),es=lib.prepare(d,lib.featureFactory(d)).map(expert.decorate),base=es.filter(e=>e.split==='train'),test=es.filter(e=>e.split==='time_test'),cache=Object.fromEntries(configs.map(c=>[c.name,es.map(e=>project(e,c,assets))]));
 let cv=[],rows=[],models={};function step(cut,end,isCV){for(let c of configs){let a=cache[c.name],train=a.filter(e=>e.c.opened<cut),val=a.filter(e=>e.split===(isCV?'train':'time_test')&&e.c.opened>=cut&&e.c.opened<end);if(train.length<4||!val.length)continue;let {m,g}=fitModel(train,c,cut,expert,gate);if(!isCV){models[cut]??={};models[cut][c.name]={m,g};}
 for(let e of val){if(m.cut>=e.c.opened)throw Error('Future labels');let mm=gate.adapted(e,m,g,c);for(let rule of ['mode','mass'])(isCV?cv:rows).push({...expert.evaluate(e,c,mm,rule),owner_near:assets.some(a=>a.confidence!=='provisional_osm_match'&&a.portfolio_known_by<=e.c.opened&&sq(e.c.p,a.p)<=150**2)});}}}
 step('2023-01-01','2024-01-01',true);step('2024-01-01','2025-01-01',true);
 let dates=['2025-01-01','2025-07-01','2026-01-01','2026-07-01','2027-01-01'];for(let i=0;i<4;i++)step(dates[i],dates[i+1],false);
 const summarize=a=>configs.flatMap(c=>['mode','mass'].map(rule=>({name:c.name,rule,...expert.stats(a.filter(r=>r.model===c.name&&r.rule===rule))})));
 let validation=summarize(cv),selected=[...validation].sort((a,b)=>a.radius_nll-b.radius_nll||b.hit3_300-a.hit3_300||b.hit1_300-a.hit1_300)[0];
 return {train:base.length,test:test.length,assets:assets.length,strict_assets:assets.filter(a=>a.confidence!=='provisional_osm_match').length,validation,selected,results:summarize(rows),models,rows,limitations:'Moscow only, known target tile, present-day POIs and incomplete historical portfolio geocodes. As-of date gating of 2020 project areas and 2021 public partnership date; no actual building availability or rents. DDX announcements and address audit are never loaded. Owner membership is an imperfect proximity proxy, not a causal effect. Reused 2025+ labels.'};
}
if(typeof module!=='undefined')module.exports={run,configs,sources,makeAssets,project,fitModel,ownerFeatures};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),expert=require('./expert_flexible'),gate=require('./local_gate_experiment'),raw=Object.fromEntries(Object.entries({...lib.paths,...sources}).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib,expert,gate);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/owner_benchmark.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({selected:r.selected,results:r.results},null,2));}
