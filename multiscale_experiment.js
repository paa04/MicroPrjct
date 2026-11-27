/* Multiscale commercial features, with chronologically selected regularization.
   No target addresses or club names enter the predictors. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320,xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY];
const sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2,sum=a=>a.reduce((s,v)=>s+v,0);
const names=['mall_180','retail_125','metro_250','venue_capacity_150','housing_500','office_500','stores_500','log_network_gain','large_venue_60','medium_venue_100','retail_50','metro_75','housing_150','office_150','stores_150','commercial_mix_250','local_housing_ratio','retail_venue_interaction'];
function spatialIndex(rows,value){let b=new Map(),B=500;for(let r of rows){let p=xy(r);if(!p.every(Number.isFinite))continue;let key=Math.floor(p[0]/B)+','+Math.floor(p[1]/B);if(!b.has(key))b.set(key,[]);b.get(key).push([p,value(r)]);}
return(p,r)=>{let a=[],x=Math.floor(p[0]/B),y=Math.floor(p[1]/B),n=Math.ceil(r/B);for(let i=x-n;i<=x+n;i++)for(let j=y-n;j<=y+n;j++)for(let [z,v] of b.get(i+','+j)||[]){let d=sq(p,z);if(d<=r*r)a.push([d,v]);}return a;};}
function featureFactory(d,lib){
 const base=lib.featureFactory(d),venue=spatialIndex(d.venues,r=>+r.footprint_m2||0),retail=spatialIndex(d.retail,()=>1),metro=spatialIndex(d.metro,()=>1),housing=spatialIndex(d.housing,r=>Math.min(85,Math.max(1,+r.value||9))),office=spatialIndex(d.office,r=>Math.min(85,Math.max(1,+r.value||4))),stores=spatialIndex(d.stores,()=>1);
 const prox=(a,h,w=()=>1)=>{let v=0;for(let [ds,x] of a)v=Math.max(v,w(x)*Math.exp(-Math.sqrt(ds)/h));return v;};
 const dens=(a,h)=>Math.log1p(sum(a.map(([ds,x])=>x*Math.exp(-ds/(2*h*h)))));
 const cache=new Map();
 return p=>{let key=p.join(',');if(cache.has(key))return cache.get(key);let a=base(p),v=venue(p,1000),r=retail(p,300),m=metro(p,450),h=housing(p,600),o=office(p,600),s=stores(p,600);
 let large=prox(v,60,x=>x>=1500?Math.log1p(Math.min(x,30000)/1500):0),medium=prox(v,100,x=>x>=500?Math.log1p(Math.min(x,6000)/500):0),r50=prox(r,50),h150=dens(h,150);
 let out=[...a,large,medium,r50,prox(m,75),h150,dens(o,150),dens(s,150),Math.log1p(sum(v.filter(([ds,x])=>x>=500).map(([ds,x])=>Math.exp(-ds/(2*250**2))))),h150-a[4],r50*medium];cache.set(key,out);return out;};
}
function decorate(e){e.group=e.f.map(a=>+(a[0]>=Math.exp(-150/180)));e.a=e.f.map((a,i)=>[...a.slice(0,7),Math.log(Math.max(e.gain[800][i],1e-14)),...a.slice(7)]);return e;}
function project(e,indices){return {...e,a:e.a.map(a=>indices.map(i=>a[i]))};}
const base=[0,1,2,3,4,5,6],struct=[...base,7],commercial=[...base,8,9,10,11,12,13,14,15,16,17],full=[...struct,...commercial.slice(7)];
const specs=[
 {name:'reference',indices:base,ridge:.5,shrink:1.5,half:1.5},
 {name:'commercial',indices:commercial,ridge:.5,shrink:1.5,half:1.5},
 {name:'commercial_structural',indices:full,ridge:.5,shrink:1.5,half:1.5}
].map(c=>({...c,mix:true,experts:true,structural:false,dynamic:true}));
function metrics(rows,expert){return expert.stats(rows);}
function fitModel(es,c,cut,expert){let m=expert.fit(es.map(e=>project(e,c.indices)),c,cut);m.cut=es.map(e=>e.c.opened).sort().at(-1);return m;}
function evaluate(e,c,m,rule,expert){return expert.evaluate(project(e,c.indices),c,m,rule);}
function run(raw,lib,expert){
 const d=Object.fromEntries(Object.keys(lib.paths).map(k=>[k,lib.csv(raw[k])])),measure=featureFactory(d,lib),es=lib.prepare(d,measure).map(decorate),base=es.filter(e=>e.split==='train'),test=es.filter(e=>e.split==='time_test');
 // Select by forward validation on pre-2025 openings only; initial 2018–2021 period supplies training.
 let validation=[],models={},rows=[];
 for(let cut of ['2023-01-01','2024-01-01']){
   let end=(+cut.slice(0,4)+1)+'-01-01',train=base.filter(e=>e.c.opened<cut),val=base.filter(e=>e.c.opened>=cut&&e.c.opened<end);if(train.length<4||!val.length)continue;
   for(let c of specs){let m=fitModel(train,c,cut,expert);for(let e of val)for(let rule of ['mode','mass'])validation.push(evaluate(e,c,m,rule,expert));}
 }
 const rank=arr=>{let out=[];for(let c of specs)for(let rule of ['mode','mass']){let a=arr.filter(r=>r.model===c.name&&r.rule===rule);out.push({name:c.name,rule,...metrics(a,expert)});}return out;};
 const cv=rank(validation),selected=[...cv].sort((a,b)=>a.radius_nll-b.radius_nll||b.hit3_300-a.hit3_300)[0];
 const periods=['2025-01-01','2025-07-01','2026-01-01','2026-07-01'];
 for(let cut of periods){let end=periods[periods.indexOf(cut)+1]||'2027-01-01',val=test.filter(e=>e.c.opened>=cut&&e.c.opened<end),past=es.filter(e=>e.c.opened<cut);if(!val.length)continue;models[cut]={};for(let c of specs){let m=fitModel(past,c,cut,expert);models[cut][c.name]=m;for(let e of val){if(m.cut>=e.c.opened)throw Error('Future labels');for(let rule of ['mode','mass'])rows.push(evaluate(e,c,m,rule,expert));}}}
 return {train:base.length,test:test.length,features:names,config:specs,validation:cv,selected,results:rank(rows),models,rows,limitations:'Moscow only, known target tile, modern POIs, historical dated competitors only. Forward validation before 2025 selects spec by radius NLL (same for both rules); rule tie broken by top3. 2025+ labels have already been examined in previous project experiments. No owner, vacancy, rent, or verified usable floor area. Multiscale OSM proxies only.'};
}
if(typeof module!=='undefined')module.exports={run,featureFactory,decorate,project,specs,fitModel,evaluate};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),expert=require('./expert_flexible'),raw=Object.fromEntries(Object.entries(lib.paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib,expert);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/multiscale_benchmark.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({selected:r.selected,results:r.results},null,2));}
