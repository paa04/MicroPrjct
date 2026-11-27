/* Dynamic Moscow-only choice model; chronological evaluation with no future labels. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320;
const sum=a=>a.reduce((s,v)=>s+v,0),sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
const median=a=>{let b=[...a].sort((x,y)=>x-y),i=b.length>>1;return b.length%2?b[i]:(b[i-1]+b[i])/2;};
const ll=p=>({lon:p[0]/MX+37.74,lat:p[1]/MY+55.654});
const tagged=r=>/(?:^|[\s,])(?:ТЦ|ТРЦ|ТРК|ТК|РЦ|Фудмол)(?=[\s«"]|$)/iu.test(r.address||'');
const MALL_CUT=Math.exp(-150/180);
const config=[
 {name:'static_joint',dynamic:false,half:Infinity,mix:false,structural:false},
 {name:'expanding_joint',dynamic:true,half:Infinity,mix:false,structural:false},
 {name:'recent_joint',dynamic:true,half:1.5,mix:false,structural:false},
 {name:'static_mixture',dynamic:false,half:Infinity,mix:true,structural:false},
 {name:'dynamic_mixture',dynamic:true,half:1.5,mix:true,structural:false},
 {name:'dynamic_structural',dynamic:true,half:1.5,mix:true,structural:true}
];
const years=(a,b)=>(Date.parse(a)-Date.parse(b))/(365.25*86400000);
function decorate(e){
 e.group=e.f.map(a=>+(a[0]>=MALL_CUT));
 e.a=e.f.map((a,i)=>[...a,Math.log(Math.max(e.gain[800][i],1e-14))]);
 return e;
}
function fit(events,c,cut){
 const es=events.filter(e=>e.allowed[e.target]),n=es.length,k=c.structural?8:7;
 if(!n)throw Error('No eligible observations');
 const w=es.map(e=>Number.isFinite(c.half)?2**(-years(cut,e.c.opened)/c.half):1),W=sum(w);
 let mu=new Float64Array(k),sd=new Float64Array(k),N=0;
 for(let e of es)for(let i=0;i<e.a.length;i++)if(e.allowed[i]){N++;for(let j=0;j<k;j++){let v=e.a[i][j];mu[j]+=v;sd[j]+=v*v;}}
 for(let j=0;j<k;j++){mu[j]/=N;sd[j]=Math.max(.01,Math.sqrt(Math.max(0,sd[j]/N-mu[j]**2)));}
 let data=es.map((e,j)=>{let ix=[];for(let i=0;i<e.a.length;i++)if(e.allowed[i]&&(!c.mix||e.group[i]===e.group[e.target]))ix.push(i);let z=new Float64Array(ix.length*k),target=new Float64Array(k);for(let l=0;l<ix.length;l++)for(let h=0;h<k;h++)z[l*k+h]=(e.a[ix[l]][h]-mu[h])/sd[h];for(let h=0;h<k;h++)target[h]=(e.a[e.target][h]-mu[h])/sd[h];return {z,target,n:ix.length,w:w[j]/W};});
 function objective(b){let g=new Float64Array(k),loss=0;for(let e of data){let s=new Float64Array(e.n),mx=-Infinity;for(let i=0;i<e.n;i++){let v=0;for(let h=0;h<k;h++)v+=e.z[i*k+h]*b[h];s[i]=v;if(v>mx)mx=v;}let den=0;for(let i=0;i<e.n;i++){s[i]=Math.exp(s[i]-mx);den+=s[i];}
 let v=0;for(let h=0;h<k;h++){v+=e.target[h]*b[h];g[h]-=e.w*e.target[h];}loss+=e.w*(mx+Math.log(den)-v);
 for(let i=0;i<e.n;i++){let p=e.w*s[i]/den;for(let h=0;h<k;h++)g[h]+=p*e.z[i*k+h];}}
 for(let h=0;h<k;h++){loss+=.25*b[h]**2;g[h]+=.5*b[h];}return {loss,g};}
 let b=Array(k).fill(0),o=objective(b);
 for(let it=0;it<50;it++){let rate=1,ok=false;for(let j=0;j<15;j++){let v=b.map((x,h)=>Math.max(0,Math.min(6,x-rate*o.g[h]))),norm=sum(v.map((x,h)=>(x-b[h])**2));if(norm<1e-12){it=50;break;}let q=objective(v),slope=sum(v.map((x,h)=>(x-b[h])*o.g[h]));if(q.loss<=o.loss+.0001*slope){b=v;o=q;ok=true;break;}rate*=.5;}if(!ok)break;}
 // Group prior uses only observed locations of earlier openings, not target address text.
 let gate=(2+sum(es.map((e,i)=>w[i]*e.group[e.target])))/(4+W);
 return {mu:[...mu],sd:[...sd],b,gate,n,ess:W*W/sum(w.map(x=>x*x)),cut};
}
function prob(e,c,m){
 const score=e.a.map((a,i)=>e.allowed[i]?sum(m.b.map((b,j)=>b*(a[j]-m.mu[j])/m.sd[j])):-Infinity);
 let p=Array(score.length).fill(0);
 for(let group of c.mix?[0,1]:[-1]){
  let idx=score.map((v,i)=>Number.isFinite(v)&&(group<0||e.group[i]===group)?i:-1).filter(i=>i>=0);if(!idx.length)continue;
  let mx=Math.max(...idx.map(i=>score[i])),den=sum(idx.map(i=>Math.exp(score[i]-mx))),mass=group<0?1:group===1?m.gate:1-m.gate;
  for(let i of idx)p[i]=mass*Math.exp(score[i]-mx)/den;
 }
 let den=sum(p);return p.map(v=>v/den);
}
function pick(e,p,rule){
 let lookup=new Map(e.p.map((z,i)=>[z.join(','),i])),off=[];for(let x=-300;x<=300;x+=100)for(let y=-300;y<=300;y+=100)if(x*x+y*y<=90000)off.push([x,y]);
 let m=rule==='mode'?p:e.p.map(z=>sum(off.map(([x,y])=>p[lookup.get([z[0]+x,z[1]+y].join(','))]||0))),out=[];
 for(let j=0;j<3;j++){let best=-1;for(let i=0;i<m.length;i++)if(e.allowed[i]&&out.every(h=>sq(e.p[i],e.p[h])>=600**2)&&(best<0||m[i]>m[best]))best=i;if(best>=0)out.push(best);}
 return out;
}
function evaluate(e,c,m,rule){
 let p=prob(e,c,m),idx=pick(e,p,rule),ds=idx.map(i=>Math.sqrt(sq(e.p[i],e.c.p))),near=sum(p.filter((_,i)=>sq(e.p[i],e.c.p)<=90000));
 return {name:e.c.name,date:e.c.opened,tile:e.tile,model:c.name,rule,train_n:m.n,train_max_date:m.cut,gate:m.gate,tagged:tagged(e.c),mall_group:e.group[e.target],top1_m:ds[0],top3_m:Math.min(...ds),hit1:+(ds[0]<=300),hit3:+(Math.min(...ds)<=300),logloss:-Math.log(Math.max(p[e.target],1e-300)),radius_logloss:-Math.log(Math.max(near,1e-300)),masked:!e.allowed[e.target],predictions:idx.map(i=>ll(e.p[i]))};
}
function stats(rows){return {n:rows.length,hit1_300:sum(rows.map(r=>r.hit1)),hit3_300:sum(rows.map(r=>r.hit3)),median_top1_m:median(rows.map(r=>r.top1_m)),median_top3_m:median(rows.map(r=>r.top3_m)),nll:sum(rows.map(r=>r.logloss))/rows.length,radius_nll:sum(rows.map(r=>r.radius_logloss))/rows.length,masked:sum(rows.map(r=>+r.masked))};}
function audit(es){return {n:es.length,address_tag:sum(es.map(e=>+tagged(e.c))),near_mall_150:sum(es.map(e=>e.group[e.target])),near_mall_300:sum(es.map(e=>+(e.f[e.target][0]>=Math.exp(-300/180)))),address_tag_vs_mall:es.map(e=>({name:e.c.name,date:e.c.opened,address_tag:tagged(e.c),mall_distance_proxy_m:e.f[e.target][0]>0?-180*Math.log(e.f[e.target][0]):null}))};}
function run(raw,lib){
 const d=Object.fromEntries(Object.keys(lib.paths).map(k=>[k,lib.csv(raw[k])])),es=lib.prepare(d,lib.featureFactory(d)).map(decorate),base=es.filter(e=>e.split==='train'),test=es.filter(e=>e.split==='time_test').sort((a,b)=>a.c.opened.localeCompare(b.c.opened));
 const periods=['2025-01-01','2025-07-01','2026-01-01','2026-07-01'];let rows=[],models={};
 for(let cut of periods){let end=periods[periods.indexOf(cut)+1]||'2027-01-01',val=test.filter(e=>e.c.opened>=cut&&e.c.opened<end);if(!val.length)continue;let past=es.filter(e=>e.c.opened<cut);
  models[cut]={};for(let c of config){let data=c.dynamic?past:base,fitCut=c.dynamic?cut:'2025-01-01',m=fit(data,c,fitCut);m.cut=data.map(e=>e.c.opened).sort().at(-1);models[cut][c.name]=m;
   for(let e of val){if(m.cut>=e.c.opened)throw Error('Temporal leakage');for(let rule of ['mode','mass'])rows.push(evaluate(e,c,m,rule));}}
 }
 let summary={};for(let c of config)for(let rule of ['mode','mass']){let a=rows.filter(r=>r.model===c.name&&r.rule===rule);summary[c.name+'_'+rule]={all:stats(a),year2025:stats(a.filter(r=>r.date<'2026-01-01')),year2026:stats(a.filter(r=>r.date>='2026-01-01'))};}
 return {train:base.length,test:test.length,config:config.map(c=>({...c,half:Number.isFinite(c.half)?c.half:null})),periods,distribution:{train:audit(base),test:audit(test)},summary,models,rows,limitations:'Moscow only, Maryino and guard tiles excluded from learning and scoring. Historical tile known, contemporary POIs, historical rivals without opening dates excluded, retrospective protocol on previously examined data. Six-month updates use only earlier completed periods. Address tags are audit only, not features. Group defined by grid-cell mall proximity, not verified building category. Configs exploratory, no fresh independent test.'};
}
if(typeof module!=='undefined')module.exports={run,decorate,fit,prob,pick,evaluate,stats,config};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),raw=Object.fromEntries(Object.entries(lib.paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/dynamic_benchmark.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({distribution:r.distribution,summary:r.summary},null,2));}
