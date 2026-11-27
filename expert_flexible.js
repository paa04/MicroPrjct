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
{name:'recent_joint',dynamic:true,half:1.5,mix:false,structural:false},
{name:'balanced_joint',dynamic:true,half:1.5,mix:false,structural:false,balance:true},
{name:'expert_static',dynamic:false,half:Infinity,mix:true,structural:false,experts:true},
{name:'expert_recent',dynamic:true,half:1.5,mix:true,structural:false,experts:true},
{name:'expert_structural',dynamic:true,half:1.5,mix:true,structural:true,experts:true}
];
const years=(a,b)=>(Date.parse(a)-Date.parse(b))/(365.25*86400000);
function decorate(e){
 e.group=e.f.map(a=>+(a[0]>=MALL_CUT));
 e.a=e.f.map((a,i)=>[...a,Math.log(Math.max(e.gain[800][i],1e-14))]);
 return e;
}
function fit(events,c,cut){
 const es=events.filter(e=>e.allowed[e.target]),n=es.length,k=es[0].a[0].length;
 if(!n)throw Error('No eligible observations');
 let w=es.map(e=>Number.isFinite(c.half)?2**(-years(cut,e.c.opened)/c.half):1); const W0=sum(w),G0=sum(es.map((e,i)=>w[i]*e.group[e.target])); if(c.balance){w=w.map((v,i)=>v*(es[i].group[es[i].target]?W0/(2*Math.max(G0,1e-9)):W0/(2*Math.max(W0-G0,1e-9))));}const W=sum(w);
 let mu=new Float64Array(k),sd=new Float64Array(k),N=0;
 for(let e of es)for(let i=0;i<e.a.length;i++)if(e.allowed[i]){N++;for(let j=0;j<k;j++){let v=e.a[i][j];mu[j]+=v;sd[j]+=v*v;}}
 for(let j=0;j<k;j++){mu[j]/=N;sd[j]=Math.max(.01,Math.sqrt(Math.max(0,sd[j]/N-mu[j]**2)));}
 let data=es.map((e,j)=>{let ix=[];for(let i=0;i<e.a.length;i++)if(e.allowed[i]&&(!c.mix||e.group[i]===e.group[e.target]))ix.push(i);let z=new Float64Array(ix.length*k),target=new Float64Array(k);for(let l=0;l<ix.length;l++)for(let h=0;h<k;h++)z[l*k+h]=(e.a[ix[l]][h]-mu[h])/sd[h];for(let h=0;h<k;h++)target[h]=(e.a[e.target][h]-mu[h])/sd[h];return {z,target,n:ix.length,w:w[j]/W,group:e.group[e.target]};});
 function objective(b,group=-1,prior=null){let g=new Float64Array(k),loss=0,ws=group<0?1:sum(data.filter(e=>e.group===group).map(e=>e.w));for(let e of data){if(group>=0&&e.group!==group)continue;let ew=e.w/Math.max(ws,1e-12);let s=new Float64Array(e.n),mx=-Infinity;for(let i=0;i<e.n;i++){let v=0;for(let h=0;h<k;h++)v+=e.z[i*k+h]*b[h];s[i]=v;if(v>mx)mx=v;}let den=0;for(let i=0;i<e.n;i++){s[i]=Math.exp(s[i]-mx);den+=s[i];}
 let v=0;for(let h=0;h<k;h++){v+=e.target[h]*b[h];g[h]-=ew*e.target[h];}loss+=ew*(mx+Math.log(den)-v);
 for(let i=0;i<e.n;i++){let p=ew*s[i]/den;for(let h=0;h<k;h++)g[h]+=p*e.z[i*k+h];}}
 for(let h=0;h<k;h++){let v=b[h]-(prior?prior[h]:0),lambda=prior?(c.shrink??1.5):(c.ridge??.5);loss+=lambda*v*v/2;g[h]+=lambda*v;}return {loss,g};}
 function solve(group=-1,prior=null){let b=prior?[...prior]:Array(k).fill(0),o=objective(b,group,prior);
 for(let it=0;it<22;it++){let rate=1,ok=false;for(let j=0;j<15;j++){let v=b.map((x,h)=>Math.max(group<0?0:-6,Math.min(6,x-rate*o.g[h]))),norm=sum(v.map((x,h)=>(x-b[h])**2));if(norm<1e-12){it=22;break;}let q=objective(v,group,prior),slope=sum(v.map((x,h)=>(x-b[h])*o.g[h]));if(q.loss<=o.loss+.0001*slope){b=v;o=q;ok=true;break;}rate*=.5;}if(!ok)break;}return b;}
 let b=solve(),expert=c.experts?[solve(0,b),solve(1,b)]:null;
 // Group prior uses only observed locations of earlier openings, not target address text.
 let gate=(2+sum(es.map((e,i)=>w[i]*e.group[e.target])))/(4+W);
 return {mu:[...mu],sd:[...sd],b,expert,gate,n,ess:W*W/sum(w.map(x=>x*x)),cut};
}
function prob(e,c,m){
 const score=e.a.map((a,i)=>e.allowed[i]?sum((m.expert?m.expert[e.group[i]]:m.b).map((b,j)=>b*(a[j]-m.mu[j])/m.sd[j])):-Infinity);
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
 for(let j=0;j<3;j++){let best=-1;for(let i=0;i<m.length;i++)if(e.allowed[i]&&(rule!=='diverse'||j!==1||e.group[i]!==e.group[out[0]])&&out.every(h=>sq(e.p[i],e.p[h])>=600**2)&&(best<0||m[i]>m[best]))best=i;if(best>=0)out.push(best);}
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
   for(let e of val){if(m.cut>=e.c.opened)throw Error('Temporal leakage');for(let rule of ['mode','mass','diverse'])rows.push(evaluate(e,c,m,rule));}}
 }
 let summary={};for(let c of config)for(let rule of ['mode','mass','diverse']){let a=rows.filter(r=>r.model===c.name&&r.rule===rule);summary[c.name+'_'+rule]={all:stats(a),year2025:stats(a.filter(r=>r.date<'2026-01-01')),year2026:stats(a.filter(r=>r.date>='2026-01-01'))};}
 return {train:base.length,test:test.length,config:config.map(c=>({...c,half:Number.isFinite(c.half)?c.half:null})),periods,distribution:{train:audit(base),test:audit(test)},summary,models,rows,limitations:'Moscow only, Maryino and guard tiles excluded from learning and scoring. Historical tile known, contemporary POIs, historical rivals without opening dates excluded, retrospective protocol on previously examined data. Six-month updates use only earlier completed periods. Address tags are audit only, not features. Group defined by grid-cell mall proximity, not verified building category. Configs exploratory, no fresh independent test.'};
}
if(typeof module!=='undefined')module.exports={run,decorate,fit,prob,pick,evaluate,stats,config};
