/* Venue-based choice model. No dependencies; Node 18+. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180), MY=111320,T=3600;
const xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY],ll=p=>({lon:p[0]/MX+37.74,lat:p[1]/MY+55.654});
const d2=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
const sum=a=>a.reduce((s,v)=>s+v,0),avg=a=>sum(a)/a.length;
const med=a=>{a=[...a].sort((a,b)=>a-b);let i=a.length>>1;return a.length%2?a[i]:(a[i]+a[i-1])/2};
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const paths={catalog:'data/economy/catalog.csv',split:'results/economy/training_split.csv',venues:'data/economy/venues.csv',shopping:'data/economy/shopping.csv',metro:'data/economy/metro.csv',stores:'data/economy/stores.csv',district:'data/economy/district_100.csv',market:'results/economy/competition.csv',reviewed:'data/clubs_reviewed.csv'};
function csv(t){let rows=[],r=[],f='',q=false;for(let i=0;i<t.length;i++){let c=t[i];if(c==='"'){if(q&&t[i+1]==='"'){f+='"';i++;}else q=!q;}else if(c===','&&!q){r.push(f);f='';}else if(c==='\n'&&!q){r.push(f.replace(/\r$/,''));rows.push(r);r=[];f='';}else f+=c;}if(f||r.length){r.push(f);rows.push(r);}let h=rows.shift();h[0]=h[0].replace(/^\uFEFF/,'');return rows.filter(r=>r.length===h.length).map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])));}
function near(p,arr){let best=Infinity;for(let q of arr)best=Math.min(best,d2(p,q));return Math.sqrt(best);}
function soft(s){let mx=Math.max(...s),w=s.map(v=>Math.exp(v-mx)),z=sum(w);return w.map(v=>v/z);}
function engine(raw){
 let d=Object.fromEntries(Object.keys(raw).map(k=>[k,csv(raw[k])])), clubs=d.catalog.map(r=>({...r,p:xy(r)})), malls=d.shopping.map(xy),metros=d.metro.map(xy),stores=d.stores.map(xy);
 // Candidate universe: commercial/public footprints >=1000 m2 plus all mall POIs, merged within 75m.
 let venues=d.venues.filter(r=>+r.footprint_m2>=1000).map(r=>({p:xy(r),area:+r.footprint_m2,name:r.name||'',source:r.osm_id}));
 for(let p of malls)if(near(p,venues.map(v=>v.p))>75)venues.push({p,area:1000,name:'mall POI',source:'shopping'});
 let cache=new Map();
 function features(p,area){
   let mall=near(p,malls),metro=near(p,metros),density=0;
   for(let q of stores){let ds=d2(p,q);if(ds<1500**2)density+=Math.exp(-ds/(2*500**2));}
   return [Math.log1p(Math.min(area,60000)/1000),Math.exp(-mall/180),Math.exp(-metro/400),Math.log1p(density)];
 }
 function tileInfo(tile){if(cache.has(tile))return cache.get(tile);let [tx,ty]=tile.split(',').map(Number),x=tx*T,y=ty*T,vs=venues.filter(v=>v.p[0]>=x&&v.p[0]<x+T&&v.p[1]>=y&&v.p[1]<y+T);
  // A low-prior 300m background grid prevents excluding all unrecorded premises.
  for(let i=150;i<T;i+=300)for(let j=150;j<T;j+=300)if(near([x+i,y+j],vs.map(v=>v.p))>100)vs.push({p:[x+i,y+j],area:0,name:'grid',source:'grid'});
  let f=vs.map(v=>features(v.p,v.area)),p=vs.map(v=>v.p),grid=[];
  for(let i=100;i<T;i+=200)for(let j=100;j<T;j+=200)grid.push([x+i,y+j]);
  let weights=grid.map(p=>{let v=0;for(let q of stores){let ds=d2(p,q);if(ds<1500**2)v+=Math.exp(-ds/(2*500**2));}return Math.sqrt(1+v);}),total=sum(weights);weights=weights.map(v=>v/total);
  let z={vs,f,p,grid,weights};cache.set(tile,z);return z;
 }
 function gain(p,grid,rho,market,tau){let den=[],own=[],outside=Math.exp(-1.125);for(let x of grid){let a=outside,b=0;for(let c of market){let w=Math.exp(-d2(x,c.p)/(2*tau*tau));a+=w;if(c.network==='DDX')b+=w;}den.push(a);own.push(b);}return p.map(z=>sum(grid.map((x,j)=>{let w=Math.exp(-d2(x,z)/(2*tau*tau));return rho[j]*w/(den[j]+w)*(1-own[j]/den[j]);}))); }
 function event(c,split){let tile=Math.floor(c.p[0]/T)+','+Math.floor(c.p[1]/T),z=tileInfo(tile),past=clubs.filter(r=>r.opened&&r.opened<c.opened&&near(r.p,z.grid)<3000);
   let g=gain(z.p,z.grid,z.weights,past,800),a=z.p.map((p,i)=>[...z.f[i],Math.log(Math.max(g[i],1e-12)),Math.log1p(Math.min(5000,near(p,past.filter(c=>c.network==='DDX').map(c=>c.p)))/500)]);
   let free=z.p.map(p=>past.every(c=>d2(p,c.p)>=150**2)),dist=z.p.map(p=>Math.sqrt(d2(p,c.p)));return {c,split,tile,...z,a,free,dist};}
 let splits=new Map(d.split.map(r=>[r.id,r.split])),es=clubs.filter(c=>c.network==='DDX'&&['train','time_test'].includes(splits.get(c.id))).map(c=>event(c,splits.get(c.id)));
 return {d,clubs,venues,features,gain,es};
}
const cfgs=[...['site','structural'].flatMap(kind=>[.03,.15,.6].flatMap(ridge=>[0,1].map(prior=>({kind,ridge,prior}))))];
function fit(es,cfg){let k=cfg.kind==='site'?4:6,mu=Array(k).fill(0),sd=Array(k).fill(0),n=0;
 for(let e of es)for(let i=0;i<e.a.length;i++)if(e.free[i]){n++;for(let j=0;j<k;j++){mu[j]+=e.a[i][j];sd[j]+=e.a[i][j]**2;}}
 for(let j=0;j<k;j++){mu[j]/=n;sd[j]=Math.max(.05,Math.sqrt(Math.max(0,sd[j]/n-mu[j]**2)));}
 let data=es.map(e=>{let z=e.a.map(a=>a.slice(0,k).map((v,j)=>(v-mu[j])/sd[j])),w=e.dist.map((r,i)=>e.free[i]?-.5*(r/100)**2:-Infinity),target=soft(w);return {z,target,free:e.free,prior:e.vs.map(v=>cfg.prior*(v.area?0:-2))};});
 function obj(b){let l=0,g=Array(k).fill(0);for(let e of data){let s=e.z.map((z,i)=>e.free[i]?dot(z,b)+e.prior[i]:-Infinity),p=soft(s);for(let i=0;i<p.length;i++)if(e.free[i]){if(e.target[i]>0)l-=e.target[i]*Math.log(Math.max(1e-300,p[i]));for(let j=0;j<k;j++)g[j]+=(p[i]-e.target[i])*e.z[i][j];}}return {l:l/data.length+cfg.ridge*dot(b,b)/2,g:g.map((v,j)=>v/data.length+cfg.ridge*b[j])};}
 let b=Array(k).fill(0),o=obj(b);for(let it=0;it<100;it++){let rate=1,ok=false;for(let j=0;j<20;j++){let v=b.map((x,i)=>Math.max(0,Math.min(8,x-rate*o.g[i]))),change=v.map((x,i)=>x-b[i]);if(dot(change,change)<1e-12)return {mu,sd,b};let p=obj(v);if(p.l<=o.l+.0001*dot(change,o.g)){b=v;o=p;ok=true;break;}rate*=.5;}if(!ok)break;}return {mu,sd,b};}
function predict(e,cfg,m){let s=e.a.map((a,i)=>e.free[i]?dot(m.b,a.slice(0,m.b.length).map((v,j)=>(v-m.mu[j])/m.sd[j]))+cfg.prior*(e.vs[i].area?0:-2):-Infinity),p=soft(s),mass=e.p.map((z,i)=>e.free[i]?sum(p.map((v,j)=>d2(z,e.p[j])<=300**2?v:0)):-Infinity),mode=s.indexOf(Math.max(...s)),top=mass.indexOf(Math.max(...mass)),center=[sum(e.p.map((z,i)=>z[0]*p[i])),sum(e.p.map((z,i)=>z[1]*p[i]))],mse=e.p.reduce((best,z,i)=>e.free[i]&&(best<0||d2(z,center)<d2(e.p[best],center))?i:best,-1),picked=[];
 for(let i of [...mass.keys()].sort((a,b)=>mass[b]-mass[a]||s[b]-s[a]))if(e.free[i]&&picked.every(j=>d2(e.p[i],e.p[j])>=600**2)){picked.push(i);if(picked.length===3)break;}
 return {top,mode,mse,picked,p,mass};}
function evaluate(e,c,m){let a=predict(e,c,m),ds=i=>e.dist[i];return {name:e.c.name,tile:e.tile,top1_m:ds(a.top),mode_m:ds(a.mode),mse_m:ds(a.mse),top3_m:Math.min(...a.picked.map(ds)),hit1:+(ds(a.top)<=300),hit3:+(Math.min(...a.picked.map(ds))<=300),p_actual_300:sum(a.p.filter((v,i)=>ds(i)<=300)),nearest_eligible_m:Math.min(...e.dist.filter((v,i)=>e.free[i]))};}
function stats(r){return {n:r.length,hit1_300:sum(r.map(x=>x.hit1)),hit3_300:sum(r.map(x=>x.hit3)),median_top1_m:med(r.map(x=>x.top1_m)),median_top3_m:med(r.map(x=>x.top3_m)),rmse_m:Math.sqrt(avg(r.map(x=>x.mse_m**2))),median_mode_m:med(r.map(x=>x.mode_m)),mean_radius_log_loss:avg(r.map(x=>-Math.log(Math.max(x.p_actual_300,1e-12)))),candidate_coverage_300:sum(r.map(x=>+(x.nearest_eligible_m<=300)))};}
function run(raw){let E=engine(raw),train=E.es.filter(e=>e.split==='train'),test=E.es.filter(e=>e.split==='time_test'),tiles=[...new Set(train.map(e=>e.tile))].sort(),fold=new Map(tiles.map((t,i)=>[t,i%4])),cv=[];
 for(let c of cfgs){let rows=[];for(let k=0;k<4;k++){let m=fit(train.filter(e=>fold.get(e.tile)!==k),c);rows.push(...train.filter(e=>fold.get(e.tile)===k).map(e=>evaluate(e,c,m)));}cv.push({...c,...stats(rows)});}
 const order=(a,b)=>b.hit1_300-a.hit1_300||a.mean_radius_log_loss-b.mean_radius_log_loss;
 let models={};for(let kind of ['site','structural']){let c=cv.filter(x=>x.kind===kind).sort(order)[0],m=fit(train,c),rows=test.map(e=>evaluate(e,c,m));models[kind]={config:c,coef:m,test:stats(rows),details:rows};}
 let selected=[...cv].sort(order)[0];return {train:train.length,test:test.length,cv,selected,models,limitations:'Conditional on actual tile; current OSM; unknown historical availability and undated rival dates. Grid fallback. Selection by training CV top1 hit then radius log loss; historical set previously reused.'};}
if(typeof module!=='undefined')module.exports={run,paths,engine,fit,predict,evaluate,stats};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),raw=Object.fromEntries(Object.entries(paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/venue_benchmark.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({selected:r.selected,models:r.models},null,2));}
