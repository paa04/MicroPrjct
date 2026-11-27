/* Inverse spatial competition experiment. Node >=18, no dependencies.
   Run from repository root: node inverse_experiment.js
   The same pure run() function can be evaluated on repository CSV snapshots. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180), MY=111320, TILE=3600, STEP=100;
const sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
const xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY];
const ll=p=>({lon:p[0]/MX+37.74,lat:p[1]/MY+55.654});
const sum=a=>a.reduce((s,v)=>s+v,0);
const dot=(a,b)=>{let s=0;for(let j=0;j<a.length;j++)s+=a[j]*b[j];return s;};
const mean=a=>sum(a)/a.length;
const median=a=>{let b=[...a].sort((x,y)=>x-y),m=b.length>>1;return b.length%2?b[m]:(b[m-1]+b[m])/2;};
function csv(t){
  if(!t) throw Error('Missing CSV input');
  let rows=[], row=[], field='', q=false;
  for(let i=0;i<t.length;i++){let c=t[i];if(c==='"'){if(q&&t[i+1]==='"'){field+='"';i++;}else q=!q;}
    else if(c===','&&!q){row.push(field);field='';}
    else if(c==='\n'&&!q){row.push(field.replace(/\r$/,''));rows.push(row);row=[];field='';}else field+=c;}
  if(field||row.length){row.push(field.replace(/\r$/,''));rows.push(row);}
  let h=rows.shift();h[0]=h[0].replace(/^\uFEFF/,'');
  return rows.filter(r=>r.length===h.length).map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])));
}
const paths={catalog:'data/economy/catalog.csv',split:'results/economy/training_split.csv',venues:'data/economy/venues.csv',shopping:'data/economy/shopping.csv',retail:'data/economy/retail.csv',metro:'data/economy/metro.csv',stores:'data/economy/stores.csv',office:'data/economy/office.csv',district:'data/economy/district_100.csv',market:'results/economy/competition.csv',reviewed:'data/clubs_reviewed.csv'};
function index(rows, value){
  let buckets=new Map(), B=500;
  for(let r of rows){let p=xy(r);if(!p.every(Number.isFinite))continue;let k=Math.floor(p[0]/B)+','+Math.floor(p[1]/B);if(!buckets.has(k))buckets.set(k,[]);buckets.get(k).push([p,value(r)]);}
  return (p,radius)=>{let out=[], x=Math.floor(p[0]/B),y=Math.floor(p[1]/B),n=Math.ceil(radius/B),r2=radius*radius;
    for(let i=x-n;i<=x+n;i++)for(let j=y-n;j<=y+n;j++)for(let [z,v] of buckets.get(i+','+j)||[]){let d=sq(p,z);if(d<=r2)out.push([d,v]);}return out;};
}
function featureFactory(d){
  let venue=index(d.venues,r=>Math.log1p(Math.min(30000,+r.footprint_m2||0)/2000)),
      mall=index(d.shopping,()=>1),retail=index(d.retail,()=>1),metro=index(d.metro,()=>1),
      stores=index(d.stores,()=>1),office=index(d.office,r=>Math.min(85,Math.max(1,+r.value||4)));
  const prox=(ix,p,h)=>{let best=0;for(let [ds,w] of ix(p,6*h))best=Math.max(best,w*Math.exp(-Math.sqrt(ds)/h));return best;};
  const dens=(ix,p,h)=>{let v=0;for(let [ds,w] of ix(p,4*h))v+=w*Math.exp(-ds/(2*h*h));return Math.log1p(v);};
  return p=>[prox(mall,p,180),prox(retail,p,125),prox(metro,p,250),prox(venue,p,150),dens(stores,p,500),dens(office,p,500)];
}
function weights(f, mode='proxy'){
  let a=f.map(x=>mode==='uniform'?1:Math.exp(.75*x[4]+.25*x[5])), s=sum(a);return a.map(v=>v/s);
}
function available(sites, clubs){return sites.map(p=>clubs.every(c=>sq(p,c.p)>=150**2));}
function kernel(points,sites,tau){
  let n=points.length, a=new Float64Array(n*sites.length), scale=2*tau*tau;
  for(let i=0;i<sites.length;i++)for(let j=0;j<n;j++)a[i*n+j]=Math.exp(-sq(sites[i],points[j])/scale);
  return a;
}
function capture(points,sites,clubs,rho,tau,K=null,network='DDX'){
  // Gaussian distance utility; outside option corresponds to 1200 metres.
  let n=points.length, den=new Float64Array(n), own=new Float64Array(n), gross=[],gain=[], outside=Math.exp(-(1200**2)/(2*tau*tau));
  for(let j=0;j<n;j++){den[j]=outside;for(let c of clubs){let w=Math.exp(-sq(points[j],c.p)/(2*tau*tau));den[j]+=w;if(c.network===network)own[j]+=w;}}
  for(let i=0;i<sites.length;i++){let g=0,b=0;for(let j=0;j<n;j++){let w=K?K[i*n+j]:Math.exp(-sq(sites[i],points[j])/(2*tau*tau)),p=w/(den[j]+w);b+=rho[j]*p;g+=rho[j]*p*(1-own[j]/den[j]);}gross.push(b);gain.push(g);}
  return {gain,gross};
}
function prepare(d,measure){
  const tiles=new Map(), splits=new Map(d.split.map(r=>[r.id,r.split]));
  const clubs=d.catalog.map(r=>({...r,p:xy(r)}));
  let events=[];
  for(let c of clubs){let split=splits.get(c.id);if(c.network!=='DDX'||!['train','time_test'].includes(split)||!/^\d{4}-\d{2}-\d{2}$/.test(c.opened))continue;
    let tx=Math.floor(c.p[0]/TILE),ty=Math.floor(c.p[1]/TILE),tile=tx+','+ty;
    if(!tiles.has(tile)){let p=[];for(let y=50;y<TILE;y+=STEP)for(let x=50;x<TILE;x+=STEP)p.push([tx*TILE+x,ty*TILE+y]);let f=p.map(measure);tiles.set(tile,{p,f,rho:weights(f)});}
    let {p,f,rho}=tiles.get(tile),target=nearest(p,c.p);
    // No undated Spirit clubs from the future; all dated preceding openings in the buffer.
    let past=clubs.filter(r=>r.opened&&r.opened<c.opened&&Math.hypot(Math.max(tx*TILE-r.p[0],r.p[0]-(tx+1)*TILE,0),Math.max(ty*TILE-r.p[1],r.p[1]-(ty+1)*TILE,0))<=3000);
    let allowed=available(p,past),gain={};
    for(let tau of [400,800])gain[tau]=capture(p,p,past,rho,tau).gain;
    events.push({c,split,tile,p,f,allowed,target,gain});
  }return events;
}
function nearest(p,z,allowed=null){let best=-1,dist=Infinity;for(let i=0;i<p.length;i++){if(allowed&&!allowed[i])continue;let d=sq(p[i],z);if(d<dist){dist=d;best=i;}}if(best<0)throw Error('No eligible candidate');return best;}
const cfgs=[{variant:'center',ridge:0,tau:0},{variant:'site',ridge:.1,tau:0},{variant:'site',ridge:.5,tau:0},{variant:'proxy_joint',ridge:.1,tau:0},{variant:'proxy_joint',ridge:.5,tau:0},...([400,800].flatMap(tau=>[.1,.5].map(ridge=>({variant:'inverse',ridge,tau}))))];
function design(f,g,c){return c.variant==='center'?f.map(()=>[]):c.variant==='site'?f.map(a=>a.slice(0,4)):c.variant==='proxy_joint'?f:f.map((a,i)=>[...a,Math.log(Math.max(g[i],1e-14))]);}
function fit(es,c){
  if(c.variant==='center')return {mu:[],sd:[],b:[]};
  let usable=es.filter(e=>e.allowed[e.target]), arrays=usable.map(e=>design(e.f,e.gain[c.tau],c)), k=arrays[0][0].length,mu=Array(k).fill(0),sd=Array(k).fill(0),n=0;
  for(let j=0;j<usable.length;j++)for(let i=0;i<arrays[j].length;i++)if(usable[j].allowed[i]){n++;for(let h=0;h<k;h++){let v=arrays[j][i][h];mu[h]+=v;sd[h]+=v*v;}}
  for(let h=0;h<k;h++){mu[h]/=n;sd[h]=Math.max(.01,Math.sqrt(Math.max(0,sd[h]/n-mu[h]**2)));}
  let z=arrays.map((a,j)=>a.map((r,i)=>usable[j].allowed[i]?r.map((v,h)=>(v-mu[h])/sd[h]):null));
  function objective(b){
    let loss=0,grad=Array(k).fill(0);
    for(let j=0;j<z.length;j++){let a=z[j],t=a[usable[j].target],s=a.map(r=>r?dot(r,b):-Infinity),mx=Math.max(...s),w=s.map(v=>Math.exp(v-mx)),den=sum(w);loss+=mx+Math.log(den)-dot(t,b);
      for(let h=0;h<k;h++)grad[h]-=t[h];
      for(let i=0;i<a.length;i++)if(a[i])for(let h=0;h<k;h++)grad[h]+=w[i]/den*a[i][h];}
    return {loss:loss/z.length+c.ridge*dot(b,b)/2,grad:grad.map((g,h)=>g/z.length+c.ridge*b[h])};
  }
  let b=Array(k).fill(0),o=objective(b);
  for(let it=0;it<100;it++){let step=1,accepted=false;for(let j=0;j<20;j++){let b2=b.map((v,h)=>Math.max(0,Math.min(6,v-step*o.grad[h]))),change=b2.map((v,h)=>v-b[h]);if(dot(change,change)<1e-12)return {mu,sd,b};let o2=objective(b2);if(o2.loss<=o.loss+1e-4*dot(o.grad,change)){b=b2;o=o2;accepted=true;break;}step*=.5;}if(!accepted)break;}
  return {mu,sd,b};
}
function score(f,g,allowed,c,m){return design(f,g,c).map((a,i)=>allowed[i]?dot(a.map((v,h)=>(v-m.mu[h])/m.sd[h]),m.b):-Infinity);}
function posterior(s){let mx=Math.max(...s);if(!Number.isFinite(mx))throw Error('No eligible site');let a=s.map(v=>Math.exp(v-mx)),den=sum(a);return a.map(v=>v/den);}
function action(p,s){
  let prob=posterior(s),allowed=s.map(Number.isFinite),center=[0,0];for(let i=0;i<p.length;i++){center[0]+=p[i][0]*prob[i];center[1]+=p[i][1]*prob[i];}
  let mode=s.indexOf(Math.max(...s)), mse=nearest(p,center,allowed), mass=p.map((z,i)=>allowed[i]?prob.reduce((a,v,j)=>a+(sq(z,p[j])<=300**2?v:0),0):-Infinity),
  picked=[],res=[...prob];
  for(let k=0;k<3;k++){let best=-1,val=-1;for(let i=0;i<p.length;i++)if(allowed[i]&&picked.every(j=>sq(p[i],p[j])>=600**2)){
      let a=0;for(let j=0;j<p.length;j++)if(sq(p[i],p[j])<=300**2)a+=res[j];if(a>val+1e-14||(Math.abs(a-val)<1e-14&&best>=0&&prob[i]>prob[best])){val=a;best=i;}}
    if(best<0)break;picked.push(best);for(let j=0;j<p.length;j++)if(sq(p[best],p[j])<=300**2)res[j]=0;}
  return {mode,mse,picked,prob,mass};
}
function evaluate(e,c,m,full=true){
  let s=score(e.f,e.gain[c.tau],e.allowed,c,m),p=posterior(s),center=[0,0];for(let i=0;i<e.p.length;i++){center[0]+=e.p[i][0]*p[i];center[1]+=e.p[i][1]*p[i];}
  let mse=nearest(e.p,center,e.allowed),mode=s.indexOf(Math.max(...s)), row={name:e.c.name,date:e.c.opened,tile:e.tile,masked:!e.allowed[e.target],mse_m:Math.sqrt(sq(e.p[mse],e.c.p)),mode_m:Math.sqrt(sq(e.p[mode],e.c.p)),nll:e.allowed[e.target]?-Math.log(Math.max(p[e.target],1e-300)):null};
  if(full){let a=action(e.p,s);row.top1_m=Math.sqrt(sq(e.p[a.picked[0]],e.c.p));row.top3_m=Math.min(...a.picked.map(i=>Math.sqrt(sq(e.p[i],e.c.p))));row.hit1_300=+(row.top1_m<=300);row.hit3_300=+(row.top3_m<=300);}
  return row;
}
function summarize(rows){let nll=rows.filter(r=>r.nll!==null).map(r=>r.nll);let o={n:rows.length,masked:rows.filter(r=>r.masked).length,rmse_m:Math.sqrt(mean(rows.map(r=>r.mse_m**2))),median_mse_action_m:median(rows.map(r=>r.mse_m)),median_mode_m:median(rows.map(r=>r.mode_m)),mean_nll:nll.length?mean(nll):null,nll_n:nll.length};if('top1_m' in rows[0])Object.assign(o,{median_top1_m:median(rows.map(r=>r.top1_m)),median_top3_m:median(rows.map(r=>r.top3_m)),hit1_300:sum(rows.map(r=>r.hit1_300)),hit3_300:sum(rows.map(r=>r.hit3_300))});return o;}
function local(d,measure,c,m){
  let cells=d.district.map(r=>[+r.x_m,+r.y_m]),f=cells.map(measure),idx=d.district.map((r,i)=>r.allowed==='1'?i:-1).filter(i=>i>=0),sites=idx.map(i=>cells[i]),sf=idx.map(i=>f[i]);
  let market=d.market.map(r=>({...r,p:xy(r)})),all=d.reviewed.filter(r=>r.eligible==='yes').map(r=>({...r,p:xy(r)}));
  // Match reviewed pins to market pins within 300m so the hidden club is truly removed.
  let extra=all.filter(r=>!market.some(t=>sq(t.p,r.p)<300**2&&t.network===r.network)),allowed=available(sites,[...market,...extra]);
  let proxy=weights(f),uniform=weights(f,'uniform'),housing=d.district.map(r=>+.6*r.housing+.15*r.office+.15*r.metro+.1*r.retail),hm=sum(housing);housing=housing.map(x=>x/hm);
  let scenario=[], predGain;
  for(let [label,rho] of [['proxy',proxy],['uniform',uniform],['legacy_housing_mix',housing]])for(let tau of [400,800,1200]){
    let z=capture(cells,sites,market,rho,tau);scenario.push({label,tau,...z});if(label==='proxy'&&tau===c.tau)predGain=z.gain;}
  let s=score(sf,predGain,allowed,c,m),a=action(sites,s),best=scenario.map(z=>Math.max(...z.gain.filter((_,i)=>allowed[i])));
  let regret=sites.map((_,i)=>allowed[i]?Math.max(...scenario.map((z,k)=>(best[k]-z.gain[i])/Math.max(best[k],1e-12))):Infinity),econ=regret.indexOf(Math.min(...regret));
  const describe=i=>({...ll(sites[i]),p_300:a.mass[i],worst_relative_regret:regret[i],scenarios:scenario.map(z=>({prior:z.label,tau:z.tau,net_gain_pp:100*z.gain[i],gross_pp:100*z.gross[i],own_transfer_pp:100*(z.gross[i]-z.gain[i])}))});
  let loo=[];for(let club of market.filter(r=>r.scope==='inside'&&r.network==='DDX')){let rest=market.filter(r=>r!==club),free=available(sites,[...rest,...extra]),g=c.tau?capture(cells,sites,rest,proxy,c.tau).gain:undefined;loo.push(evaluate({c:club,tile:'Maryino',p:sites,f:sf,allowed:free,target:nearest(sites,club.p),gain:{[c.tau]:g}},c,m));}
  return {eligible:allowed.filter(Boolean).length,predictive_mse:describe(a.mse),predictive_top3:a.picked.map(describe),economic_minimax:describe(econ),loo};
}
function run(raw){
  const d=Object.fromEntries(Object.keys(paths).map(k=>[k,csv(raw[k])])),measure=featureFactory(d),es=prepare(d,measure),train=es.filter(e=>e.split==='train'),test=es.filter(e=>e.split==='time_test');
  if(train.length<8||test.length<1)throw Error('Insufficient openings');
  let tiles=[...new Set(train.map(e=>e.tile))].sort(),fold=new Map(tiles.map((t,i)=>[t,i%4])),cv=[];
  for(let c of cfgs){let rows=[];for(let k=0;k<4;k++){let model=fit(train.filter(e=>fold.get(e.tile)!==k),c);rows.push(...train.filter(e=>fold.get(e.tile)===k).map(e=>evaluate(e,c,model,false)));}cv.push({...c,...summarize(rows)});}
  let selected=[...cv].sort((a,b)=>a.rmse_m-b.rmse_m)[0], variants=['center','site','proxy_joint','inverse'], models={}, details={};
  for(let v of variants){let c=cv.filter(r=>r.variant===v).sort((a,b)=>a.rmse_m-b.rmse_m)[0],m=fit(train,c),rows=test.map(e=>evaluate(e,c,m));models[v]={config:c,model:m,test:summarize(rows)};details[v]=rows;}
  let choice=models[selected.variant], localResult=local(d,measure,choice.config,choice.model);
  return {train:train.length,test:test.length,train_tiles:tiles.length,selection:'minimum grouped training CV coordinate RMSE; no 2025+ labels used for fit or selection',cv,selected:choice.config,models,historical_details:details,maryino:localResult,
    limitations:['Known historical tile, current POIs, surviving dated clubs only; undated competitors excluded. Historical test reused by earlier repository development.','Demand proxy uses stores and office floors, not measured population or memberships. Local sensitivity also includes existing housing-based prior.','Choice utility is Gaussian distance decay and scenario outside option; no walking graph, capacity, rent, vacancy or sales. Economic minimax maximizes normalized demand gain only, not profit.','Current local base competition comes from the existing repository snapshot. Extra reviewed gyms are excluded as occupied sites but are not added as demand competitors.']};
}
if(typeof module!=='undefined')module.exports={run,paths,capture,action};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),raw=Object.fromEntries(Object.entries(paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),report=run(raw);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/report.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({selected:report.selected,models:report.models,maryino:report.maryino},null,2));}
