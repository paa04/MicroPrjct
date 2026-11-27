/* Local robust decision analysis using trained historical site model. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180), MY=111320;
const xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY],ll=p=>({lon:p[0]/MX+37.74,lat:p[1]/MY+55.654});
const d2=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
const sum=a=>a.reduce((s,v)=>s+v,0);
function run(raw,model,lib){
 const d=Object.fromEntries(Object.entries(raw).map(([k,v])=>[k,lib.csv(v)])), measure=lib.featureFactory(d),cells=d.district.map(r=>[+r.x_m,+r.y_m]),f=cells.map(measure);
 const market=d.market.map(r=>({...r,p:xy(r)})),all=d.reviewed.filter(r=>r.eligible==='yes').map(r=>({...r,p:xy(r)})),uncertain=new Set(d.prices.filter(r=>r.include==='uncertain').map(r=>r.club_id)),wide=[...market,...all.filter(r=>uncertain.has(r.club_id))];
 const extra=all.filter(r=>!market.some(t=>t.network===r.network&&d2(t.p,r.p)<300**2)),occupied=[...market,...extra];
 const venue=d.venues.filter(r=>+r.footprint_m2>=1000).map(r=>({...r,p:xy(r)}));
 const indices=d.district.map((r,i)=>r.allowed==='1'&&occupied.every(c=>d2(c.p,cells[i])>=150**2)?i:-1).filter(i=>i>=0),sites=indices.map(i=>cells[i]);
 const nearest=sites.map(p=>venue.reduce((a,v)=>d2(v.p,p)<d2(a.p,p)?v:a,venue[0]));
 const close=sites.map((p,i)=>d2(p,nearest[i].p)<=150**2);
 if(!close.some(Boolean))throw Error('No venue-adjacent sites');
 const norm=a=>{let s=sum(a);return a.map(v=>v/s);};
 const priors={housing: norm(f.map(a=>Math.exp(.8*a[4]+.15*a[5]+.05*a[6]))),uniform:norm(f.map(()=>1)),legacy:norm(d.district.map(r=>.6*(+r.housing)+.15*(+r.metro)+.15*(+r.retail)+.1*(+r.office)))};
 const sc=[];
 for(let tau of [400,800,1200]){
   let K=new Float64Array(sites.length*cells.length);for(let i=0;i<sites.length;i++)for(let j=0;j<cells.length;j++)K[i*cells.length+j]=Math.exp(-d2(sites[i],cells[j])/(2*tau*tau));
   for(let [m,clubs] of [['base',market],['wide',wide]])for(let [prior,rho] of Object.entries(priors)){let z=lib.capture(cells,sites,clubs,rho,tau,K);sc.push({m,prior,tau,...z,best:Math.max(...z.gain.filter((_,i)=>close[i]))});}
 }
 const score=indices.map(i=>model.b.reduce((a,b,j)=>a+b*(f[i][j]-model.mu[j])/model.sd[j],0));
 const action=lib.action(sites,score),peak=Math.max(...action.mass.filter((_,i)=>close[i]));
 const metrics=sites.map((p,i)=>({i,regret:Math.max(...sc.map(z=>(z.best-z.gain[i])/Math.max(z.best,1e-12))),minGain:Math.min(...sc.map(z=>z.gain[i])),p:action.mass[i],close:close[i]}));
 const feasible=metrics.filter(z=>z.close),order=(a,b)=>a.regret-b.regret;
 const robust=[...feasible].sort(order)[0],plausible=[...feasible].sort((a,b)=>b.p-a.p)[0],compromise=feasible.filter(z=>z.p>=.5*peak).sort(order)[0],maximin=[...feasible].sort((a,b)=>b.minGain-a.minGain)[0];
 const describe=z=>({...ll(sites[z.i]),nearest_venue:nearest[z.i].name||nearest[z.i].osm_id,venue_osm_id:nearest[z.i].osm_id,venue_footprint_m2:+nearest[z.i].footprint_m2,venue_distance_m:Math.sqrt(d2(sites[z.i],nearest[z.i].p)),historical_model_mass_300:z.p,worst_relative_regret:z.regret,min_gain_pp:100*z.minGain,max_gain_pp:100*Math.max(...sc.map(s=>s.gain[z.i])),scenarios:sc.map(s=>({market:s.m,prior:s.prior,tau:s.tau,net_gain_pp:100*s.gain[z.i],gross_pp:100*s.gross[z.i],own_transfer_pp:100*(s.gross[z.i]-s.gain[z.i])}))});
 let pareto=feasible.filter(a=>!feasible.some(b=>b.p>=a.p&&b.regret<=a.regret&&(b.p>a.p||b.regret<a.regret))).sort(order);
 return {eligible:sites.length,venue_adjacent:feasible.length,base_competitors:market.length,wide_competitors:wide.length,scenarios:sc.length,robust:describe(robust),maximin:describe(maximin),historical:describe(plausible),compromise:describe(compromise),pareto:pareto.map(describe),limitations:'Commercial footprint >=1000 m2 within 150 m is not verified vacant floor area. 50% of peak historical mass is illustrative, not estimated. Euclidean distances, potential market within Maryino, scenario elasticities and outside option, not profit.'};
}
if(typeof module!=='undefined')module.exports={run};
if(typeof require!=='undefined'&&require.main===module){let fs=require('fs'),lib=require('./radius_experiment'),paths={...lib.paths,prices:'data/economy/local_prices.csv'},raw=Object.fromEntries(Object.entries(paths).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),model=JSON.parse(fs.readFileSync('results/inverse_experiment/radius_benchmark.json','utf8')).models.site.model,r=run(raw,model,lib);fs.mkdirSync('results/inverse_experiment',{recursive:true});fs.writeFileSync('results/inverse_experiment/robust_local.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify({robust:r.robust,compromise:r.compromise,historical:r.historical},null,2));}
