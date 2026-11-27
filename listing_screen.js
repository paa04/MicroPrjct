/* Evaluate actual rental leads against the unrestricted economic optimum.
   Unconfirmed listings are leads, not verified available space. */
'use strict';
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320,xy=r=>[(+r.lon-37.74)*MX,(+r.lat-55.654)*MY];
const sq=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2,sum=a=>a.reduce((s,v)=>s+v,0);
const sources={leads:'data/real_estate/maryino_listing_leads.csv',prices:'data/economy/local_prices.csv',robust:'results/inverse_experiment/robust_local.json'};
function run(raw,lib){
 const keys=[...Object.keys(lib.paths), 'leads','prices'],d=Object.fromEntries(keys.map(k=>[k,lib.csv(raw[k])])),old=JSON.parse(raw.robust),measure=lib.featureFactory(d);
 const cells=d.district.map(r=>[+r.x_m,+r.y_m]),f=cells.map(measure),market=d.market.map(r=>({...r,p:xy(r)})),all=d.reviewed.filter(r=>r.eligible==='yes').map(r=>({...r,p:xy(r)})),uncertain=new Set(d.prices.filter(r=>r.include==='uncertain').map(r=>r.club_id)),wide=[...market,...all.filter(r=>uncertain.has(r.club_id))],extra=all.filter(r=>!market.some(t=>t.network===r.network&&sq(t.p,r.p)<300**2));
 const free=p=>[...market,...extra].every(c=>sq(p,c.p)>=150**2),norm=a=>{let s=sum(a);return a.map(v=>v/s);};
 const priors={housing:norm(f.map(a=>Math.exp(.8*a[4]+.15*a[5]+.05*a[6]))),uniform:norm(f.map(()=>1)),legacy:norm(d.district.map(r=>.6*(+r.housing)+.15*(+r.metro)+.15*(+r.retail)+.1*(+r.office)))};
 const leads=d.leads.filter(r=>r.status==='advertised_lease_unconfirmed'&&r.lon&&r.lat&&+r.area_m2>=1000).map(r=>({...r,p:xy(r)}));
 const items=[...leads,...['robust','compromise','historical'].map(k=>({id:k,address:k,lon:old[k].lon,lat:old[k].lat,p:xy(old[k]),status:'model_zone_not_listing'}))];
 const nearest=items.map(r=>Math.min(...cells.map(p=>Math.sqrt(sq(p,r.p)))));
 const sc=[];for(let tau of [400,800,1200])for(let [m,clubs] of [['base',market],['wide',wide]])for(let [prior,rho] of Object.entries(priors)){let z=lib.capture(cells,items.map(r=>r.p),clubs,rho,tau);sc.push({m,prior,tau,...z});}
 let report=items.map((r,i)=>({id:r.id,address:r.address,lon:+r.lon,lat:+r.lat,status:r.status,area_m2:r.area_m2?+r.area_m2:null,monthly_rent_rub:r.monthly_rent_rub?+r.monthly_rent_rub:null,distance_to_robust_m:Math.sqrt(sq(r.p,xy(old.robust))),nearest_district_cell_m:nearest[i],unoccupied_radius_150:free(r.p),min_gain_pp:100*Math.min(...sc.map(s=>s.gain[i])),max_gain_pp:100*Math.max(...sc.map(s=>s.gain[i])),scenarios:sc.map(s=>({market:s.m,prior:s.prior,tau:s.tau,net_gain_pp:100*s.gain[i],gross_pp:100*s.gross[i],own_transfer_pp:100*(s.gross[i]-s.gain[i])}))}));
 return {candidates:report,screened_out:d.leads.filter(r=>!leads.some(l=>l.id===r.id)).map(r=>({id:r.id,address:r.address,status:r.status,area_m2:r.area_m2,note:r.note})),limitations:'One geocoded >=1000m2 advertised lead in this small nonexhaustive sample. Actual availability, layout, zoning, lease price, capacity and dates not verified. Exact-address centroid, not entrance. Euclidean 18 scenarios; normalized demand, not profit. Reference zones are not claimed available.'};
}
module.exports={run,sources};
if(require.main===module){let fs=require('fs'),lib=require('./ensemble_experiment'),raw=Object.fromEntries(Object.entries({...lib.paths,...sources}).map(([k,p])=>[k,fs.readFileSync(p,'utf8')])),r=run(raw,lib);fs.writeFileSync('results/inverse_experiment/listing_screen.json',JSON.stringify(r,null,2)+'\n');console.log(JSON.stringify(r,null,2));}
