/* Historical OSM retrieval. Node 18+. No credentials.
   Dry run: node fetch_historical_osm.js
   Download: node fetch_historical_osm.js --download --limit 3
   Files are raw snapshots, not yet converted to training features. */
'use strict';
const fs=require('node:fs'),path=require('node:path');
const {csv}=require('./ensemble_experiment');
const MX=111320*Math.cos(55.654*Math.PI/180),MY=111320,T=3600,BUFFER=3000;
const filters={
 housing:'building in (apartments,residential,dormitory)',
 office:'building=office',
 commercial:'building in (retail,commercial,supermarket) or shop=mall',
 stores:'shop=supermarket',
 metro:'railway=subway_entrance',
 bus:'highway=bus_stop'
};
function plan(catalog,split){
 const allowed=new Set(split.filter(r=>['train','time_test'].includes(r.split)).map(r=>r.id)),groups=new Map();
 for(let c of catalog){if(c.network!=='DDX'||!allowed.has(c.id)||!/^\d{4}-\d{2}-\d{2}$/.test(c.opened))continue;
  const x=Math.floor((+c.lon-37.74)*MX/T),y=Math.floor((+c.lat-55.654)*MY/T);
  const cutoff=new Date(Date.parse(c.opened)-180*86400000),year=cutoff.getUTCFullYear(),month=1+3*Math.floor(cutoff.getUTCMonth()/3),date=year+'-'+String(month).padStart(2,'0')+'-01';
  if(date>=c.opened)throw Error('Future snapshot');const key=[x,y,date].join('_');
  if(!groups.has(key))groups.set(key,{key,date,tile:[x,y],bbox:[37.74+(x*T-BUFFER)/MX,55.654+(y*T-BUFFER)/MY,37.74+((x+1)*T+BUFFER)/MX,55.654+((y+1)*T+BUFFER)/MY],events:[]});
  groups.get(key).events.push({id:c.id,opened:c.opened});
 }
 return [...groups.values()].sort((a,b)=>a.date.localeCompare(b.date)||a.key.localeCompare(b.key));
}
async function download(job,layer,dir){
 const filename=path.join(dir,job.key+'_'+layer+'.geojson');
 if(fs.existsSync(filename)){let old=JSON.parse(fs.readFileSync(filename,'utf8'));if(old.type==='FeatureCollection')return;}
 const body=new URLSearchParams({bboxes:job.bbox.join(','),time:job.date,filter:filters[layer],properties:'tags',clipGeometry:'false'});
 const res=await fetch('https://api.ohsome.org/v1/elements/geometry',{method:'POST',body,signal:AbortSignal.timeout(180000)});
 if(!res.ok)throw Error('HTTP '+res.status+' '+(await res.text()).slice(0,200));
 const data=await res.json();if(data.type!=='FeatureCollection'||!Array.isArray(data.features))throw Error('Invalid GeoJSON response');
 fs.writeFileSync(filename+'.tmp',JSON.stringify(data));fs.renameSync(filename+'.tmp',filename);
 console.log(job.key,layer,data.features.length);
}
async function main(){let jobs=plan(csv(fs.readFileSync('data/economy/catalog.csv','utf8')),csv(fs.readFileSync('results/economy/training_split.csv','utf8'))),dir='data/historical_osm';
 fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(path.join(dir,'manifest.json'),JSON.stringify({source:'https://api.ohsome.org/v1',license:'ODbL; © OpenStreetMap contributors',lag_days:180,filters,jobs},null,2));
 console.log(jobs.length+' tile/date snapshots, '+jobs.length*Object.keys(filters).length+' requests; dry run unless --download');
 if(!process.argv.includes('--download'))return;
 const ix=process.argv.indexOf('--limit'),limit=ix>=0?Number(process.argv[ix+1]):jobs.length;if(!Number.isInteger(limit)||limit<1)throw Error('Invalid limit');
 for(let job of jobs.slice(0,limit))for(let layer of Object.keys(filters)){await download(job,layer,dir);await new Promise(r=>setTimeout(r,1500));}
}
module.exports={plan,filters};
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1;});
