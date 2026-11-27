'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict'),dir='results/inverse_experiment/',owner=require('./owner_experiment'),lag=require('./owner_lag_experiment');
for(let name of ['owner_benchmark','owner_lag_benchmark']){
 const r=JSON.parse(fs.readFileSync(dir+name+'.json','utf8'));assert.equal(r.train,28);assert.equal(r.test,17);assert.equal(r.assets,25);assert.equal(r.strict_assets,19);assert.equal(r.selected.n,13);
 for(let row of r.rows){assert(row.train_max_date<row.date);assert(Number.isFinite(row.logloss));assert(row.top3_m<=row.top1_m+1e-8);assert.equal(row.predictions.length,3);}
 const a=r.rows.filter(x=>x.model==='owner_area'&&x.rule==='mass');assert.equal(a.filter(x=>x.hit1).length,8);assert.equal(a.filter(x=>x.hit3).length,10);assert.equal(a.filter(x=>x.owner_near&&x.hit1).length,4);
}
const assets=[{p:[0,0],confidence:'external_coordinate',portfolio_known_by:'2014-12-15',area:{known_by:'2020-05-06',planned_gla_m2:10000}}],cfg={owner:true,area:true,broad:false};
assert.deepEqual(owner.ownerFeatures([0,0],'2014-01-01',assets,cfg),[0,0,0]);
assert.deepEqual(owner.ownerFeatures([0,0],'2020-01-01',assets,cfg),[1,0,0]);
assert.equal(owner.ownerFeatures([0,0],'2021-10-19',assets,cfg)[1],1);
assert.equal(lag.ownerFeatures([0,0],'2021-10-19',assets,cfg)[1],0);
assert.equal(lag.ownerFeatures([0,0],'2020-07-01',assets,cfg)[2],0);
const m=JSON.parse(fs.readFileSync(dir+'owner_maryino.json','utf8'));assert.equal(m.training_openings,45);assert.equal(m.eligible,1005);for(let v of Object.values(m.results))assert(Math.abs(v.posterior_total-1)<1e-10);
const l=JSON.parse(fs.readFileSync(dir+'listing_screen.json','utf8'));let a=l.candidates.find(x=>x.id==='icx-24489');assert.equal(a.area_m2,1300);assert(a.unoccupied_radius_150);assert(a.distance_to_robust_m<300);assert.equal(a.scenarios.length,18);
for(let s of a.scenarios)assert(Math.abs(s.gross_pp-s.net_gain_pp-s.own_transfer_pp)<1e-8);
assert(l.screened_out.some(x=>x.status==='existing_ddx_building_excluded'));
console.log('Portfolio date masks, chronological benchmarks and listing accounting assertions passed.');
