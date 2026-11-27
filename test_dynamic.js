'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict');
const path='results/inverse_experiment/';
for(let name of ['dynamic_benchmark','expert_benchmark']){
 const r=JSON.parse(fs.readFileSync(path+name+'.json','utf8'));
 assert.equal(r.train,28);assert.equal(r.test,17);
 assert.equal(r.distribution.train.near_mall_150,21);assert.equal(r.distribution.test.near_mall_150,7);
 for(let row of r.rows){assert(row.train_max_date<row.date);assert(Number.isFinite(row.logloss));assert(row.top3_m<=row.top1_m+1e-8);}
 for(let model of Object.values(r.models))for(let m of Object.values(model)){assert(m.gate>0&&m.gate<1);assert(m.ess<=m.n+1e-8);assert(m.b.every(Number.isFinite));}
 for(let s of Object.values(r.summary)){assert.equal(s.all.n,17);assert.equal(s.year2025.n+s.year2026.n,17);}
}
const r=JSON.parse(fs.readFileSync(path+'dynamic_maryino.json','utf8'));
assert.equal(r.training_openings,45);assert.equal(r.eligible,1005);
for(let model of Object.values(r.results)){assert(Math.abs(model.posterior_total-1)<1e-10);assert.equal(model.loo.length,4);for(let a of Object.values(model.recommendations))for(let q of a){assert(q.gross_pp>=q.net_gain_pp&&q.net_gain_pp>=0);assert(Math.abs(q.gross_pp-q.net_gain_pp-q.own_transfer_pp)<1e-8);}}
console.log('Dynamic split, chronology, probability and network accounting checks passed.');
