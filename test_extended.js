'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict'),dir='results/inverse_experiment/';
for(let name of ['multiscale_benchmark','relative_benchmark','commercial_regime_benchmark','local_gate_benchmark']){
 const r=JSON.parse(fs.readFileSync(dir+name+'.json','utf8'));assert.equal(r.train,28);assert.equal(r.test,17);
 assert.equal(r.selected.n,13);assert(r.validation.some(x=>x.name===r.selected.name&&x.rule===r.selected.rule));
 for(let row of r.rows){assert(row.train_max_date<row.date);assert(Number.isFinite(row.logloss));assert(row.top3_m<=row.top1_m+1e-8);assert.equal(row.predictions.length,3);}
 for(let x of r.results)assert.equal(x.n,17);
}
const g=JSON.parse(fs.readFileSync(dir+'local_gate_benchmark.json','utf8')),a=g.rows.filter(x=>x.model==='fixed_gate'&&x.rule==='mode'),b=g.rows.filter(x=>x.model==='local_gate'&&x.rule==='mode');
assert.equal(a.filter(x=>x.hit3).length,10);assert.equal(b.filter(x=>x.hit3).length,11);
const map=new Map(a.map(x=>[x.name,x]));assert(b.every(x=>map.has(x.name)&&x.gate>0&&x.gate<1));
assert.equal(b.filter(x=>x.hit3&&!map.get(x.name).hit3).length,1);assert.equal(b.filter(x=>!x.hit3&&map.get(x.name).hit3).length,0);
const r=JSON.parse(fs.readFileSync(dir+'local_gate_maryino.json','utf8'));assert.equal(r.training_openings,45);assert.equal(r.eligible,1005);
for(let x of Object.values(r.results)){assert(Math.abs(x.posterior_total-1)<1e-10);assert(x.local_gate>0&&x.local_gate<1);for(let a of Object.values(x.recommendations))for(let q of a)assert(Math.abs(q.gross_pp-q.net_gain_pp-q.own_transfer_pp)<1e-8);}
console.log('Extended chronological, paired hit, and network accounting assertions passed.');
