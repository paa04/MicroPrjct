'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),lib=require('./radius_experiment');
const own=lib.capture([[0,0]],[[0,0]],[{p:[0,0],network:'DDX'}],[1],400);
const rival=lib.capture([[0,0]],[[0,0]],[{p:[0,0],network:'Spirit'}],[1],400);
assert(own.gain[0]<own.gross[0]);assert(Math.abs(rival.gain[0]-rival.gross[0])<1e-12);
const a=lib.action([[0,0],[100,0],[1000,0]],[0,0,0]);assert.equal(a.mse,1);assert(Math.abs(a.prob.reduce((s,v)=>s+v,0)-1)<1e-12);
const r=JSON.parse(fs.readFileSync('results/inverse_experiment/robust_local.json','utf8')),b=JSON.parse(fs.readFileSync('results/inverse_experiment/radius_benchmark.json','utf8'));
assert.equal(r.scenarios,18);assert.equal(r.base_competitors,8);assert.equal(r.wide_competitors,13);assert.equal(r.venue_adjacent,144);assert.equal(b.train,28);assert.equal(b.test,17);
for(const k of ['robust','maximin','historical','compromise']){let q=r[k];assert.equal(q.scenarios.length,18);assert(q.venue_distance_m<=150);
for(const s of q.scenarios){assert(s.net_gain_pp>=0&&s.gross_pp>=s.net_gain_pp);assert(Math.abs(s.gross_pp-s.net_gain_pp-s.own_transfer_pp)<1e-10);}
assert(Math.abs(Math.min(...q.scenarios.map(s=>s.net_gain_pp))-q.min_gain_pp)<1e-10);}
console.log('All inverse experiment assertions passed.');
