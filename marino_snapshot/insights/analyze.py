"""Independent fixed-price, diversion, entry and commuting experiments; no fitted parameters."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,itertools,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from pyproj import Transformer
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent; O=Path(__file__).parent; OUT=O/'results'; OUT.mkdir(exist_ok=True)
a=np.load(R/'data/distances.npz'); D=a['candidates_m'].astype(float)/80; C=a['clubs_m'].astype(float)/80; w=pd.read_csv(R/'results/housing_match.csv').new_weight.to_numpy(); P=w.sum(); clubs=pd.read_csv(R/'data/clubs.csv'); net=clubs.network.to_numpy(); names, ni=np.unique(net,return_inverse=True); I=np.eye(len(names)); mobile=np.flatnonzero(clubs.in_district); grid=pd.read_csv(R/'data/candidates.csv'); b=pd.read_csv(R/'results/buildings.csv'); S=np.unique(b.query('in_1100').site.to_numpy(int)); hist=pd.read_csv(O/'data/ratings_openings.csv').set_index('id'); tariff=pd.read_csv(R/'data/tariffs.csv').set_index('network').reindex(names); known=tariff.monthly_rub.to_numpy(float); checks={}; eps=1e-8

def prob(cost,half=0):
 m=cost.min(0); t=cost-m
 if half: v=np.exp(-np.log(2)/half*t)
 else: v=(t<=eps).astype(float)
 return v/v.sum(0)
def shares(cost,n,half=0):return (prob(cost,half)@w)@I[n]/P*100

def add(cost,n,new,k,half=0):
 old=prob(cost,half); own=old[n==k].sum(0); mn=cost.min(0)
 if half:
  oldz=np.exp(-np.log(2)/half*(cost-mn)).sum(0); a=np.exp(np.clip(-np.log(2)/half*(new-mn),-700,700)); s=a/(oldz+a)
 else:
  cnt=(np.abs(cost-mn)<=eps).sum(0); s=(new<mn-eps)+ (np.abs(new-mn)<=eps)/(cnt+1)
 return (s@(w*(1-own)))*100/P, s@w*100/P

def scenario(model,fill=2500,tau=100,half=10,pool_bonus=0):
 p=np.where(np.isfinite(known),known,fill); pool=tariff.pool_confirmed.fillna(0).to_numpy(); effective=p-pool_bonus*pool
 if model=='nearest':return C,D,np.zeros(len(names)),0
 return C+effective[ni,None]/tau,D,effective/tau,half if model=='logit' else 0

def assess(cc,dd,off,half,j,others,sites=S):
 k=ni[j]; actual,gross=add(cc[others],ni[others],cc[j],k,half); g,gr=add(cc[others],ni[others],dd[sites]+off[k],k,half); idx=np.argmax(g); best=max(actual,g[idx]); return dict(actual_pp=float(actual),gross_pp=float(gross),cannibal_pp=float(gross-actual),best_pp=float(best),regret_pp=float(best-actual),efficiency=float(actual/best) if best>eps else np.nan,best_site=int(sites[idx]) if g[idx]>actual+eps else -1)
scenarios=[('nearest',2500,100,0,0)]+[(m,f,t,h,0) for m in ['generalized','logit'] for f in [2000,2500,3000] for t in [50,100,200] for h in ([0] if m=='generalized' else [5,10,20])]
rows=[]; re=[]; corr=[]; disp=[]; divers=[]; ddx=[]; effects=[]
for m,f,t,h,bns in scenarios:
 cc,dd,off,half=scenario(m,f,t,h,bns); base=shares(cc,ni,half); pclub=prob(cc,half)@w/P*100
 for k,n in enumerate(names):rows.append(dict(model=m,fill=f,tau=t,half=h,network=n,share_pp=base[k]))
 for j in mobile:
  r=assess(cc,dd,off,half,j,np.flatnonzero(np.arange(len(C))!=j)); re.append(dict(model=m,fill=f,tau=t,half=h,id=j,club=clubs.name[j],network=net[j],club_share_pp=pclub[j],**r))
 k=list(names).index('DDX'); g,gr=add(cc,ni,dd[S]+off[k],k,half); i=g.argmax();ddx.append(dict(model=m,fill=f,tau=t,half=h,site=S[i],gain_pp=g[i],gross_pp=gr[i]))
 central=m=='nearest' or (f==2500 and t==100 and (m=='generalized' or h==10))
 if not central:continue
 for k,n in enumerate(names):
  keep=ni!=k; new=shares(cc[keep],ni[keep],half); gain=new-base; assert np.isclose(gain.sum(),0); assert np.isclose(gain[k],-base[k]);
  for l,name in enumerate(names):
   if k!=l:divers.append(dict(model=m,removed=n,receiver=name,base_share_pp=base[k],gain_pp=gain[l],diversion_pct=100*gain[l]/base[k] if base[k]>eps else np.nan))
 for j in [0,1,2,5]:
  keep=np.arange(len(C))!=j; p=shares(cc[keep],ni[keep],half); delta=p-base
  for l,name in enumerate(names):disp.append(dict(model=m,removed_club=clubs.name[j],receiver=name,delta_pp=delta[l]))
 # sequential case holding all others current
 for j,ex in [(2,[0,1,2]),(0,[0,1]),(1,[1])]:
  keep=np.array([i for i in range(len(C)) if i not in ex]); before=shares(cc[keep],ni[keep],half); after=shares(cc[np.r_[keep,j]],ni[np.r_[keep,j]],half)
  assert abs((after-before).sum())<1e-7
  for k,n in enumerate(names):effects.append(dict(model=m,entry=clubs.name[j],network=n,before_pp=before[k],after_pp=after[k],delta_pp=after[k]-before[k]))
 for q in ['efficiency','regret_pp','actual_pp','club_share_pp']:
  r=pd.DataFrame(re).query('model==@m and fill==@f and tau==@t and half==@h').set_index('id');
  for y in ['rating','ratings_count','year_proxy']:
   sp=spearmanr(r[q],hist.loc[r.index,y]);corr.append(dict(model=m,x=q,y=y,rho=sp.statistic,p_asymptotic=sp.pvalue,n=len(r)))
pd.DataFrame(rows).to_csv(OUT/'shares.csv',index=False);pd.DataFrame(re).to_csv(OUT/'relocation_prices.csv',index=False);pd.DataFrame(corr).to_csv(OUT/'correlations.csv',index=False);pd.DataFrame(divers).to_csv(OUT/'diversion_networks.csv',index=False);pd.DataFrame(disp).to_csv(OUT/'diversion_clubs.csv',index=False);pd.DataFrame(ddx).to_csv(OUT/'ddx_entry_sensitivity.csv',index=False);pd.DataFrame(effects).to_csv(OUT/'entry_effects.csv',index=False)
# Conditional historical scenarios. Year proxies not exact dates; uncertain same-year rivals are included/excluded.
hr=[]
for j in mobile:
 year=hist.loc[j,'year_proxy']; earlier=[i for i in mobile if hist.loc[i,'year_proxy']<year]; same=[i for i in mobile if i!=j and hist.loc[i,'year_proxy']==year]; external=[i for i in range(len(C)) if i not in mobile]
 for bg in ['external_fixed','local_only']:
  for bits in itertools.product([False,True],repeat=len(same)):
   included=[i for i,x in zip(same,bits) if x]; others=np.array(earlier+included+(external if bg=='external_fixed' else []),int)
   if not len(others):continue
   for m in ['nearest','generalized','logit']:
    cc,dd,off,half=scenario(m); r=assess(cc,dd,off,half,j,others); today=assess(cc,dd,off,half,j,np.flatnonzero(np.arange(len(C))!=j)); hr.append(dict(model=m,id=j,club=clubs.name[j],year_proxy=year,background=bg,other_ids=';'.join(map(str,others)),today_regret_pp=today['regret_pp'],today_efficiency=today['efficiency'],**r))
pd.DataFrame(hr).to_csv(OUT/'history_all_clubs.csv',index=False)
# Quality compensation: one observed amenity proxy, NOT estimated. Stress for confirmed pools only.
qual=[]
for bonus in [0,500,1000,1342,1500,1842,2000]:
 for m in ['generalized','logit']:
  cc,dd,off,h=scenario(m,pool_bonus=bonus); base=shares(cc,ni,h); k=list(names).index('DDX'); g,_=add(cc,ni,dd[S]+off[k],k,h); z=g.argmax();
  for n in ['Magneto','DDX','Spirit']:
   l=list(names).index(n);j=np.flatnonzero((ni==l)&clubs.in_district.to_numpy())[0];r=assess(cc,dd,off,h,j,np.flatnonzero(np.arange(len(C))!=j));qual.append(dict(model=m,pool_bonus_rub=bonus,network=n,share_pp=base[l],regret_pp=r['regret_pp'],ddx_best_site=S[z],ddx_gain_pp=g[z]))
pd.DataFrame(qual).to_csv(OUT/'pool_quality_sensitivity.csv',index=False)
# Metro experiment: closest of the 14 archived entrances (not inferred actual origin), split fixed populations.
m=np.load(O/'data/metro_distances.npz'); ent=m['home'].argmin(0); hmin=m['home'][ent,np.arange(len(w))]; cm=m['club'][ent,:].T; dm=m['grid'][ent,:].T; detC=C+cm-hmin;detD=D+dm-hmin;assert detC.min()>-1e-5 and detD.min()>-1e-5;detC=np.maximum(detC,0);detD=np.maximum(detD,0)
metro=[];metr=[]
for model in ['nearest','generalized','logit']:
 cc,dd,off,h=scenario(model); cc2=detC+off[ni,None]; dd2=detD; k=list(names).index('DDX');g,_=add(cc,ni,dd[S]+off[k],k,h);g2,_=add(cc2,ni,dd2[S]+off[k],k,h)
 for frac in [0,.25,.5,.75,1]:
  v=(1-frac)*g+frac*g2;z=v.argmax();base=(1-frac)*shares(cc,ni,h)+frac*shares(cc2,ni,h);metro.append(dict(model=model,commute_fraction=frac,ddx_site=S[z],ddx_gain_pp=v[z],ddx_share_pp=base[k],opt_lat=grid.lat[S[z]],opt_lon=grid.lon[S[z]]))
  for j in mobile:
   keep=np.flatnonzero(np.arange(len(C))!=j);k0=ni[j];a1,gross1=add(cc[keep],ni[keep],cc[j],k0,h);a2,gross2=add(cc2[keep],ni[keep],cc2[j],k0,h);p1,_=add(cc[keep],ni[keep],dd[S]+off[k0],k0,h);p2,_=add(cc2[keep],ni[keep],dd2[S]+off[k0],k0,h);actual=(1-frac)*a1+frac*a2;gross=(1-frac)*gross1+frac*gross2;opt=max(actual,np.max((1-frac)*p1+frac*p2));metr.append(dict(model=model,commute_fraction=frac,id=j,club=clubs.name[j],actual_pp=actual,gross_pp=gross,regret_pp=opt-actual,efficiency=actual/opt if opt>eps else np.nan))
pd.DataFrame(metro).to_csv(OUT/'metro_entry.csv',index=False);pd.DataFrame(metr).to_csv(OUT/'metro_relocation.csv',index=False)
# Observed vs potential locations, raw infrastructure proxies. No supervised learning.
poi=pd.read_csv(O/'data/poi.csv'); tr=Transformer.from_crs(4326,32637,always_xy=True)
def xy(df):return np.column_stack(tr.transform(df.lon.to_numpy(),df.lat.to_numpy()))
loc=pd.concat([clubs.iloc[mobile][['name','lat','lon']].assign(kind='actual',site=-1,id=mobile),grid.iloc[S][['lat','lon']].assign(name='candidate',kind='potential',site=S,id=-1)],ignore_index=True);lc=xy(loc);shop=poi[poi.category.isin(['large_store','shopping_mall'])];sxy=xy(shop);mall=xy(poi[poi.category=='shopping_mall']); loc['stores_malls_300m']=(cdist(lc,sxy)<=300).sum(1);loc['mall_nearest_m']=cdist(lc,mall).min(1);loc['metro_walk_min']=np.r_[m['club'][:,mobile].min(0),m['grid'][:,S].min(0)];loc['residents_10min']=np.r_[(C[mobile]<=10)@w,(D[S]<=10)@w];loc['other_clubs_10min_straight_proxy']=(cdist(lc,xy(clubs))<=800).sum(1)-(loc.kind=='actual').astype(int);loc.to_csv(OUT/'site_features.csv',index=False)
checks['population']=bool(np.isclose(P,273463));checks['all_nonnegative_regret']=bool(pd.DataFrame(re).regret_pp.min()>=-1e-8);checks['nearest_matches_final_loo']=bool(np.allclose(pd.DataFrame(re).query('model=="nearest"').set_index('id').loc[hist.index,'efficiency'],hist.efficiency));checks['diversion_conservation']=bool(np.allclose(pd.DataFrame(divers).dropna(subset=['diversion_pct']).groupby(['model','removed']).diversion_pct.sum(),100));checks['metro_zero_matches']=bool(np.allclose(pd.DataFrame(metr).query('commute_fraction==0 and model=="nearest"').efficiency,hist.efficiency));assert all(checks.values()),checks
(OUT/'checks.json').write_text(json.dumps(checks,indent=2)); print(checks)
# Correlation robustness and exact permutation (n=9); undefined ratios excluded, no parameter selection.
from scipy.stats import rankdata
ret=pd.DataFrame(re); cr=[]; ex=[]
y=rankdata(hist.rating);y=y-y.mean();per=np.array(list(itertools.permutations(y)))
for key,g in ret.groupby(['model','fill','tau','half']):
 g=g.set_index('id').loc[hist.index]; valid=g.efficiency.notna();rho=spearmanr(g.loc[valid,'efficiency'],hist.loc[valid,'rating']).statistic if valid.sum()>2 else np.nan
 cr.append(dict(zip(['model','fill','tau','half'],key),rho=rho,n=int(valid.sum()),magneto_eff=g.loc[5,'efficiency'],spirit_eff=g.loc[2,'efficiency'],mall_eff=g.loc[1,'efficiency']))
 model,f,t,h=key
 if f==2500 and t==100 and h==(10 if model=='logit' else 0):
  x=rankdata(g.efficiency);x=x-x.mean();den=np.linalg.norm(x)*np.linalg.norm(y);pv=np.mean(np.abs(per@x/den)>=abs(rho)-1e-12);loo=[spearmanr(g.efficiency.drop(j),hist.rating.drop(j)).statistic for j in g.index];ex.append(dict(model=model,rho=rho,p_exact=pv,loo_min=min(loo),loo_max=max(loo)))
pd.DataFrame(cr).to_csv(OUT/'rating_sensitivity.csv',index=False);pd.DataFrame(ex).to_csv(OUT/'rating_permutation.csv',index=False)
# External demand stress: mass is chosen by analyst, not inferred population. Same competitors throughout.
buf=np.load(O/'data/buffer_distances.npz'); bh=pd.read_csv(O/'data/buffer_buildings.csv'); assert np.array_equal(buf['site'],S); extrows=[]; wi=w.copy(); p0=P
for radius in [500,1000,1500]:
 incl=bh.boundary_distance_m.to_numpy()<=radius; bw=bh.footprint_m2.to_numpy()[incl];bw=bw/bw.sum()
 for fraction in [0,.10,.25,.50]:
  w=np.r_[wi,bw*p0*fraction];P=w.sum()
  for model in ['nearest','generalized','logit']:
   cc,dd,off,h=scenario(model); c2=np.c_[cc,buf['club'][:,incl]+off[ni,None]]; d2=np.c_[dd[S],buf['grid'][:,incl]];k=list(names).index('DDX');g,_=add(c2,ni,d2+off[k],k,h);z=g.argmax()
   for j in [0,1,2]:
    keep=np.flatnonzero(np.arange(len(C))!=j);r=assess(c2,d2,off,h,j,keep,sites=np.arange(len(S)));extrows.append(dict(model=model,radius_m=radius,external_mass_ratio=fraction,external_buildings=int(incl.sum()),id=j,club=clubs.name[j],ddx_best_site=int(S[z]),ddx_gain_pp=g[z],**r))
w=wi;P=p0; ex=pd.DataFrame(extrows);ex.loc[ex.best_site>=0,'best_site']=S[ex.loc[ex.best_site>=0,'best_site'].to_numpy(int)];ex.to_csv(OUT/'boundary_sensitivity.csv',index=False)
assert np.allclose(ex.query('external_mass_ratio==0 and radius_m==500 and model=="nearest"').efficiency, hist.loc[[0,1,2],'efficiency']);checks['boundary_zero_matches']=True;(OUT/'checks.json').write_text(json.dumps(checks,indent=2))
# Advertised lower bounds are not comparable transaction prices: vary two key prices directly.
price_rows=[]; kp=list(names).index('Magneto');ka=list(names).index('DDX');ks=list(names).index('Spirit')
for pdx,pmag in itertools.product([1900,2300,2400,2800],[3742,4500,5000]):
 p=np.where(np.isfinite(known),known,2500).copy();p[ka]=pdx;p[kp]=pmag;off=p/100;cc=C+off[ni,None]
 for model,h in [('generalized',0),('logit',10)]:
  base=shares(cc,ni,h);g,_=add(cc,ni,D[S]+off[ka],ka,h);z=g.argmax();mr=assess(cc,D,off,h,5,np.flatnonzero(np.arange(len(C))!=5));sr=assess(cc,D,off,h,2,np.flatnonzero(np.arange(len(C))!=2));lr=assess(cc,D,off,h,1,np.flatnonzero(np.arange(len(C))!=1));price_rows.append(dict(model=model,ddx_price=pdx,magneto_price=pmag,ddx_share_pp=base[ka],magneto_share_pp=base[kp],spirit_share_pp=base[ks],magneto_eff=mr['efficiency'],spirit_eff=sr['efficiency'],mall_eff=lr['efficiency'],ddx_best_site=S[z],ddx_gain_pp=g[z]))
pd.DataFrame(price_rows).to_csv(OUT/'advertised_price_sensitivity.csv',index=False)
