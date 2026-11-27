"""Marino: deterministic location game, transparent OSM building filter, fixed-price sensitivity.
Offline reproduction from the archived OSM snapshot. No fitted models.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point, mapping
from shapely.ops import transform
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent; DATA=R/'data'; OUT=R/'results'; FIG=R/'figures'; OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
a=np.load(DATA/'distances.npz'); D=a['candidates_m'].astype(float)/80; C=a['clubs_m'].astype(float)/80; w=a['weights'].astype(float); P=w.sum(); grid=pd.read_csv(DATA/'candidates.csv'); clubs=pd.read_csv(DATA/'clubs.csv'); names,net=np.unique(clubs.network,return_inverse=True); K=len(names); A=list(names).index('DDX'); B=list(names).index('Spirit'); mobile=np.flatnonzero(clubs.in_district); xy=grid[['x','y']].to_numpy(); checks={}; eps=1e-8

def housing_weights():
 import re
 def key(st,h):
  return (str(st).strip().lower().replace('ё','е'), re.sub(r'\s+','',str(h).lower()).translate(str.maketrans({'k':'к','c':'с','a':'а','b':'в'})))
 f=json.loads((DATA/'demand.geojson').read_text())['features']; f=pd.DataFrame([x['properties'] for x in f]); assert np.allclose(f.residents,w)
 reg=pd.read_csv(DATA/'housing_registry.csv',dtype={'house':str}); reg['key']=[key(x.street,x.house) for x in reg.itertuples()]; assert not reg.key.duplicated().any()
 f['key']=[key(x,y) for x,y in zip(f['addr:street'],f['addr:housenumber'])]; f['registry_apartments']=f.key.map(reg.set_index('key').apartments); f['matched']=f.registry_apartments.fillna(0)>0
 f['allocated_apartments']=f.registry_apartments*f.floor_proxy/f.groupby('key').floor_proxy.transform('sum'); matched=f.matched
 ratio=f.loc[matched,'allocated_apartments'].sum()/f.loc[matched,'floor_proxy'].sum(); f.loc[~matched,'allocated_apartments']=f.loc[~matched,'floor_proxy']*ratio
 f['new_weight']=P*f.allocated_apartments/f.allocated_apartments.sum(); f['old_weight']=w; f.to_csv(OUT/'housing_match.csv',index=False)
 (OUT/'housing_summary.json').write_text(json.dumps(dict(registry_rows=len(reg),registry_apartments=float(reg.apartments.sum()),matched_polygons=int(matched.sum()),matched_addresses=int(f.loc[matched,'key'].nunique()),matched_apartments=float(f.loc[matched,'allocated_apartments'].sum()),imputed_apartment_equivalents=float(f.loc[~matched,'allocated_apartments'].sum()),imputed_population=float(f.loc[~matched,'new_weight'].sum())),ensure_ascii=False,indent=2))
 return f.new_weight.to_numpy()
w_osm=w.copy(); w=housing_weights()

def check(key,val):
 checks[key]=bool(val); assert val,key

def alloc(c,n):
 t=np.abs(c-c.min(0))<=eps; return (t/t.sum(0))@w@np.eye(K)[n]

def gain(c,n,d):
 dm=c.min(0); t=np.abs(c-dm)<=eps; cnt=t.sum(0); own=(t/cnt).T@np.eye(K)[n]; p=(d<dm-eps)+(np.abs(d-dm)<=eps)/(cnt+1); return p@(w[:,None]*(1-own)),p@w

def buildings():
 boundary=shape(json.loads((DATA/'boundary.geojson').read_text())['features'][0]['geometry']); f=json.loads((DATA/'address_buildings.geojson').read_text())['features']; tr=Transformer.from_crs(4326,32637,always_xy=True).transform; tree=cKDTree(xy); cp=[Point(*tr(c.lon,c.lat)) for c in clubs.itertuples()]; rows=[]; features=[]; ninside=0
 for el in f:
  g=shape(el['geometry']); p=el['properties']; r=g.representative_point()
  if not boundary.covers(r):continue
  ninside+=1
  if p['building'] not in ['retail','commercial','sports_centre']:continue
  G=transform(tr,g); rp=G.representative_point(); dist,site=tree.query([rp.x,rp.y]); distclub=np.array([G.distance(c) for c in cp]); j=int(distclub.argmin()); occupied=str(p['osm_id'])=='46679150'; free=distclub[j]>30 and not occupied
  row=dict(osm_id=str(p['osm_id']),osm_url='https://www.openstreetmap.org/way/'+str(p['osm_id']),address=p['address'],name=p['name'],building=p['building'],footprint_m2=G.area,lat=r.y,lon=r.x,site=int(site),snap_m=dist,club_distance_m=distclub[j],nearest_club=clubs.name[j],near_existing=bool(distclub[j]<=30 or occupied),known_specialized=occupied,in_500=bool(G.area>=500 and dist<=200 and free),in_1100=bool(G.area>=1100 and dist<=200 and free),in_1500=bool(G.area>=1500 and dist<=200 and free),in_1100_all=bool(G.area>=1100 and dist<=200))
  rows.append(row); features.append({'type':'Feature','geometry':mapping(g),'properties':{'osm_id':row['osm_id'],'footprint_m2':G.area,'in_1100':row['in_1100']}})
 b=pd.DataFrame(rows);b.to_csv(OUT/'buildings.csv',index=False);(OUT/'commercial_buildings.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False));print('address polygons inside',ninside,'commercial',len(b),flush=True);return b,ninside

def best(s,v):return int(s[np.argmax(v[s])])

def main():
 check('dimensions',D.shape==(3894,365) and C.shape==(23,365) and len(mobile)==9 and np.isclose(P,273463));b,ninside=buildings(); sets={'grid':np.arange(len(D))}
 for label in ['500','1100','1500','1100_all']:
  sets[label]=np.unique(b.loc[b['in_'+label],'site'].to_numpy(int))
 check('nested',set(sets['1500'])<=set(sets['1100'])<=set(sets['500']));check('nonempty',all(len(s)>0 for s in sets.values()));pd.DataFrame([{'strategy':k,'sites':len(s),'buildings':len(b[b['in_'+k]]) if k!='grid' else np.nan} for k,s in sets.items()]).to_csv(OUT/'strategy_sets.csv',index=False)
 base=alloc(C,net); check('mass',np.isclose(base.sum(),P)); market=[]; ddx=[];opts=[]
 for m in ['narrow','wide','broad']:
  keep=clubs[m].to_numpy(bool); v=alloc(C[keep],net[keep]); g,gross=gain(C[keep],net[keep],D);ddx.append(g[:,A]*100/P); sh=100*v/P;market.append(dict(market=m,clubs=keep.sum(),networks=len(set(net[keep])),hhi=sh@sh,cr3=np.sort(sh)[-3:].sum(),ddx_share=sh[A]));
  for label,s in sets.items():
   for k in sorted(set(net[keep])):
    z=best(s,g[:,k]);opts.append(dict(market=m,strategy=label,network=names[k],site=z,lat=grid.lat[z],lon=grid.lon[z],gain_pp=g[z,k]*100/P,gross_pp=gross[z]*100/P,cannibal_pp=(gross[z]-g[z,k])*100/P))
 pd.DataFrame(market).to_csv(OUT/'market.csv',index=False);pd.DataFrame(opts).to_csv(OUT/'optima.csv',index=False);ddx=np.array(ddx); rr=[]
 for label,s in sets.items():
  z=best(s,ddx.min(0));rr.append(dict(strategy=label,site=z,lat=grid.lat[z],lon=grid.lon[z],narrow=ddx[0,z],wide=ddx[1,z],broad=ddx[2,z],worst=ddx[:,z].min()))
 pd.DataFrame(rr).to_csv(OUT/'robust.csv',index=False)
 overlap=[]
 for minute in [5,10,15,20]:
  count=np.sum(np.array([np.any(C[net==k]<=minute,axis=0) for k in range(K)]),axis=0);overlap.append(dict(minutes=minute,none=100*w[count==0].sum()/P,one=100*w[count==1].sum()/P,two_plus=100*w[count>=2].sum()/P))
 pd.DataFrame(overlap).to_csv(OUT/'overlap.csv',index=False)
 # Single-club relocation (one firm, current position is always an option).
 reloc=[]
 for j in mobile:
  keep=np.arange(len(C))!=j;k=net[j];g,_=gain(C[keep],net[keep],D);v=alloc(C[keep],net[keep])[k]+g[:,k]-base[k]
  for label,s in sets.items():
   z=best(s,v); r=max(0,v[z])*100/P;reloc.append(dict(strategy=label,club=clubs.name[j],network=names[k],gain_pp=r,site=z if r>eps else -1))
 re=pd.DataFrame(reloc);re.to_csv(OUT/'relocation.csv',index=False);re2=re.groupby(['strategy','network'],as_index=False).gain_pp.max();re2.to_csv(OUT/'network_regret.csv',index=False);summary=re2.groupby('strategy').gain_pp.agg(['mean','max']).reset_index();summary.to_csv(OUT/'regret_summary.csv',index=False);check('regret_monotone',summary.set_index('strategy').loc['1500','max']<=summary.set_index('strategy').loc['1100','max']<=summary.set_index('strategy').loc['grid','max']+eps)
 # Full finite game on each restricted set. Distinct premises is a scenario (not necessarily impossible to share a building).
 eqs=[]; game_summary=[]; payrows=[]
 for label in ['500','1100','1500','1100_all']:
  s=sets[label];n=len(s);u=np.empty((n,n));v=u.copy()
  for j,y in enumerate(s):
   after=alloc(np.vstack([C,D[y]]),np.r_[net,B]);gg,_=gain(np.vstack([C,D[y]]),np.r_[net,B],D[s]);u[:,j]=(after[A]+gg[:,A]-base[A])*100/P
  for i,x in enumerate(s):
   after=alloc(np.vstack([C,D[x]]),np.r_[net,A]);gg,_=gain(np.vstack([C,D[x]]),np.r_[net,A],D[s]);v[i,:]=(after[B]+gg[:,B]-base[B])*100/P
  for distinct in [False,True]:
   ok=np.ones_like(u,dtype=bool);np.fill_diagonal(ok,not distinct); ua=np.where(ok,u,-np.inf);vb=np.where(ok,v,-np.inf); eq=np.argwhere(ok&np.isclose(ua,ua.max(0),atol=1e-8,rtol=0)&np.isclose(vb,vb.max(1)[:,None],atol=1e-8,rtol=0));game_summary.append(dict(strategy=label,distinct=distinct,sites=n,equilibria=len(eq)))
   for i,j in eq:
    x,y=int(s[i]),int(s[j]);true=(alloc(np.vstack([C,D[x],D[y]]),np.r_[net,A,B])-base)*100/P;check(f'game_{label}_{distinct}_{i}_{j}',np.allclose([u[i,j],v[i,j]],true[[A,B]]));eqs.append(dict(strategy=label,distinct=distinct,ddx_site=x,spirit_site=y,ddx_gain_pp=u[i,j],spirit_gain_pp=v[i,j],distance_m=np.linalg.norm(xy[x]-xy[y])))
  if label=='1100':
   for i,j in itertools.product(range(n),repeat=2):payrows.append(dict(ddx_site=s[i],spirit_site=s[j],ddx_gain_pp=u[i,j],spirit_gain_pp=v[i,j]))
 pd.DataFrame(eqs).to_csv(OUT/'equilibria.csv',index=False);pd.DataFrame(game_summary).to_csv(OUT/'game_summary.csv',index=False);pd.DataFrame(payrows).to_csv(OUT/'game_payoffs.csv',index=False)
 # Verify the two unrestricted equilibria previously found, not a new exhaustive search.
 ix={int(z):i for i,z in enumerate(grid.grid_index)}; e,west=ix[3018],ix[1081];full=[]
 for x,y in [(e,west),(west,e)]:
  true=alloc(np.vstack([C,D[x],D[y]]),np.r_[net,A,B]); regret=[]
  for k,o,z in [(A,B,y),(B,A,x)]:
   c=np.vstack([C,D[z]]);n=np.r_[net,o];gg,_=gain(c,n,D);regret.append((alloc(c,n)[k]+gg[:,k].max()-true[k])*100/P)
  full.append(dict(ddx_site=x,spirit_site=y,ddx_gain_pp=(true[A]-base[A])*100/P,spirit_gain_pp=(true[B]-base[B])*100/P,max_deviation_pp=max(regret),verified=max(regret)<1e-7))
 pd.DataFrame(full).to_csv(OUT/'unrestricted_verified.csv',index=False)
 # Three demand assumptions on identical price vectors, same mass and margin; no fitting.
 tariffs=pd.read_csv(DATA/'tariffs.csv').set_index('network').reindex(names); known=tariffs.monthly_rub.to_numpy(float);theta=.05;margin=900;area=1300;fixed=300000; sens=[]; surfaces=[]; props=[]; property=pd.read_csv(DATA/'properties.csv');
 def emit(model,fill,tau,half,values):
  for label,s in sets.items():
   z=best(s,values); q=values[z];sens.append(dict(model=model,missing_price=fill,tau=tau,half=half,strategy=label,site=z,gain_pp=q,gain_members=q*P*theta/100,margin=q*P*theta/100*margin,rent_max=q*P*theta/100*margin/area,rent_max_after_fixed=(q*P*theta/100*margin-fixed)/area))
  if (model=='nearest' or (fill==2500 and tau==100 and (model=='generalized' or half==10))):
   surfaces.append(pd.DataFrame({'model':model,'site':np.arange(len(D)),'gain_pp':values}));
   for r in property.itertuples():
    g=values[int(r.site)]*P*theta/100*margin;props.append(dict(model=model,address=r.address,area=r.area,rent=r.rent,margin=g,after_rent=g-r.rent,after_other_fixed=g-r.rent-fixed,area_pass=r.area_pass,status=r.status))
 emit('nearest',0,0,0,ddx[2]);
 for fill,tau in itertools.product([2000,2500,3000],[50,100,200]):
  p=np.where(np.isfinite(known),known,fill);c=C+p[net,None]/tau; d=D+p[A]/tau;gg,_=gain(c,net,d);emit('generalized',fill,tau,0,gg[:,A]*100/P)
  for half in [5,10,20]:
   gamma=np.log(2)/half; exp=np.exp(-gamma*(C+p[net,None]/tau));den=exp.sum(0);own=exp[net==A].sum(0); x=np.exp(-gamma*(D+p[A]/tau));g=(x/(den+x)*(1-own/den))@w;emit('logit',fill,tau,half,100*g/P)
 pd.DataFrame(sens).to_csv(OUT/'demand_sensitivity.csv',index=False);pd.concat(surfaces).to_csv(OUT/'comparison_surfaces.csv',index=False);pd.DataFrame(props).to_csv(OUT/'property_economics.csv',index=False)
 # Compare the three demand models on exactly the same 35 site strategies.
 s=sets['1100']; p=np.where(np.isfinite(known),known,2500); tau=100.; gamma=np.log(2)/10; cg=C+p[net,None]/tau; da=D[s]+p[A]/tau; db=D[s]+p[B]/tau
 comparison=[]; ceq=[]
 for model in ['nearest','generalized','logit']:
  if model=='nearest':
   old=base; cc=C; ca=D[s]; cb=D[s]
  elif model=='generalized':
   old=alloc(cg,net);cc=cg;ca=da;cb=db
  else:
   ex=np.exp(-gamma*cg);den=ex.sum(0);oa=ex[net==A].sum(0);ob=ex[net==B].sum(0);old=(ex/den)@w@np.eye(K)[net];xa=np.exp(-gamma*da);xb=np.exp(-gamma*db)
  u=np.empty((len(s),len(s)));v=u.copy()
  for i in range(len(s)):
   if model=='logit':
    z=den+xa[i]+xb; u[i]=((oa+xa[i])/z@w-old[A])*100/P;v[i]=((ob+xb)/z@w-old[B])*100/P
   else:
    for j in range(len(s)):
     q=(alloc(np.vstack([cc,ca[i],cb[j]]),np.r_[net,A,B])-old)*100/P;u[i,j],v[i,j]=q[[A,B]]
  for distinct in [False,True]:
   ok=np.ones_like(u,dtype=bool);np.fill_diagonal(ok,not distinct); ua=np.where(ok,u,-np.inf);vb=np.where(ok,v,-np.inf);eq=np.argwhere(ok&np.isclose(ua,ua.max(0),atol=1e-8,rtol=0)&np.isclose(vb,vb.max(1)[:,None],atol=1e-8,rtol=0));comparison.append(dict(model=model,distinct=distinct,equilibria=len(eq)))
   for i,j in eq:ceq.append(dict(model=model,distinct=distinct,ddx_site=s[i],spirit_site=s[j],ddx_gain_pp=u[i,j],spirit_gain_pp=v[i,j],distance_m=np.linalg.norm(xy[s[i]]-xy[s[j]])))
 pd.DataFrame(comparison).to_csv(OUT/'comparison_game_summary.csv',index=False);pd.DataFrame(ceq).to_csv(OUT/'comparison_equilibria.csv',index=False)
 check('comparison_game_nearest',pd.DataFrame(comparison).query('model=="nearest"').equilibria.tolist()==pd.DataFrame(game_summary).query('strategy=="1100"').equilibria.tolist())
 # Direct numerical checks for ties and add-vs-reallocate.
 rng=np.random.default_rng(42); gg,gross=gain(C,net,D)
 for i in rng.choice(len(D),20,replace=False):
  for k in [A,B]:
   direct=alloc(np.vstack([C,D[i]]),np.r_[net,k]);check(f'direct_{i}_{k}',np.isclose(direct[k]-base[k],gg[i,k],atol=1e-7) and np.isclose(direct.sum(),P))
 check('tie_split',np.allclose(alloc(np.zeros((2,len(w))),np.array([A,B]))[[A,B]],P/2))
 # Scenario table: fixed-price sensitivity in conversion, margin and non-rent costs.
 st=pd.DataFrame(sens);q=st[(st.strategy=='1100')&((st.model=='nearest')|((st.missing_price==2500)&(st.tau==100)&((st.model=='generalized')|(st.half==10))))];econ=[]
 for r in q.itertuples():
  for th,m,f in itertools.product([.03,.05,.08,.10,.12,.165],[600,900,1200],[0,300000,600000]):econ.append(dict(model=r.model,site=r.site,theta=th,margin_per_member=m,other_fixed=f,rent_max=(r.gain_pp/100*P*th*m-f)/area))
 pd.DataFrame(econ).to_csv(OUT/'rent_sensitivity.csv',index=False);check('overlap_mass',np.allclose(pd.DataFrame(overlap)[['none','one','two_plus']].sum(1),100));check('no_negative_additions',st.gain_pp.min()>=-eps)
 empirical(sets,q); (OUT/'checks.json').write_text(json.dumps({'checks':checks,'address_polygons_inside':ninside,'commercial_polygons':len(b),'population':P},indent=2));figures(b,sets,re,summary,pd.DataFrame(overlap),pd.DataFrame(opts),pd.DataFrame(sens));print('checks',len(checks),flush=True)

def figures(b,sets,re,summary,overlap,opts,sens):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10}); boundary=shape(json.loads((DATA/'boundary.geojson').read_text())['features'][0]['geometry']);
 def baseplot(ax):
  for g in ([boundary] if boundary.geom_type=='Polygon' else boundary.geoms):
   x,y=g.exterior.xy;ax.plot(x,y,color='.5',lw=1)
  ax.scatter(clubs.lon[mobile],clubs.lat[mobile],marker='x',s=35,label='Действующие клубы');ax.set_aspect(1/np.cos(np.deg2rad(55.65)));ax.set_xlabel('Долгота');ax.set_ylabel('Широта');ax.ticklabel_format(useOffset=False)
 def save(fig,name):fig.text(.01,.01,'© OpenStreetMap contributors; архивный снимок, расчёты авторов',fontsize=8);fig.savefig(FIG/name,dpi=160,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5));baseplot(ax);demand=json.loads((DATA/'demand.geojson').read_text())['features']; coords=np.array([f['geometry']['coordinates'] for f in demand]);ax.scatter(coords[:,0],coords[:,1],s=4+70*w/w.max(),alpha=.5,label='Жилые здания (вес населения)');ax.scatter(clubs.lon[net==A],clubs.lat[net==A],s=60,marker='o',label='DDX');ax.legend(fontsize=8);ax.set_title('География потенциального спроса');save(fig,'market.png')
 fig,ax=plt.subplots(figsize=(9,5));baseplot(ax);ax.scatter(b.lon,b.lat,s=10,color='.65',label='Коммерческие контуры'); bb=b[b.in_1100];ax.scatter(bb.lon,bb.lat,s=32,label='Прокси 1100 м²'); z=opts[(opts.market=='broad')&(opts.strategy=='1100')&(opts.network=='DDX')].iloc[0];ax.scatter(z.lon,z.lat,marker='*',s=160,label='Максимум DDX (Хотеллинг)');ax.legend(fontsize=8);ax.set_title('Коммерческие здания и допустимые стратегии');save(fig,'buildings.png')
 fig,ax=plt.subplots(figsize=(9,4.7));order=['grid','500','1100','1500'];t=summary.set_index('strategy').loc[order];ax.bar(np.arange(4)-.17,t['mean'],.34,label='Среднее по сетям');ax.bar(np.arange(4)+.17,t['max'],.34,label='Максимум');ax.set_xticks(np.arange(4),['Сетка','500 м²','1100 м²','1500 м²']);ax.set_ylabel('Выигрыш от переезда, п.п.');ax.legend();save(fig,'regret.png')
 fig,ax=plt.subplots(figsize=(7,4));bot=np.zeros(len(overlap));
 for col,lab in [('none','Ни одной'),('one','Одна сеть'),('two_plus','Две и более')]:ax.bar(overlap.minutes,overlap[col],bottom=bot,width=3,label=lab);bot+=overlap[col].to_numpy()
 ax.set_xticks(overlap.minutes);ax.set_xlabel('Время пешком, мин');ax.set_ylabel('Доля населения, %');ax.legend();save(fig,'overlap.png')
 fig,ax=plt.subplots(figsize=(9,5));baseplot(ax);bb=b[b.in_1100];ax.scatter(bb.lon,bb.lat,s=16,color='.6',label='Прокси площадок');ss=sens[(sens.strategy=='1100')&((sens.model=='nearest')|((sens.missing_price==2500)&(sens.tau==100)&((sens.model=='generalized')|(sens.half==10))))];
 for r in ss.itertuples():ax.scatter(grid.lon[r.site],grid.lat[r.site],s=90,marker={'nearest':'*','generalized':'s','logit':'D'}[r.model],label={'nearest':'Ближайший клуб','generalized':'Цена + дорога','logit':'Logit'}[r.model])
 ax.legend(fontsize=8);ax.set_title('Оптимумы при разных предпосылках спроса');save(fig,'demand.png')
def empirical(sets,q):
 global w,K
 new=w.copy(); rows=[]; loo=[]; stress=[]
 for typ,weights in [('floor',w_osm),('apartments',new)]:
  w=weights; base=alloc(C,net); gg,_=gain(C,net,D)
  for m in ['narrow','wide','broad']:
   keep=clubs[m].to_numpy(bool); v=alloc(C[keep],net[keep])*100/P; g,_=gain(C[keep],net[keep],D)
   for label in ['grid','1100']:
    z=best(sets[label],g[:,A]); rows.append(dict(weights=typ,market=m,strategy=label,hhi=v@v,cr3=np.sort(v)[-3:].sum(),ddx_share=v[A],site=z,gain_pp=g[z,A]*100/P))
  for j in mobile:
   k=net[j]; keep=np.arange(len(C))!=j; without=alloc(C[keep],net[keep])[k]; g,_=gain(C[keep],net[keep],D); actual=base[k]-without
   for label in ['grid','1100']:
    z=best(sets[label],g[:,k]); optimum=max(actual,g[z,k]); loo.append(dict(weights=typ,strategy=label,club=clubs.name[j],network=names[k],actual_pp=actual*100/P,best_pp=optimum*100/P,regret_pp=(optimum-actual)*100/P,efficiency=actual/optimum if optimum>0 else np.nan,site=z))
  for label,cc,nn in [('base',C,net),('miss_as_universal',np.vstack([C,D[825]]),np.r_[net,K]),('exclude_uncertain_external',C[~clubs.origin.isin(['repo_uncertain_price','repo_reviewed'])],net[~clubs.origin.isin(['repo_uncertain_price','repo_reviewed'])])]:
   oldK=K;K=max(K,int(nn.max())+1);v=alloc(cc,nn)*100/P;g,_=gain(cc,nn,D);z=best(sets['1100'],g[:,A]);stress.append(dict(weights=typ,scenario=label,clubs=len(cc),hhi=v@v,ddx_share=v[A],site=z,gain_pp=g[z,A]*100/P));K=oldK
 w=new
 pd.DataFrame(rows).to_csv(OUT/'population_comparison.csv',index=False);pd.DataFrame(loo).to_csv(OUT/'loo.csv',index=False);pd.DataFrame(stress).to_csv(OUT/'club_stress.csv',index=False)
 check('housing_population',np.isclose(w.sum(),P));check('housing_positive',np.all(w>0));check('loo_bounds',pd.DataFrame(loo).efficiency.dropna().between(-1e-8,1+1e-8).all())
 be=[];fin=[]
 for r in q.itertuples():
  for rate in [1167,1300,1650,1833,2500]:
   for m in [600,900,1200]:
    be.append(dict(model=r.model,site=r.site,rent_per_m2=rate,margin_per_member=m,other_fixed=300000,theta_break_even=(rate*1300+300000)/(r.gain_pp/100*P*m)))
  for th in [.03,.05,.08,.10,.12,.165]:
   members=r.gain_pp/100*P*th;fin.append(dict(model=r.model,site=r.site,theta=th,net_members=members,margin=members*900,rent_max_after_fixed=(members*900-300000)/1300))
 pd.DataFrame(be).to_csv(OUT/'break_even_penetration.csv',index=False);f=pd.DataFrame(fin);f.to_csv(OUT/'finance_thresholds.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4.5));
 for model,label in [('nearest','Ближайший клуб'),('generalized','Цена + дорога'),('logit','Logit')]:
  d=f[f.model==model];ax.plot(d.theta*100,d.rent_max_after_fixed,marker='o',label=label)
 ax.axhspan(1167,1833,alpha=.13,label='Архивные ставки аренды');ax.set_xlabel('Доля покупателей, %');ax.set_ylabel('Порог аренды, руб./м²/мес.');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'finance.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
