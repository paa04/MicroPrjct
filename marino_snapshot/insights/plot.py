"""Figures for the empirical section, from saved/recomputed results."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
R=ROOT/'insights'/'results'
F=ROOT/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
models=['nearest','generalized','logit']
labels=['Ближайший клуб','Цена + дорога','Logit']
ratings=pd.read_csv(ROOT/'insights'/'data'/'ratings_openings.csv')[['id','rating']]
rel=pd.read_csv(R/'relocation_prices.csv').query('fill==2500 and tau==100 and half in [0,10]').merge(ratings,on='id')
perm=pd.read_csv(R/'rating_permutation.csv').set_index('model')
fig,axs=plt.subplots(1,3,figsize=(10,3.3),sharey=True)
for ax,model,label in zip(axs,models,labels):
    s=rel[rel.model==model]
    ax.scatter(s.efficiency*100,s.rating,s=35,color='#23627d',alpha=.9)
    for i,short in [(1,'Люблю Молл'),(2,'Spirit'),(5,'Magneto')]:
        q=s[s.id==i].iloc[0]
        # Near coincidences between Spirit and Magneto are deliberately offset.
        offset={1:(4,-13),2:(-5,-13),5:(-5,8)}[i]
        ax.annotate(short,(q.efficiency*100,q.rating),xytext=offset,textcoords='offset points',ha='left' if i==1 else 'right',fontsize=8)
    ax.set(title=f'{label}\nρ = {perm.loc[model,"rho"]:.3f}',xlim=(-3,108),ylim=(4.43,5.10),xlabel='Эффективность, %')
    ax.grid(alpha=.22)
axs[0].set_ylabel('Рейтинг')
fig.tight_layout(); fig.savefig(F/'ratings_comparison.png',dpi=180,bbox_inches='tight'); plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
metro=pd.read_csv(R/'metro_relocation.csv').query('id==1')
bound=pd.read_csv(R/'boundary_sensitivity.csv').query('id==1 and radius_m==1000')
for m,l,c in zip(models,labels,['#23627d','#b16c26','#338576']):
    s=metro[metro.model==m].sort_values('commute_fraction'); axs[0].plot(s.commute_fraction*100,s.efficiency*100,'o-',label=l,color=c)
    s=bound[bound.model==m].sort_values('external_mass_ratio'); axs[1].plot(s.external_mass_ratio*100,s.efficiency*100,'o-',label=l,color=c)
axs[0].set(xlabel='Доля посещений по пути от метро, %',ylabel='Эффективность Люблю Молл, %',title='Маршрут от метро'); axs[1].set(xlabel='Внешняя масса к исходной, %',title='Пограничный спрос (буфер 1 км)')
for ax in axs: ax.grid(alpha=.25); ax.set_ylim(0,100)
axs[1].legend(loc='upper left',fontsize=8)
fig.tight_layout(); fig.savefig(F/'mall_sensitivity.png',dpi=180,bbox_inches='tight'); plt.close(fig)
