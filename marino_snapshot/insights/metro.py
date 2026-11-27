"""Reconstruct metro walking distances using the archived graph; no new routing requests."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer
R=Path(sys.argv[1]); graph=Path(sys.argv[2]); O=Path(__file__).parent
p=pd.read_csv(O/'data/poi.csv'); p=p[p.category=='metro_entrance'].reset_index(drop=True); p.to_csv(O/'data/metro_entrances.csv',index=False)
g=nx.read_graphml(graph); ns=list(g.nodes); ix={n:i for i,n in enumerate(ns)}; xy=np.array([(float(g.nodes[n]['x']),float(g.nodes[n]['y'])) for n in ns]); tree=cKDTree(xy); tr=Transformer.from_crs(4326,32637,always_xy=True)
def project(lon,lat):return np.column_stack(tr.transform(np.asarray(lon),np.asarray(lat)))
def snap(x):return tree.query(x)
co=pd.read_csv(R/'data/clubs.csv');gr=pd.read_csv(R/'data/candidates.csv');f=json.loads((R/'data/demand.geojson').read_text())['features']; h=np.array([i['geometry']['coordinates'] for i in f]);
so,si=snap(project(p.lon,p.lat));ho,hi=snap(project(h[:,0],h[:,1]));coff,ci=snap(project(co.lon,co.lat));do,di=snap(gr[['x','y']].to_numpy()); edges={}
for a,b,d in g.edges(data=True):
 i,j=ix[a],ix[b];l=float(d['length']);
 if i!=j: edges[i,j]=min(l,edges.get((i,j),np.inf));edges[j,i]=min(l,edges.get((j,i),np.inf))
ij=np.array(list(edges));mat=coo_matrix((list(edges.values()),(ij[:,0],ij[:,1])),shape=(len(ns),len(ns))).tocsr(); dm=dijkstra(mat,indices=si)
def dist(idx,off):return (dm[:,idx]+so[:,None]+off[None,:])/80
H=dist(hi,ho); C=dist(ci,coff);D=dist(di,do); assert np.isfinite(H).all() and np.isfinite(C).all() and np.isfinite(D).all()
np.savez_compressed(O/'data/metro_distances.npz',home=H,club=C,grid=D);print('metro',len(p),'nodes',len(ns),'shape',H.shape,C.shape,D.shape)
