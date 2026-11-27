"""Boundary stress test using existing out-of-district OSM contours; external demand mass is hypothetical."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
R=Path(sys.argv[1]); O=Path(__file__).parent;tr=Transformer.from_crs(4326,32637,always_xy=True).transform
b=shape(json.loads((R/'data/boundary.geojson').read_text())['features'][0]['geometry']);B=transform(tr,b);f=json.loads((R/'data/address_buildings.geojson').read_text())['features'];rows=[]
for el in f:
 if el['properties'].get('building')!='apartments':continue
 g=shape(el['geometry']);p=g.representative_point();
 if b.covers(p):continue
 G=transform(tr,g);r=G.representative_point(); dist=B.distance(r)
 if dist<=1500: rows.append(dict(osm_id=el['properties']['osm_id'],address=el['properties']['address'],x=r.x,y=r.y,footprint_m2=G.area,boundary_distance_m=dist))
H=pd.DataFrame(rows);H.to_csv(O/'data/buffer_buildings.csv',index=False)
g=nx.read_graphml(sys.argv[2]);ns=list(g.nodes); ix={n:i for i,n in enumerate(ns)};xy=np.array([(float(g.nodes[n]['x']),float(g.nodes[n]['y'])) for n in ns]);tree=cKDTree(xy);edges={}
for a,b,d in g.edges(data=True):
 i,j=ix[a],ix[b];l=float(d['length']);
 if i!=j:edges[i,j]=min(l,edges.get((i,j),np.inf));edges[j,i]=min(l,edges.get((j,i),np.inf))
ij=np.array(list(edges));mat=coo_matrix((list(edges.values()),(ij[:,0],ij[:,1])),shape=(len(ns),len(ns))).tocsr();c=pd.read_csv(R/'data/clubs.csv');gr=pd.read_csv(R/'data/candidates.csv');s=np.unique(pd.read_csv(R/'results/buildings.csv').query('in_1100').site.to_numpy(int)); cx=np.column_stack(tr(c.lon.to_numpy(),c.lat.to_numpy())); X=np.vstack([cx,gr.iloc[s][['x','y']].to_numpy()]);so,si=tree.query(X);ho,hi=tree.query(H[['x','y']].to_numpy());dist=dijkstra(mat,indices=si)[:,hi]+so[:,None]+ho[None,:];assert np.isfinite(dist).all();np.savez_compressed(O/'data/buffer_distances.npz',club=dist[:len(c)]/80,grid=dist[len(c):]/80,site=s);print('buffer buildings',len(H), 'distance matrix',dist.shape)
