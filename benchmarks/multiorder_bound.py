"""Bound of D0 / straight for multiorder_filter_operator on a mixed-level set.

Builds a deterministic set of leaves at levels 11 to 15 refined towards a wavy
coast (lat = 51.4 + 0.05 sin(8 (lon - 4.0)), levels 12 to 15 within 25, 12, 5
and 1.5 km of it), balances it 2:1, and takes the shortest path over the leaf
graph (every leaf passable, edge and corner contacts) from 400 random leaves
(seed 0) to every leaf within 30 km.  Prints the largest ratio of that path to
the straight distance: 1.38 (22765 leaves, 2.6 million pairs).  It sizes the
search factor 1.5 of ``correction="lattice"``.

    python multiorder_bound.py [lon lat]      # default 4.0 51.4
"""
import sys, numpy as np
from healpix_geo import nested
from healpix_analyse.multiorder_filter import leaf_adjacency, _centres, _Graph, _dijkstra
from healpix_analyse._neighbourhood import _wgs84_distance

lon0, lat0 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (4.0, 51.4)
def coast_dist(lon, lat):  # km-ish: distance to a wavy line lat = lat0 + 0.05 sin(8(lon-lon0))
    cl = lat0 + 0.05*np.sin(8*(lon-lon0))
    return np.abs(lat-cl)*111.0
thresh = {11:0.0,12:25.0,13:12.0,14:5.0,15:1.5}  # refine cell of level l if within thresh[l+1] km
def refine(level, ids):
    out=[]; stack=[(level,int(i)) for i in ids]
    while stack:
        l,i=stack.pop()
        lon,lat=nested.healpix_to_lonlat(np.array([i],np.uint64),l,ellipsoid="WGS84")
        if l<15 and coast_dist(lon[0],lat[0])<thresh[l+1]+ (6.0 if l==11 else 0):
            stack += [(l+1,4*i+k) for k in range(4)]
        else: out.append((l,i))
    return out
c=np.asarray(nested.lonlat_to_healpix(np.array([lon0]),np.array([lat0]),11,ellipsoid="WGS84"),dtype=np.uint64)
ring=np.asarray(nested.kth_neighbourhood(c,11,20))[0]; ring=ring[ring>=0]
leaves=set(refine(11,ring))
# balance
while True:
    lv=np.array([a for a,_ in leaves]); ids=np.array([b for _,b in leaves])
    try:
        leaf_adjacency(lv,ids); break
    except ValueError as e:
        msg=str(e); import re
        l=int(re.search(r"level (\d+) with id (\d+)",msg).group(1)); i=int(re.search(r"id (\d+)",msg).group(1))
        up=int(re.search(r"touches a leaf (\d+)",msg).group(1))
        # find the coarse neighbour: refine all leaves coarser than l-1 touching any l leaf -> simply split every leaf with level<l-1 that touches
        nb=np.asarray(nested.neighbours(np.array([i],np.uint64),l,connectivity="all"))[0]
        done=False
        for u in range(2,up+1):
            for n in nb:
                if n<0: continue
                a=(l-u,int(n)>>(2*u))
                if a in leaves:
                    leaves.remove(a); leaves |= {(a[0]+1,4*a[1]+k) for k in range(4)}; done=True
        assert done
lv=np.array([a for a,_ in leaves]); ids=np.array([b for _,b in leaves])
print("leaves",lv.size,{int(l):int((lv==l).sum()) for l in np.unique(lv)})
first,second,edge=leaf_adjacency(lv,ids)
lon,lat=_centres(lv,ids,"WGS84")
length=_wgs84_distance(lon[first],lat[first],lon[second],lat[second])
adj=_Graph(lv.size,first,second,length).lists()
rng=np.random.default_rng(0)
worst=0; pairs=0; 
srcs=rng.choice(lv.size,min(lv.size,400),replace=False)
for s in srcs:
    d=_dijkstra(adj,int(s),30_000.0)
    t=np.array(list(d.keys())); dd=np.array(list(d.values()))
    st=_wgs84_distance(np.full(t.size,lon[s]),np.full(t.size,lat[s]),lon[t],lat[t])
    m=st>0
    r=dd[m]/st[m]; pairs+=r.size
    if r.max()>worst: worst=r.max(); w=(int(s),int(t[m][r.argmax()]),st[m][r.argmax()])
print("pairs",pairs,"max D0/straight",worst,"at",w, "levels",lv[w[0]],lv[w[1]])
