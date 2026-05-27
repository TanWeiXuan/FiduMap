from __future__ import annotations

def build_tracks_union_find(edges):
    parent={}; images={}
    def f(x):
        parent.setdefault(x,x)
        while parent[x]!=x: x=parent[x]
        return x
    def u(a,b):
        ra,rb=f(a),f(b)
        if ra==rb:return
        ia={k[0] for k in images.get(ra,{a})}; ib={k[0] for k in images.get(rb,{b})}
        if ia&ib:return
        parent[rb]=ra; images[ra]=images.get(ra,{a})|images.get(rb,{b})
    for a,b in edges: u(a,b)
    comps={}
    for x in list(parent): comps.setdefault(f(x),set()).add(x)
    return [v for v in comps.values() if len(v)>=2]
