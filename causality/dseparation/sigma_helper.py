from ast import excepthandler
from curses import endwin
import pdb
import argparse

from itertools import chain, combinations, product

import networkx as nx
import matplotlib.pyplot as plt


def is_ancestor(g, n, z):
    for zi in z:
        if zi in g and nx.has_path(g, n, zi):
            return True
    return False


def _ancestors(graph, node, mark=None):
        ancestors = set([node])
        mark[node] = True
        for parent in graph.predecessors(node):
            if parent not in mark:
                ancestors = ancestors.union(_ancestors(graph, parent, mark))
        return sorted(ancestors)


def _descendants(graph, node, mark=None):
        descendents = set([node])
        mark[node] = True
        for child in graph.successors(node):
            if child not in mark:
                descendents = descendents.union(_descendants(graph, child, mark))
        return sorted(descendents)


def SCC(g, n, gh=None, mark=None):
    '''
    Returns the strongly connected component of node n in graph g
    '''
    hk = gh + str(n)
    if hk not in mark:
        mark[hk] = sorted(set(_ancestors(g, n, {})).intersection(_descendants(g, n, {}))) 
    return mark[hk]


def d_condition(ogg, egg, path, n):
    return True


def sigma_condition(ogg, egg, path, n, oggh=None, mark=None):
    ni = path.index(n)
    scc_n = SCC(ogg, n, oggh, mark)

    scc_n_prev = SCC(ogg, path[ni-1], oggh, mark)
    scc_n_next = SCC(ogg, path[ni+1], oggh, mark)

    # if ','.join([str(n) for n in path]) == '[A].X3,[A].X1,[A].X2,[A].X4,[A].X5' and str(n) == '[A].X1':
    #     pdb.set_trace()

    if egg.has_edge(n, path[ni-1]) and (scc_n or scc_n_prev) and (scc_n != scc_n_prev):
        return True

    
    if egg.has_edge(n, path[ni+1]) and (scc_n or scc_n_next) and (scc_n != scc_n_next):
        return True

    return False


def get_sep_func(sep_type):
    if sep_type == 'd':
        return d_condition
    elif sep_type == 'sigma':
        return sigma_condition
    else:
        raise NotImplemented


def split_edges(gg):
    uni_edges = []
    bi_edges = []
    for e in gg.edges():
        if (e[1], e[0]) in gg.edges():
            if e[0] < e[1]:
                bi_edges.append([e, (e[1], e[0])])
        else:
            uni_edges.append(e)

    return uni_edges, bi_edges


def expand_gg(gg):
    uni_edges, bi_edges = split_edges(gg)

    egg = []
    for edges in sorted(product(*bi_edges)):
        eg = nx.DiGraph(list(edges) + uni_edges)
        # draw_graph(eg, True)
        egg.append(eg)
    return egg


def is_blocked_expand(g, x, y, z, sep_type):
    eggs = expand_gg(g)
    return all(is_blocked(g, egg, x, y, z, sep_type) for egg in eggs)


def is_blocked(ogg, g, x, y, z, sep_type):
    '''	
    Assuming g as aDMG where x, y are single nodes, z is a set of nodes (can be empty)	
    '''	
    def is_collider(g, path, n):
        ni = path.index(n)
        return g.has_edge(path[ni-1], n) and g.has_edge(path[ni+1], n)
    sep_condition = get_sep_func(sep_type)
    gg = nx.Graph(g)
    result = True

    if (x not in gg) or (y not in gg):
        return True

    if (x in z) or (y in z):
        return True

    if len(z) == 0 and ((x,y) in gg.edges()) or ((y,x) in gg.edges()):
        return False

    condition = False #str(x) == '[A, AC, C, AC, A].X2' and str(y) == '[A].X1' and len(z) > 0 and str(list(z)[0]) == '[A, AC, C, AC, A].X3'
    if condition:
        pdb.set_trace()

    bcache = {}
    oggh = str(id(ogg))
    paths = list(nx.all_simple_paths(gg, source=x, target=y))
    for path in paths:
        inner_nodes = list(path)[1:-1]

        if len(z) == 0 or len(set(inner_nodes).intersection(set(z))) == 0:
            reachable = True
            for n in inner_nodes:
                if is_collider(g, path, n):
                    reachable = False
                    break
            if reachable:
                result = False
                if condition: 
                    pdb.set_trace()
                break
            else:
                continue

        if condition: 
            pdb.set_trace()
        
        for n in inner_nodes:
            if n in z:
                result = not is_collider(g, path, n) and sep_condition(ogg, g, path, n, oggh, bcache)
                if condition: 
                    pdb.set_trace()
            else:
                result = is_collider(g, path, n) and not is_ancestor(g, n, z)
                if condition: 
                    pdb.set_trace()
            if result:
                if condition: 
                    pdb.set_trace()
                break

        if result == False:
            if condition: 
                pdb.set_trace()
            break

    return result


def dseparations(gg, sep_type='d'):

    # gg.remove_nodes_from(list(nx.isolates(gg)))

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return sorted(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

    dseps = [] 
    for x,y in combinations(gg.nodes(), 2):
        if gg.has_edge(x,y) or gg.has_edge(y,x):
            continue
        
        others = list(set(gg.nodes()) - set([x, y]))
        for z in powerset(others):
            # print('%s _|_ %s | %s' % (x, y, ''.join(z)))
            if is_blocked_expand(gg, x, y, z, sep_type):
                xy = sorted([x, y])
                dsep = "%s _|_ %s | {%s}" % (xy[0], xy[1], ", ".join(sorted([str(zi) for zi in z])))
                dseps.append(dsep)
    
    return dseps


def align_dseps(dseps):
    all_dseps = set()
    for k,dsep in dseps.items():
        all_dseps = all_dseps.union(set(dsep))

    adseps = {}
    for k in dseps.keys():
        adseps[k] = []

    for ds in all_dseps:
        for k in dseps.keys():
            adseps[k].append(ds if ds in dseps[k] else '***')
    return adseps


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-config", type=str, default='conf/synth_no_peer_dep.json', help="config", required=True)
    parser.add_argument("--sm", help="Generate synthetic models", action='store_true', required=False)
    parser.add_argument("-i", type=int, default=-1, help="Specific id of synthetic model to run", required=False)
    parser.add_argument("-sep", type=str, default='d', help="d-sep or sigma-sep", required=False)
    parser.add_argument("--o", help="Output only?", action='store_true', required=False)
    parser.add_argument("--t", help="Times only?", action='store_true', required=False)
    parser.add_argument("--no-draw", help="Disable drawing?", action='store_true', required=False)
    args = parser.parse_args()


if __name__ == '__main__':
    main()