import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.utils import py_random_state
from matplotlib.colors import ListedColormap

import pdb

def plot_graph(graph, name, dpi=200, width=0.5, layout='spring'):
    plt.figure(figsize=(10, 10))
    pos = nx.spiral_layout(graph)
    if layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    nx.draw(graph, pos=pos, node_size=100, width=width)
    plt.savefig('figs/graph_view_{}.png'.format(name), dpi=dpi, transparent=True)
    
def load_graph(name, verbose=False, seed=1):
    if 'raw' in name:
        name = name[:-3]
        directed = True
    else:
        directed = False
    filename = '{}.txt'.format(name)
    # filename = 'pycls/datasets/{}.txt'.format(name)
    with open(filename) as f:
        content = f.readlines()
    content = [list(x.strip()) for x in content]
    adj = np.array(content).astype(int)
    if not directed:
        adj = np.logical_or(adj.transpose(), adj).astype(int)

    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    if verbose:
        print(type(graph))
        print(graph.number_of_nodes(), graph.number_of_edges())
        print(compute_stats(graph))
        print(len(graph.edges))
        # plot_graph(graph, 'mc_whole', dpi=60, width=1, layout='circular')
        cmap = ListedColormap(['w', 'k'])
        plt.matshow(nx.to_numpy_matrix(graph), cmap=cmap)
        plt.show()
    return graph