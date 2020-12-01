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

def load_graph(filename):
    saved_model = np.load(filename)
    
    masks = []
    for file in saved_model.__dict__['files']:
        if file.startswith('mask'):
            masks.append(saved_model[file])
            
    mask = masks[0].astype(int)
    graph = nx.from_numpy_array(mask, create_using=nx.DiGraph)
    
    return graph

def get_graph_stats(graph):
    return {
        "clustering_coefficient" : nx.average_clustering(graph),
        "average_path_length" : nx.average_shortest_path_length(graph)
    }