import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as io



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_protein():
    n = io.loadmat("data/Homo_sapiens.mat")
    return n['network'], n['group']

def load_enzyme():
    adj = sp.lil_matrix((125, 125))
    features = sp.lil_matrix((125, 1))
    for line in open("data/ENZYMES_g296.edges"):
        vals = line.split()
        x = int(vals[0]) - 2
        y = int(vals[1]) - 2
        adj[y, x] = adj[x, y] = 1
    return adj, features

def load_florida():
    adj = sp.lil_matrix((128, 128))
    features = sp.lil_matrix((128, 1))
    for line in open("data/eco-florida.edges"):
        vals = line.split()
        x = int(vals[0]) - 1
        y = int(vals[1]) - 1
        val = float(vals[2])
        adj[y, x] = adj[x, y] = val
    return adj, features

def load_brain():
    adj = sp.lil_matrix((1780, 1780))
    features = sp.lil_matrix((1780, 1))
    nums = []
    for line in open("data/bn-fly-drosophila_medulla_1.edges"):
        vals = line.split()
        x = int(vals[0]) - 1
        y = int(vals[1]) - 1
        adj[y, x] = adj[x, y] = adj[x, y] + 1
    return adj, features


def load_data(dataset):
    if dataset == 'florida':
        return load_florida()
    elif dataset == 'brain':
        return load_brain()
    elif dataset == 'enzyme':
        return load_enzyme()
    elif dataset == 'protein':
        return load_protein()

    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    # if dataset == 'cora':
    #     names = ['y', 'ty', 'ally']
    #     objects = []
    #     for i in range(len(names)):
    #         objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    #     y, ty, ally = tuple(objects)

    #     labels = np.vstack((ally, ty))
    #     labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #     np.save('labels', labels)


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features



