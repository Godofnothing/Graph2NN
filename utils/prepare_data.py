from utils import log_parser
from utils import graph_utils

def collect_logs_and_graphs(cfg, param_grid):

    cs_ = []
    ls_ = []
    train_acc_ = []
    test_acc_ = []

    for sparsity in param_grid["sparsities"]:
        for p in param_grid["rewiring_probabilities"]:
            for seed in param_grid["random_seeds"]:

                graph_dir = f"{cfg['OUT_DIR']}/{cfg['MODEL']['TYPE']}/{cfg['TRAIN']['DATASET']}/graphs/{cfg['RGRAPH']['GRAPH_TYPE']}"
                log_dir = f"{cfg['OUT_DIR']}/{cfg['MODEL']['TYPE']}/{cfg['TRAIN']['DATASET']}/logs/{cfg['RGRAPH']['GRAPH_TYPE']}"
                graph = graph_utils.load_graph(f"{graph_dir}/gsparsity={sparsity}_p={p}_gseed={seed}.npz")

                try:
                  graph_stats = graph_utils.get_graph_stats(graph)
                  train_stats, test_stats = log_parser.parse_log(f"{log_dir}/log_gsparsity={sparsity}_p={p}_gseed={seed}.txt")

                  cs_.append(graph_stats['clustering_coefficient'])
                  ls_.append(graph_stats['average_path_length'])
                  train_acc_.append(train_stats[-1, 1])
                  test_acc_.append(test_stats[-1, 0])

                except:
                  print("Graph is disconnected")
                
    return np.array(cs_), np.array(ls_), np.array(train_acc_), np.array(test_acc_)

def join_graph_data(cs_s, ls_s, train_acc_s, test_acc_s):
    return np.hstack(cs_s), np.hstack(ls_s), np.hstack(train_acc_s), np.hstack(test_acc_s)

def make_griddata(cs, ls, train_acc, test_acc, c_points = 100, l_points = 100):
    grid_c, grid_l = np.mgrid[cs.min():cs.max():c_points * 1j, ls.min():ls.max():l_points * 1j]

    grid_train_acc = interpolate.griddata(
          np.hstack((cs[:, None], ls[:, None])), 
          train_acc, 
          (grid_c, grid_l), method='linear'
    )

    grid_test_acc = interpolate.griddata(
          np.hstack((cs[:, None], ls[:, None])), 
          test_acc, 
          (grid_c, grid_l), method='linear'
    )
    
    return grid_train_acc, grid_test_acc