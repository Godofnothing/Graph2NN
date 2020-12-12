import os

import matplotlib.pyplot as plt

def plot_interpolated_data2d(cfg, grid_c, grid_l, grid_acc, data, train = True, save_image = False, num_lines = 15):
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.contour(grid_c, grid_l, grid_acc, num_lines, linewidths=0.5, colors='k')
    ctrf = ax.contourf(grid_c, grid_l, grid_acc, num_lines)
    
    cs_ws, ls_ws = data["cs_ws"], data["ls_ws"]
    cs_er, ls_er = data["cs_er"], data["ls_er"]

    plt.scatter(cs_ws, ls_ws, marker = '1', s = 100, color = 'blue', label = "watts-strogatz")
    plt.scatter(cs_er, ls_er, marker = '2', s = 100, color = 'red', label = "erdos-renyi")

    plt.colorbar(ctrf, ax = ax)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax.set_xlabel("$C$, clustering coefficient", fontsize = 18)
    ax.set_ylabel("$L$, average path length", fontsize = 18)
    
    if cfg['TRAIN']['DATASET'] == "cifar10":
        ax.set_title("Accuracy on CIFAR-10", fontsize = 20)
    if cfg['TRAIN']['DATASET'] == "mnist":
        ax.set_title("Accuracy on MNIST", fontsize = 20)   
        
    plt.legend(fontsize = 18)

    if save_image == True:
        img_dir = f"{cfg['OUT_DIR']}/{cfg['MODEL']['TYPE']}/{cfg['TRAIN']['DATASET']}/images"
        os.makedirs(img_dir, exist_ok=True)
        if train == True:
            plt.savefig(f"{img_dir}/train_acc_vs_l_and_c.png")
        else:
            plt.savefig(f"{img_dir}/test_acc_vs_l_and_c.png")