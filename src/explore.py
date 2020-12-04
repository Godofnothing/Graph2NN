import torch
import torch.optim as optim
import torch.nn as nn

import os

from src import train

from models import mlp, cnn

from loaders import cifar10, mnist

import utils.scheduler as sch

def build_model(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if cfg["MODEL"]["TYPE"] == 'mlpnet':
        model = mlp.MLPNet(cfg).to(device)
    elif cfg["MODEL"]["TYPE"] == 'cnnnet':
        model = cnn.CNN(cfg).to(device)
    else:
        raise NotImplementedError()
        
    return model
    
def prepare_data(cfg):
    if cfg["TRAIN"]["DATASET"] == "cifar10":
        train_loader, test_loader = cifar10.prepare_data(cfg)
    elif cfg["TRAIN"]["DATASET"] == "mnist":
        train_loader, test_loader = mnist.prepare_data(cfg)
        
    return train_loader, test_loader

def run_on_param_grid(cfg, param_grid, log = False, return_metrics = True,  evaluate_on_test=True, save_graph = True):
    assert type(param_grid) == dict
    
    if cfg["MODEL"]["LOSS_FUN"] == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()
        
    train_loader, test_loader = prepare_data(cfg)
        
    for sparsity in param_grid["sparsities"]:
        for p in param_grid["rewiring_probabilities"]:
            for graph_seed in param_grid["random_seeds"]:
                cfg["RGRAPH"]["SPARSITY"] = sparsity
                cfg["RGRAPH"]["P"] = p
                cfg["RGRAPH"]["SEED_GRAPH"] = graph_seed
                cfg["RGRAPH"]["SAVE_GRAPH"] = save_graph
                
                model = build_model(cfg)
                
                optimizer = optim.SGD(model.parameters(), 
                                      lr=cfg["OPTIM"]["BASE_LR"], 
                                      momentum=cfg["OPTIM"]["MOMENTUM"], 
                                      weight_decay=cfg["OPTIM"]["WEIGHT_DECAY"])

                scheduler = sch.create_scheduler(optimizer, cfg)
                
                history = train.run_training_procedure(model, 
                                                       cfg=cfg, 
                                                       train_loader=train_loader, 
                                                       loss_fn=loss_fn, 
                                                       optimizer=optimizer, 
                                                       test_loader=test_loader, 
                                                       scheduler=scheduler, 
                                                       log=log, 
                                                       return_metrics=return_metrics, 
                                                       evaluate_on_test=evaluate_on_test)
                
                if return_metrics:
                    out_dir = f"{cfg['OUT_DIR']}/{cfg['MODEL']['TYPE']}/{cfg['TRAIN']['DATASET']}/logs/{cfg['RGRAPH']['GRAPH_TYPE']}"
                    out_file = f"log_gsparsity={sparsity}_p={p}_gseed={graph_seed}.txt"
                    os.makedirs(f"{out_dir}", exist_ok=True)
                    
                    with open(f"{out_dir}/{out_file}", 'w') as f:
                        f.write("Train\nLoss\tTop 1 acc\tTop5 acc\n")
                        for train_loss, acc1, acc5 in zip(history["train_loss"], history["train_acc1"], history["train_acc5"]):
                            f.write(f"{train_loss}\t{acc1}\t{acc5}\n")
                    
                        f.write("Test\nTop 1 acc\tTop5 acc\n")
                        if evaluate_on_test:
                            for acc1, acc5 in zip(history["test_acc1"], history["test_acc5"]):
                                f.write(f"{acc1}\t{acc5}\n")
                    
                