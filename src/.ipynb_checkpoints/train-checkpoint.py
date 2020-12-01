import torch
import os
import numpy as np

from utils import net, metrics

def train_epoch(train_loader, model, loss_fn, optimizer, scheduler = None, log = False, return_metrics = False):
    """Performs one epoch of training."""
        
    avg_loss, avg_top1_err, avg_top5_err = 0.0, 0.0, 0.0
    
    # Enable training mode
    model.train()

    for cur_iter, (X_batch, y_batch) in enumerate(train_loader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Transfer the data to the current GPU device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Perform the forward pass
        y_pred = model(X_batch)
        # Compute the loss
        loss = loss_fn(y_pred, y_batch)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        if log or return_metrics:
            # Compute the errors
            top1_err, top5_err = metrics.topk_errors(y_pred, y_batch, [1, 5])
            # Copy the stats from GPU to CPU (sync point)
            avg_loss += loss.item()
            avg_top1_err += top1_err.item()
            avg_top5_err += top5_err.item()
            
    # Log stats
    if log or return_metrics:
        avg_loss /= (cur_iter + 1)
        avg_top1_err /= (cur_iter + 1)
        avg_top5_err /= (cur_iter + 1)
        
    if log:
        print(f"Train\nloss : {avg_loss}\ntop1_err : {avg_top1_err}\ntop5_err : {avg_top5_err}\n---\n")
        
    if scheduler:
        scheduler.step()
        
    if return_metrics:
        return avg_loss, avg_top1_err, avg_top5_err
            
@torch.no_grad()
def eval_epoch(test_loader, model, cfg, log = False, return_metrics = False):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    
    avg_top1_err, avg_top5_err = 0.0, 0.0

    for cur_iter, (X_batch, y_batch)  in enumerate(test_loader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Transfer the data to the current GPU device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Compute the predictions
        y_pred = model(X_batch)
        # Compute the errors
        top1_err, top5_err = metrics.topk_errors(y_pred, y_batch, [1, 5])
        # Copy the errors from GPU to CPU (sync point)
        avg_top1_err += top1_err.item()
        avg_top5_err += top5_err.item()
    
    # Log stats
    avg_top1_err /= (cur_iter + 1)
    avg_top5_err /= (cur_iter + 1)
    if log:
        print(f"Test\ntop1_err : {avg_top1_err}\ntop5_err : {avg_top5_err}\n---\n")
        
    # Log epoch stats
    # test_meter.log_epoch_stats(cur_epoch,writer_eval,params,flops)
    if cfg["RGRAPH"]["SAVE_GRAPH"]:
        adj_dict = net.model2adj(model)
        
        out_dir = f"{cfg['OUT_DIR']}/{cfg['MODEL']['TYPE']}/{cfg['TRAIN']['DATASET']}/graphs"
        rgraph = cfg["RGRAPH"]
        graph_name = f"gsparsity={rgraph['SPARSITY']}_p={rgraph['P']}_gseed={rgraph['SEED_GRAPH']}.npz"
        
        os.makedirs(f"{out_dir}", exist_ok=True)
        np.savez(f"{out_dir}/{graph_name}", **adj_dict)
        
    if return_metrics:
        return avg_top1_err, avg_top5_err
    
def run_training_procedure(
    model, 
    cfg, 
    train_loader, 
    loss_fn, optimizer, 
    test_loader = None, 
    scheduler = None, 
    log = False, 
    return_metrics = False,
    evaluate_on_test = False
):
    
    if return_metrics:
        train_loss, train_acc1, train_acc5 = [], [], []
        test_acc1, test_acc5 = [], []
    
    for epoch in range(cfg["OPTIM"]["MAX_EPOCH"]):
        
        return_metrics_on_epoch = return_metrics & (epoch % cfg["TRAIN"]["EVAL_PERIOD"] == 0)
        
        train_metrics = train_epoch(train_loader, model, loss_fn, optimizer, scheduler, log, return_metrics_on_epoch)
        
        if return_metrics_on_epoch:
            train_loss.append(train_metrics[0])
            train_acc1.append(train_metrics[1])
            train_acc5.append(train_metrics[2])
            
            if evaluate_on_test:
                assert test_loader != None
                test_metrics = eval_epoch(test_loader, model, cfg, log, return_metrics_on_epoch)
                test_acc1.append(test_metrics[0])
                test_acc5.append(test_metrics[1])
                
    if return_metrics:
        return {
            "train_loss" : train_loss,
            "train_acc1" : train_acc1,
            "train_acc5" : train_acc5,
            "test_acc1" : test_acc1,
            "test_acc5" : test_acc5
        }