import numpy as np

from torch import optim

class LR_Steps:
    def __init__(self, cfg):
        self.lr_base = cfg["OPTIM"]["BASE_LR"] 
        self.lr_mult = cfg["OPTIM"]["LR_MULT"] 
        self.steps   =  cfg["OPTIM"]["STEPS"]
        # the current index in the steps
        self.ind = 0
        
    def __call__(self, epoch):
        if epoch > self.steps[self.ind]:
            self.ind += 1
        
        return self.lr_base * (self.lr_mult ** self.ind)
    
class LR_Exp:
    def __init__(self, cfg):
        self.lr_base = cfg["OPTIM"]["BASE_LR"] 
        self.lr_mult = cfg["OPTIM"]["LR_MULT"] 
        
    def __call__(self, epoch):
        return self.lr_base * (self.lr_mult ** epoch)
    
class LR_Cos:
    def __init__(self, cfg):
        self.lr_base = cfg["OPTIM"]["BASE_LR"] 
        self.max_epoch = cfg["OPTIM"]["MAX_EPOCH"]
        
    def __call__(self, epoch):
        return 0.5 * self.lr_base * (1.0 + np.cos(np.pi * epoch / self.max_epoch))
    
lr_policy_dict = {
    "cos" : LR_Cos,
    "exp" : LR_Exp,
    "steps" : LR_Steps
}

def create_scheduler(optimizer, cfg):
    return optim.lr_scheduler.LambdaLR(optimizer, lr_policy_dict[cfg["OPTIM"]["LR_POLICY"]](cfg))