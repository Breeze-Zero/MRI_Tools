"""
Built from Breeze.
"""
import torch
from torchmetrics import Metric

class DistributedMetricSum(Metric):
    def __init__(self,dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch
        self.num += 1

    def compute(self):
        return self.quantity/self.num
    
    
def normalized_root_mean_squared_error(true, pred):
    '''
    NRMSE function based pytorch
    '''
    squared_error = torch.square((true - pred))
    sum_squared_error = torch.sum(squared_error)
    rmse = torch.sqrt(sum_squared_error / true.size)
    nrmse_loss = rmse/torch.std(pred)
    return nrmse_loss
