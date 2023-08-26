import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    #data_range = data_range[:, None, None, None]
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return 1 - S.mean()

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    
    
class LCC(nn.Module):
    """
    local (over window) normalized cross correlation (square)
    """
    def __init__(self, win=[9, 9], eps=1e-5):
        super(LCC, self).__init__()
        self.win = win
        self.eps = eps
        
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        
        filters = Variable(torch.ones(1, 1, self.win[0], self.win[1])).to(I)

        padding = (self.win[0]//2, self.win[1]//2)
        
        I_sum = F.conv2d(I, filters, stride=1, padding=padding)
        J_sum = F.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)
        
        win_size = self.win[0]*self.win[1]
 
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
 
        cc = cross*cross / (I_var*J_var + self.eps)#np.finfo(float).eps
        lcc = -1.0 * torch.mean(cc) + 1
        return lcc
    
    
# loss = LCC()

# print(loss(torch.randn(2,1,128,128),torch.randn(2,1,128,128)))