import torch
import fastmri
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR
from MRI_Tools.SMS.readoutCascadeSmsData import readoutCascadeRestoreReconSlices,SlicesRestore
import numpy as np
from timm.models.layers import trunc_normal_
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class SSRAKI_training_dataset(torch.utils.data.Dataset):
    def __init__(self,acs,index_list,coils,slice_index,channel_index):
        self.listOfSliceIdcs = index_list
        self.coils = coils
        self.acs = acs
        
        self.calX = torch.zeros((len(self.listOfSliceIdcs),self.acs.shape[1],self.acs.shape[2],self.acs.shape[3]),dtype=torch.float32)
        #Y is the ideal, single-slice, single-coil data: len × 2 (R/I) x rows x cols 
        self.calY = torch.zeros((len(self.listOfSliceIdcs),2,self.acs.shape[2],self.acs.shape[3]),dtype=torch.float32)
        #Loop and make the calibration/unaliased sets
        for idx in range(len(self.listOfSliceIdcs)):
            #Sum across the aliased slices
            self.calX[idx] = torch.sum(self.acs[self.listOfSliceIdcs[idx],:,:,:],dim=0)
            self.calX[idx] /= float(len(self.listOfSliceIdcs[idx]))
            
            #Set the ideal, single-slice, single-coil data to be non-zero if that slice/coil is in the aliased set
            if slice_index in self.listOfSliceIdcs[idx]:
                self.calY[idx,0] = self.acs[slice_index,channel_index]
                self.calY[idx,1] = self.acs[slice_index,channel_index+self.coils]
        
    def __len__(self):
        return len(self.listOfSliceIdcs)

    def __getitem__(self, idx: int): 
        return self.calX[idx],self.calY[idx]


class SplitSliceRAKI(nn.Module):
    def __init__(self,kspace,acs,kernel_size=9,width=64,num_layers=7,epochs = 200,learning_rate=1e-4,batch_size = 4,device='cuda:0'):
        super(SplitSliceRAKI, self).__init__()
        '''
        kspace: coil,x,y,2
        acs: slice,coil,x,y,2
        '''
        self.coils = acs.shape[1]
        self.kernel_size = kernel_size
        self.width = width
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        self.kspace = torch.concat((kspace[...,0],kspace[...,1]),dim=0).unsqueeze(0)
        self.aliasedMean = torch.mean(self.kspace) 
        self.aliasedVar = torch.var(self.kspace) 
        self.kspace = (self.kspace-self.aliasedMean)/self.aliasedVar
        self.kspace = self.kspace.to(self.device)
        self.acs = torch.concat((acs[...,0],acs[...,1]),dim=1)
        self.acs = (self.acs-torch.mean(self.acs))/torch.var(self.acs)
 
        
        self.smsR = acs.shape[0]

        self.build_split_slice_training_index()
        
        self.loss_func = self.define_loss().to(self.device)
        

    def build_split_slice_training_index(self):
        listOfIterators = []
        self.listOfSliceIdcs = []
        for idx in range(1,self.smsR+1):
            listOfIterators.append(itertools.combinations(np.arange(self.smsR),idx))
        for iteratorInst in listOfIterators:
            for element in iteratorInst:
                tmpElement = list(element)
                self.listOfSliceIdcs.append(tuple(tmpElement))
    
    def build_split_slice_trainset(self,slice_index,channel_index):
        return SSRAKI_training_dataset(self.acs,self.listOfSliceIdcs,self.coils,slice_index,channel_index)
        
        
    def build_network(self):
        """
        BASE version. Other modes override this function
        :return: pytorch net object
        """
        mid_layers = []
        for i in range(self.num_layers-2):
            out_chan = self.width
            # if i == self.num_layers-3:
                # out_chan = 1024
            mid_layers.append(nn.Conv2d(in_channels=self.width,out_channels=out_chan,kernel_size=1,padding='same',bias=True))
                # mid_layers.append(nn.InstanceNorm2d(out_chan))
                # mid_layers.append(nn.ReLU(inplace=True))
                
            # else:
            #     mid_layers.append(nn.Conv2d(in_channels=self.width,out_channels=self.width,kernel_size=self.kernel_size,padding='same',bias=True))
            #     # mid_layers.append(nn.InstanceNorm2d(self.width))
            #     # mid_layers.append(nn.ReLU(inplace=True))
            #     # mid_layers.append(nn.Dropout2d(0.5))
            
        net = nn.Sequential(
            nn.Conv2d(in_channels=self.coils*2, out_channels=self.width, kernel_size=self.kernel_size,padding='same',bias=False),
            # nn.BatchNorm2d(self.width),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5),
            *mid_layers,
            nn.Conv2d(in_channels=self.width, out_channels=2, kernel_size=self.kernel_size,padding='same',bias=False),
        ).to(self.device)
        
        return net.apply(self.weights_init)
    
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    # @staticmethod
    def define_loss(self):
        """
        define the loss function for the network
        :return: loss function handle
        """
        return torch.nn.L1Loss()#reduction='sum'

    def define_opt(self):
        """
        define the network optimizer
        :return: optimizer object
        """
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def define_lr_sched(self):
        return CosineAnnealingLR(optimizer = self.optimizer,T_max = self.num_epochs)

    
    def train(self,slice_index,channel_index):
        train_loader = torch.utils.data.DataLoader(self.build_split_slice_trainset(slice_index,channel_index),
                                                   batch_size=self.batch_size,shuffle=True,drop_last=False,
                                                   num_workers=0)
        
        for epoch in range(self.num_epochs):
            sum_loss = 0
            for step, (inputs, labels) in enumerate(train_loader):
                inputs, labels= inputs.to(self.device), labels.to(self.device)
                pred = self.net(inputs)
                loss = self.loss_func(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print_loss = loss.data.item()
                sum_loss += print_loss
                
            self.scheduler.step()
            avg_loss = sum_loss / len(train_loader)
            if epoch+1==self.num_epochs:
                print('Epoch:{}, loss:{}'.format(epoch+1, avg_loss))

    def recon(self):
        recon = torch.zeros((self.smsR,self.coils,self.kspace.shape[2],self.kspace.shape[3],2),dtype=torch.float32)
        for slice in range(self.smsR):
            for coil in range(self.coils):
                self.net = self.build_network()
                self.optimizer = self.define_opt()
                self.scheduler = self.define_lr_sched()
                self.net.train()
                self.train(slice,coil)
                self.net.eval()
                with torch.no_grad():
                    output = self.net(self.kspace)[0].cpu()
                recon[slice,coil,...,0] = output[0]
                recon[slice,coil,...,1] = output[1]
        recon = recon*self.aliasedVar+self.aliasedMean
        return recon
    
    
if __name__ == "__main__":    
    from MRI_Tools.SMS.BuildDataset import SliceDataset_SMS_val
    MB_FACTOR = 4
    SHIFT_FOV = 4
    pd=30
    dataset = SliceDataset_val(root='/data0/mlf_temp_dataset/val',challenge='multicoil',pd = 15 ,mb_factor = MB_FACTOR,shift_fov = SHIFT_FOV)
    data = dataset[0]
    _,_,x,y,_ = data[3].shape
    acs = data[3][...,(x-pd)//2:(x+pd)//2,(y-pd)//2:(y+pd)//2,:]
    kspace = data[0]
    # print(torch.max(kspace),torch.max(acs))
    # kspace.shape,acs.shape
    model = SplitSliceRAKI(kspace,acs,kernel_size=9,width=64,num_layers=7,epochs = 600,learning_rate=1e-4,batch_size = 2**4).cuda()
    output = model.recon()
    
    o = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(output)), dim=1).numpy()
    o = SlicesRestore(o,[i/MB_FACTOR for i in range(MB_FACTOR)])
    target = SlicesRestore(data[1].numpy(),[i/MB_FACTOR for i in range(MB_FACTOR)])
    o = o/np.max(o)
    target = target/np.max(target)
    print(ssim(target,o).item())
    print(psnr(target,o))

    from matplotlib import pyplot as plt
    def show_slice(data, slice_nums, cmap='gray',vmax=None):
        fig = plt.figure(figsize=(10, 10))
        for i, num in enumerate(slice_nums):
            plt.subplot(1, len(slice_nums), i + 1)
            plt.imshow(data[num], cmap=cmap,vmax=vmax)   
            plt.axis('off')
    show_slice(o, range(o.shape[0]))
    plt.savefig('./SSRAKI_linear.png',dpi=400,bbox_inches = 'tight')
    show_slice(np.abs(target-o), range(target.shape[0]),'jet',vmax=np.max(target)/10)
    plt.savefig('./SSRAKI_linear_error.png',dpi=400,bbox_inches = 'tight')