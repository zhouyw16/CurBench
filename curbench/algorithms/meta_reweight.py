from .base import BaseTrainer, BaseCL
import torch
from torch.utils.data import Subset
import numpy as np
import torch.nn as nn
from torch.optim.sgd import SGD
from torch_geometric.data.data import Data as pygData
import copy

from .utils import set_parameter



class MetaReweight(BaseCL):
    """Meta Reweight CL Algorithm.
    
    Learning to reweight examples for robust deep learning. http://proceedings.mlr.press/v80/ren18a/ren18a.pdf
    """
    def __init__(self, ):
        super(MetaReweight, self).__init__()
        self.name = 'meta_reweight'
    

    def randomSplit(self):
        """split data into train and validation data by proportion 9:1"""
        sample_size = self.data_size//10
        temp = np.array(range(self.data_size))
        np.random.shuffle(temp)
        valid_index = temp[:sample_size].tolist()
        train_index = temp[sample_size:].tolist()
        self.validationData = self._dataloader(Subset(self.dataset, valid_index), shuffle=False)
        self.trainData = self._dataloader(Subset(self.dataset, train_index))
        self.iter = iter(self.trainData)
        self.iter2 = iter(self.validationData)
        self.weights = torch.zeros(self.data_size)


    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self.randomSplit()


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.weights = self.weights.to(self.device)
        

    def data_curriculum(self, **kwargs):
        self.net.train()
        
        try:
            temp = next(self.iter)
        except StopIteration:
            # self.trainData = DataLoader(self.trainData.dataset, self.batch_size, shuffle=True)
            self.iter = iter(self.trainData)
            temp = next(self.iter)
        
        try:
            temp2 = next(self.iter2)
        except StopIteration:
            # self.validationData = DataLoader(self.validationData.dataset, self.batch_size, shuffle=True)
            self.iter2 = iter(self.validationData)
            temp2 = next(self.iter2)
        if isinstance(temp,list):
            inputs, labels, indices = temp
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            indices = indices.to(self.device)
        elif isinstance(temp, dict):
            for _, data in enumerate(self.trainData):
                inputs = {k: v.to(self.device) for k, v in data.items() 
                          if k not in ['labels', 'indices']}
                labels = data['labels'].to(self.device)
                indices = data['indices'].to(self.device)
        elif isinstance(temp[0],pygData):
            for _,data in enumerate(self.trainData):
                inputs = data.to(self.device)
                labels = data.y.to(self.device)
                indices = data.i
        else:
            NotImplementedError()
        pseudonet = copy.deepcopy(self.net)
        if isinstance(temp, dict):
            with torch.backends.cudnn.flags(enabled=False):
                out = pseudonet(**inputs)[0]
        else :
            out = pseudonet(inputs)
        loss = self.criterion(out, labels)
        eps = nn.Parameter(torch.zeros(loss.size())).to(self.device)
        lr = 0.001
        totalloss1 = torch.sum(eps * loss)

        grad = torch.autograd.grad(totalloss1, pseudonet.parameters(), create_graph=True, retain_graph=True)

        for (name, parameter), j in zip(pseudonet.named_parameters(), grad):
            parameter.detach_()
            set_parameter(pseudonet, name, parameter.add(j, alpha = -lr))


        totalloss2 = 0
        if isinstance(temp2,list):
            inputs2, labels2, indices2 = temp2
            inputs2 = inputs2.to(self.device)
            labels2 = labels2.to(self.device)
            indices2 = indices2.to(self.device)
        elif isinstance(temp, dict):
            for _, data in enumerate(self.validationData):
                inputs2 = {k: v.to(self.device) for k, v in data.items() 
                          if k not in ['labels', 'indices']}
                labels2 = data['labels'].to(self.device)
                indices2 = data['indices'].to(self.device)
        elif isinstance(temp2[0],pygData):
            for _,data in enumerate(self.validationData):
                inputs2 = data.to(self.device)
                labels2 = data.y.to(self.device)
                indices2 = data.i
        if isinstance(temp2, dict):
            with torch.backends.cudnn.flags(enabled=False):
                out2 = pseudonet(**inputs2)[0]
        else:
            out2 = pseudonet(inputs2)
        loss2 = self.criterion(out2, labels2)
        totalloss2 += torch.sum(loss2)

        grad_eps = torch.autograd.grad(totalloss2, eps)
        w_tilde = torch.clamp(-grad_eps[0], min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde
        w = w * self.batch_size
        self.weights[indices] = w.view(1, -1).detach()
        return self._dataloader(self.dataset)


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        return torch.mean(self.criterion(outputs, labels) * self.weights[indices])



class MetaReweightTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed):
        
        cl = MetaReweight()

        super(MetaReweightTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)