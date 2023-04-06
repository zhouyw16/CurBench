from .base import BaseTrainer, BaseCL

import copy

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.optim.sgd import SGD
import numpy as np

from .utils import VNet_, set_parameter



class DDS(BaseCL):
    """
    
    Optimizing data usage via differentiable rewards. http://proceedings.mlr.press/v119/wang20p/wang20p.pdf
    """
    def __init__(self, catnum, epsilon, lr, net_type):
        super(DDS, self).__init__()

        self.name = 'dds'
        self.catnum = catnum
        self.epsilon = epsilon
        self.lr = lr
        self.net_type = net_type

    def randomSplit(self):
        """split data into train and validation data by proportion 9:1"""
        sample_size = self.data_size//10
        temp = np.array(range(self.data_size))
        np.random.shuffle(temp)
        valid_index = temp[:sample_size]
        train_index = temp[sample_size:]
        self.validationData = self._dataloader(Subset(self.dataset, valid_index), shuffle=False)
        self.trainData = self._dataloader(Subset(self.dataset, train_index))
        self.iter1 = iter(self.trainData)
        self.iter2 = iter(self.validationData)

        self.weights = torch.zeros(self.data_size)
       

    def data_prepare(self, loader, **kwargs):
        super().data_prepare(loader)
        self.randomSplit()


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.weights = self.weights.to(self.device)
        self.last_net = copy.deepcopy(self.net)
        self.vnet_ = copy.deepcopy(self.net)
        self.linear = VNet_(self.catnum, 1).to(self.device)
        # vision
        if self.net_type == 'vision':
            self.input, self.label, self.indices = next(self.iter1)
        # text
        if self.net_type == 'text':
            temp = next(self.iter1)
            self.input = temp['input_ids']
            self.label = temp['labels']
            self.indices = temp['indices']
        # graph
        if self.net_type == 'graph':
            # input: (x, edge_index, batch), y, i
            # edge_index, x, edge_attr, y, i, batch, ptr = 
            self.input = next(self.iter1)
            self.label = self.input.y
            self.indices = self.input.i


    def data_curriculum(self, **kwargs):
        self.net.train()
        self.vnet_.train()
        self.linear.train()
        try:
            temp2 = next(self.iter2)
        except StopIteration:
            self.validationData = self._dataloader(self.validationData.dataset)
            self.iter2 = iter(self.validationData)
            temp2 = next(self.iter2)

        if self.net_type in ['text', 'vision']:
            input, labels, indices = self.input, self.label, self.indices
        if self.net_type == 'graph':
            input, labels, indices = self.input, self.label, self.indices
                   
        # Sample B training data points
        input = input.to(self.device)
        labels = labels.to(self.device)
        indices = indices.to(self.device)

        if self.net_type == 'vision':        
            out = self.last_net(input)
        if self.net_type == 'text':
            out = self.last_net(input)[0]
        if self.net_type == 'graph':
            out = self.last_net(input)    
        # out = self.last_net(input)
#        self.last_net.zero_grad()
        with torch.no_grad():       
            loss = self.criterion(out, labels)


        # Sample B validation data points
        if self.net_type == 'vision':        
            input2, labels2, indices2 = temp2
        if self.net_type == 'text':
            input2 = temp2['input_ids']
            labels2 = temp2['labels']
        if self.net_type == 'graph':
            input2 = temp2
            labels2 = input2.y
            # indicesw = input2.i
        
        input2 = input2.to(self.device)
        labels2 = labels2.to(self.device)
        
        if self.net_type == 'vision':        
            out2 = self.net(input2)
        if self.net_type == 'text':
            out2 = self.net(input2)[0]
        if self.net_type == 'graph':
            out2 = self.net(input2)
        
        loss2 = self.criterion(out2, labels2)
        totalloss2 = torch.mean(loss2)
        self.net.zero_grad()
        # \Delta_\theta l(x'_i, y'_i; \theta_t)
        grad = torch.autograd.grad(totalloss2, self.net.parameters(), create_graph=True, retain_graph=True)

        for (name, parameter), j in zip(self.last_net.named_parameters(), grad):
            parameter.detach_()
            set_parameter(self.last_net, name, parameter.add(j, alpha = -self.epsilon))
        # l(x', y'; \theta_t)
        with torch.no_grad():
            if self.net_type == 'vision':        
                loss3 = self.criterion(self.last_net(input), labels)
            if self.net_type == 'text':
                loss3 = self.criterion(self.last_net(input)[0], labels)
            if self.net_type == 'graph':
                loss3 = self.criterion(self.last_net(input), labels)

        # r_i ?
        r = (loss3 -loss)/self.epsilon

        if self.net_type == 'vision':        
            out3 = self.vnet_(input)
        if self.net_type == 'text':
            out3 = self.vnet_(input)[0]
        if self.net_type == 'graph':
            out3 = self.vnet_(input)
        

        out4 = out3.reshape(out3.size() , -1)
        out5 = self.linear(out4)
        out5_norm = torch.sum(out5)
        # grad0 = torch.autograd.grad(out5_norm, self.vnet_.parameters())
        if out5_norm != 0:
            out5_ = out5/out5_norm
        else:
            out5_ = out5
        L = torch.sum(r * torch.log(out5_) )
        # print(grad2)
        grad1 = torch.autograd.grad(L, self.linear.parameters(), create_graph=True, retain_graph=True)
        grad2 = torch.autograd.grad(L, self.vnet_.parameters())
        # grad2 = torch.autograd.grad(L, self.vnet_.parameters())
        #print(grad1)
        # print(grad2)
        
        # replace nan with 0
        # for a in grad2:
        #     a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
        for (name, parameter), j in zip(self.linear.named_parameters(), grad1):
            set_parameter(self.linear, name, parameter.add(j, alpha = -self.lr))
        for (name, parameter), j in zip(self.vnet_.named_parameters(), grad2):
            set_parameter(self.vnet_, name, parameter.add(j, alpha = -self.lr))
        del grad1
        del grad2
        del self.last_net
        self.last_net = copy.deepcopy(self.net)

        try:
            temp = next(self.iter1)
        except StopIteration:
            self.trainData = self._dataloader(self.trainData.dataset)
            self.iter1 = iter(self.trainData)
            temp = next(self.iter1)
            
        if self.net_type == 'vision':        
            a, b, i = temp
        if self.net_type == 'text':
            a = temp['input_ids']
            b = temp['labels']
            i = temp['indices']
        if self.net_type == 'graph':
            a = temp
            b = temp.y
            i = temp.i

        self.input = copy.deepcopy(a)
        self.label = copy.deepcopy(b)

        a = a.to(self.device)
        #print(a)
        self.vnet_.eval()
        self.linear.eval()

        if self.net_type == 'vision':        
            z = self.vnet_(a)
        if self.net_type == 'text':
            z = self.vnet_(a)[0]
        if self.net_type == 'graph':
            z = self.vnet_(a)

        #print(z)
        w = self.linear(z)    
        w_norm = torch.sum(w)
        if w_norm != 0:
            w_ = w / w_norm
        else :
            w_ = w
        #print(w_)
#        c = Variable(a, requires_grad=False)
#        d = Variable(b, requires_grad=False)
#        e = Variable(w_, requires_grad=False)
        a.detach_()
        b.detach_()
        w_.detach_()
#        del a
#        del b
        # print(i)
        self.weights[torch.tensor(i).type(torch.long)] = w.view(1, -1).detach()
        if self.net_type == 'graph':
            return [a,]
        if self.net_type in ['vision', 'text']:
            return [[a, b, i]]


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        return torch.mean(self.criterion(outputs, labels) * self.weights[indices])



class DDSTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed,
                 catnum, epsilon, lr, net_type):
        
        cl = DDS(catnum, epsilon, lr, net_type)

        super(DDSTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)