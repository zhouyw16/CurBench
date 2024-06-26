import math
import torch
from torch.utils.data import Subset
from torch_geometric.data.batch import Batch as pygBatch

from .base import BaseTrainer, BaseCL



class SPL(BaseCL):
    """Self-Paced Learning. 

    Self-paced learning for latent variable models. https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf
    A Survey on Curriculum Learning. https://arxiv.org/pdf/2010.13166.pdf

    Attributes:
        name, dataset, data_size, batch_size, n_batches: Base class attributes.
        epoch: An integer count of current training epoch.
        start_rate: An float indicating the initial proportion of the sampled data instances.
        grow_epochs: An integer for the epoch when the proportion of sampled data reaches 1.0.
        grow_fn: Pacing function or Competence function that how the proportion of sampled data grows.
        net: The network itself for calculating the loss.
        device: The used cuda (currently only support single cuda training).
        criterion: The loss function.
        weights: The weights of all training data instances.
    """
    def __init__(self, start_rate, grow_epochs, grow_fn, weight_fn):
        super(SPL, self).__init__()

        self.name = 'spl'
        self.epoch = 0
        self.teacher_net = None

        self.start_rate = start_rate
        self.grow_epochs = grow_epochs
        self.grow_fn = grow_fn
        self.weight_fn = weight_fn


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        if self.teacher_net is None:                            # In Self-Paced Learning, the network is itself.
            self.teacher_net = net                              # In Transfer Teacher, the network is teacher net.
        self.teacher_net.to(self.device)


    def data_curriculum(self, **kwargs):
        self.epoch += 1

        data_rate = min(1.0, self._subset_grow())               # Current proportion of sampled data.
        data_size = int(math.ceil(self.data_size * data_rate))  # Current number of sampled data.
        data_loss = self._loss_measure()                        # Calculate loss as the measurement of difficulty. 
        loss_topk = data_loss.topk(k=data_size, largest=False, sorted=False)
        data_indices = loss_topk.indices.tolist()               # Sample the easist data according to the loss value.
        loss_threshold = loss_topk.values.max()                 # Derive the loss of the hardest data instance.

        if self.weight_fn == 'hard':                            # Data Sampling (hard selection).
            dataset = Subset(self.dataset, data_indices)
        else:                                                   # Data Reweighting (soft selection).
            dataset = self.dataset
            self.weights = self._data_weight(data_loss, loss_threshold)
        return self._dataloader(dataset)


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        if self.weight_fn == 'hard':
            return torch.mean(self.criterion(outputs, labels))
        else:
            return torch.mean(self.criterion(outputs, labels) * self.weights[indices])


    def _subset_grow(self):
        if self.grow_fn == 'linear':                            # Linear Function.
            return self.start_rate + (1.0 - self.start_rate) / self.grow_epochs * self.epoch
        elif self.grow_fn == 'geom':                            # Geometric Function.
            return 2.0 ** ((math.log2(1.0) - math.log2(self.start_rate)) / self.grow_epochs * self.epoch + math.log2(self.start_rate))
        elif self.grow_fn[:5] == 'root-' and self.grow_fn[5:].isnumeric():
            p = int(self.grow_fn[5:])                           # Root-p Function.
            return (self.start_rate ** p + (1.0 - self.start_rate ** p) / self.grow_epochs * self.epoch) ** 0.5
        else:
            raise NotImplementedError()


    def _loss_measure(self):
        loss = []
        self.teacher_net.eval()
        with torch.no_grad():
            for data in self._dataloader(self.dataset, shuffle=False):
                if isinstance(data, list):                      # data from torch.utils.data.Dataset
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)
                    outputs = self.teacher_net(inputs)
                elif isinstance(data, dict):                    # data from datasets.arrow_dataset.Dataset
                    inputs = {k: v.to(self.device) for k, v in data.items() 
                              if k not in ['labels', 'indices']}
                    labels = data['labels'].long().to(self.device)
                    outputs = self.teacher_net(**inputs)[0]
                elif isinstance(data, pygBatch):                # data from torch_geometric.datasets
                    inputs = data.to(self.device)
                    labels = data.y.view(-1).to(self.device)
                    outputs = self.teacher_net(inputs)
                else:
                    not NotImplementedError()
                loss.append(self.criterion(outputs, labels).detach())
        return torch.concat(loss)


    def _data_weight(self, loss, threshold):
        mask = loss < threshold                                 # The weight of data whose loss greater than threshold is zero.
        if self.weight_fn == 'linear':
            return mask * (1.0 - loss / threshold)
        elif self.weight_fn == 'logarithmic':
            return mask * (torch.log(loss + 1.0 - threshold) / torch.log(1.0 - threshold))
        elif self.weight_fn == 'logistic':
            return (1.0 + torch.exp(-threshold)) / (1.0 + torch.exp(loss - threshold))
        elif self.weight_fn[:11] == 'polynomial-' and self.weight_fn[11:].isnumeric():
            t = int(self.weight_fn[11:])
            return mask * ((1.0 - loss / threshold) ** 1.0 / (t - 1.0))      
        else:
            raise NotImplementedError()


class SPLTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed, 
                 start_rate, grow_epochs, grow_fn, weight_fn):
        
        cl = SPL(start_rate, grow_epochs, grow_fn, weight_fn)

        super(SPLTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)