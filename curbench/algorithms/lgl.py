import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from .base import BaseTrainer, BaseCL



class LGL(BaseCL):
    """Local to Global Learning.
    
    Local to Global Learning: Gradually Adding Classes for Training Deep Neural Networks
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_Local_to_Global_Learning_Gradually_Adding_Classes_for_Training_Deep_CVPR_2019_paper.pdf
    """
    def __init__(self, start_rate, grow_rate, grow_interval, strategy):
        super(LGL, self).__init__()

        self.name = 'lgl'
        self.epoch = 0
        self.classes = np.array([], dtype=int)

        self.start_rate = start_rate
        self.grow_rate = grow_rate
        self.grow_interval = grow_interval
        self.strategy = strategy


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler, **kwargs):
        super().model_prepare(net, device, epochs, criterion, optimizer, lr_scheduler)
        self.class_size = self.net.num_classes
        self.init_scheduler_state = self.lr_scheduler.state_dict()
        # TODO: update T_max of lr_scheduler

        self.class_indices = [[] for _ in range(self.class_size)]
        for _, label, index in self.dataset:
            self.class_indices[label].append(index)


    def data_curriculum(self, **kwargs):
        self.epoch += 1

        class_rate = min(1.0, self._subclass_grow())
        class_size = int(math.ceil(self.class_size * class_rate))
        if self.classes.shape[0] < class_size:
            if self.classes.shape[0] == 0:
                classes_select = np.random.choice(
                    self.class_size, size=class_size, replace=False
                )
            else:
                classes_remain = np.delete(np.arange(self.class_size), self.classes)
                if self.strategy == 'random':
                    classes_select = np.random.choice(classes_remain, 
                        size=class_size - self.classes.shape[0], replace=False
                    )
                elif self.strategy[-7:] == 'similar':
                    similaries = []
                    for class_remain in classes_remain:
                        similaries.append(self._similarity_measure(class_remain))
                    args_sorted = np.argsort(similaries)
                    if self.strategy == 'similar':
                        args_select = args_sorted[:class_size - self.classes.shape[0]]
                    else:
                        args_select = args_sorted[self.classes.shape[0] - class_size:]
                    classes_select = classes_remain[args_select]
                else:
                    raise NotImplementedError()
            self.classes = np.append(self.classes, classes_select)
            self.label_dict = {label: index for index, label in enumerate(self.classes)}
            self.label_indices = []
            for label in self.classes:
                self.label_indices.extend(self.class_indices[label])

        dataset = Subset(self.dataset, self.label_indices)
        return self._dataloader(dataset)


    def model_curriculum(self, **kwargs):
        if (self.epoch - 1) % self.grow_interval == 0:
            self.lr_scheduler.load_state_dict(self.init_scheduler_state)
        return self.net


    def loss_curriculum(self, outputs, labels, indices, **kwargs):
        return torch.mean(self.criterion(outputs[:, self.classes], self._label_map(labels)))

    
    def _subclass_grow(self):
        return self.start_rate + self.grow_rate * ((self.epoch - 1) // self.grow_interval + 1)


    def _similarity_measure(self, label):
        dataset = Subset(self.dataset, self.class_indices[label])
        loader = self._dataloader(dataset)
        entropy = 0.0
        for data in loader:
            outputs = self.net(data[0].to(self.device))
            logits = F.softmax(outputs[:, self.classes], dim=1)
            entropy += torch.sum(-logits * torch.log(logits)).item()
        return entropy / len(dataset)


    def _label_map(self, labels):
        return torch.LongTensor([self.label_dict[label.item()] for label in labels]) \
                    .to(self.device)



class LGLTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, gpu_index, num_epochs, random_seed,
                 start_rate, grow_rate, grow_interval, strategy):
        
        cl = LGL(start_rate, grow_rate, grow_interval, strategy)

        super(LGLTrainer, self).__init__(
            data_name, net_name, gpu_index, num_epochs, random_seed, cl)