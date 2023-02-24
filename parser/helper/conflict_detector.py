import torch
from copy import deepcopy

from parser.helper.loader_wrapper import DataPrefetcher
from .data_module import ByLengthSampler
from fastNLP.core.batch import DataSetIter


class ConflictDetector:
    """Detect conflicts in the dataset

    Args:
        dataset (torch.utils.data.Dataset): dataset
        model (nn.Module): model
        index (int): index of the dataset. A sentence to check conflicts
    """
    def __init__(self, dataset, model, optimizer):
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.prior = None
        self.posterior = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = deepcopy(optimizer)

    def set_dataloader(self, x):
        sentence = x["word"].tolist()
        dataset = self.dataset.drop(lambda x: x["word"] in sentence)
        sampler = ByLengthSampler(dataset, batch_size=8)
        dataloader = DataSetIter(dataset=dataset, batch_sampler=sampler)
        self.dataloader = DataPrefetcher(dataloader, device=self.model.device)

    def calculate_prior(self, x):
        self.sentence = x
        self.set_dataloader(x)
        partition_function = self.partition_function()[0]
        dataset_probability = self.dataset_probability()
        sentence_probability = self.sentence_probability(x)
        probs = {
            'partition_function': partition_function,
            'dataset_probability': dataset_probability,
            'sentence_probability': sentence_probability,
        }
        self.prior = probs

    def calculate_posterior(self):
        self.set_dataloader(self.sentence)
        partition_function = self.partition_function()[0]
        dataset_probability = self.dataset_probability()
        sentence_probability = self.sentence_probability(self.sentence)
        probs = {
            'partition_function': partition_function,
            'dataset_probability': dataset_probability,
            'sentence_probability': sentence_probability,
        }
        if self.prior is not None:
            # log-scale
            # if probability is increased,
            # then the log-probability approaches to 0
            # positive value: increase of probability
            # negative value: decrease of probability
            self.difference = {
                k: probs[k] - v for k, v in self.prior.items()
            }
        self.posterior = probs

    @torch.no_grad()
    def partition_function(self):
        self.model.eval()
        return self.model.partition_function()
    
    @torch.no_grad()
    def dataset_probability(self):
        self.model.eval()
        dataset_probability = []
        for x, _ in self.dataloader:
            output = -self.model.loss(x)
            output = output.logsumexp(-1)
            dataset_probability.append(output)
        dataset_probability = torch.stack(dataset_probability).logsumexp(-1)
        return dataset_probability

    @torch.no_grad()
    def sentence_probability(self, x):
        self.model.eval()
        sentence_probability = -self.model.loss(x)
        sentence_probability = sentence_probability.logsumexp(-1)
        return sentence_probability