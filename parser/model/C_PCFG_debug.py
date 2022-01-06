import torch
from ..model import CompoundPCFG

class CompoundPCFG_D(CompoundPCFG):
    def __init__(self, args, dataset):
        super(CompoundPCFG_D, self).__init__(args, dataset)

    def loss(self, input):
        rules = self.forward(input)
        result =  self.pcfg._inside(rules=rules, lens=input['seq_len'])
        # Partition function
        if self.depth > 0:
            pf = self.pcfg._partition_function(rules=rules, depth=self.depth)
            result['partition'] = result['partition'] - pf

        loss =  (-result['partition'] + rules['kl']).mean()
        return rules, loss

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input, evaluating=True)
        if decode_type == 'viterbi':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=True, mbr=False)
        elif decode_type == 'mbr':
            result = self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError

        result['partition'] -= rules['kl']
        result['rules'] = rules
        return result