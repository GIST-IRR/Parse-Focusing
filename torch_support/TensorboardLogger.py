from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir=None) -> None:
        self.writer = SummaryWriter(log_dir)
        self._metric_list = {}
        self._metric_dict = {}
    
    def write_list(self, epoch, tag=None):
        for k, v in self._metric_list.items():
            self.writer.add_scalar(f"{tag}/{k}", v, epoch)

    def write_dict(self, tag=None):
        for k, v in self._metric_dict.items():
            for e, val in v.items():
                self.writer.add_scalar(f"{tag}/{k}", val, e)