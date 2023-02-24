class MetricDict:
    def __init__(self, metrics) -> None:
        if not isinstance(metrics, dict):
            raise TypeError('MetricDict only container a dict.')
        self.metrics = list(metrics)

    def add_metric(self, metric_name, value):
        if not isinstance(value, (function, str)):
            raise TypeError(
                f'{metric_name} must be a function or a string, but got {type(value)}'
            )
        self.metrics.update({metric_name, value})

    def update(self):
        for k, v in self.metrics.items():
            if isinstance(v, str):
                pass
            elif isinstance(v, function):
                result = v()
            else:
                raise TypeError(f'{k} must be a function or a string, but got {type(v)}')

    def _save(self, key):
        pass

    def save_to_tensorboard(self):
        pass