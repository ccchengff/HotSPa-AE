from .utils import Args

class ModelWrapper(Args):
    def __init__(self, model_class, model_config):
        self.model_class = model_class
        self.model_config = model_config

    def create_model(self, ds_parallel_configs):
        return self.model_class(config=self.model_config, ds_parallel_configs=ds_parallel_configs)
    
class OptimizerWrapper(Args):
    def __init__(self, optimizer_class):
        self.optimizer_class = optimizer_class

    def create_optimizer(self, **kwargs):
        return self.optimizer_class(**kwargs)
    
class DatasetWrapper(Args):
    def __init__(self, dataset_class):
        self.dataset_class = dataset_class
        
    def create_dataset(self, **kwargs):
        return self.dataset_class(**kwargs)
