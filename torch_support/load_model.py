import inspect
import parser.model

try:
    import torch_optimizer as optim
except:
    try:
        import torch.optim as optim
    except:
        raise ImportError("Could not import torch.optim or torch_optimizer")

def get_model(model_name, args=None, device='cpu'):
    # Get the module of model from the path
    # model_modules = {
    #     name: obj for name, obj in inspect.getmembers(parser.model)
    #     if inspect.ismodule(obj)
    # }
    # try:
    #     module = model_modules[model_name]
    # except:
    #     raise KeyError("Model name not found in model directory")

    # Please match the python filename and the model class name
    # Get the class of model from the module
    models = {
        name: obj for name, obj in inspect.getmembers(parser.model)
        if inspect.isclass(obj)
    }
    try:
        model = models[model_name]
    except:
        raise KeyError("Model name not found in model module")
    
    return model(args).to(device)

def get_model_args(args, device):
    # get model name
    name = args.name
    # get model kwargs
    args.pop("name")
    # get model
    model = get_model(name, args, device)
    return model

def get_optimizer(optimizer_name, params, kwargs):
    # supported optimizers:
    # ASGD, Adadelta, Adagrad, Adam, AdamW, Adamax, LBFGS, NAdam, RAdam,
    # RMSprop, Rprop, SGD, SparseAdam
    optim_modules = {
        name: obj for name, obj in inspect.getmembers(optim)
        if inspect.isclass(obj)
    }
    try:
        optimizer = optim_modules[optimizer_name]
    except:
        raise KeyError("Optimizer name not found in torch.optim")

    return optimizer(params=params, **kwargs)

def get_optimizer_args(args, params):
    # get optimizer name
    name = args.name
    # get optimizer kwargs
    args.pop("name")
    # get optimizer
    optimizer = get_optimizer(name, params, args)
    return optimizer