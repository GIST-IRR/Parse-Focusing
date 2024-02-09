import sys
import inspect
import importlib
from pathlib import Path

try:
    import torch_optimizer as optim
except:
    try:
        import torch.optim as optim
    except:
        raise ImportError("Could not import torch.optim or torch_optimizer")


model_dir_path = "model"


def set_model_dir(path):
    global model_dir_path
    assert isinstance(path, str), "Path should be a string"
    # if path is consisted of /, replace it with .
    model_dir_path = path.replace("/", ".")
    return model_dir_path


def get_model(model_name, args=None, device="cpu"):
    """
    Description:

    """
    # Please match the python filename and the model class name
    # Get the class of model from the module
    # model_path = str(model_dir_path / (model_name+'.py'))
    model_path = model_dir_path + "." + model_name

    try:
        module = importlib.import_module(model_path)
        sys.modules[model_name] = module
        model = module.__dict__[model_name]

    except:
        raise KeyError("Model name not found in model module")

    return model(args).to(device)


def get_model_args(args, device, state_dict=None):
    # get model name
    name = args.name
    # get model
    model = get_model(name, args, device)

    # Load statd dict if is not None
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def get_optimizer(optimizer_name, params, kwargs):
    # supported optimizers:
    # ASGD, Adadelta, Adagrad, Adam, AdamW, Adamax, LBFGS, NAdam, RAdam,
    # RMSprop, Rprop, SGD, SparseAdam
    optim_modules = {
        name: obj
        for name, obj in inspect.getmembers(optim)
        if inspect.isclass(obj)
    }
    try:
        optimizer = optim_modules[optimizer_name]
    except:
        raise KeyError("Optimizer name not found in torch.optim")

    return optimizer(params=params, **kwargs)


def get_optimizer_args(args, params, state_dict=None):
    # get optimizer name
    name = args.name
    # get optimizer kwargs
    args.pop("name")
    # get optimizer
    optimizer = get_optimizer(name, params, args)

    # Load statd dict if is not None
    if state_dict is not None:
        optimizer.load_state_dict(state_dict)

    return optimizer
