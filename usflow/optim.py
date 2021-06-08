import torch


def load_optimizer(optim_type, parameters, lr=1e-4, momentum=0.9, weight_decay=1e-4):
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(
            params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(
            params=parameters, lr=lr, weight_decay=weight_decay
        )
    elif optim_type == "asgd":
        optimizer = torch.optim.ASGD(
            params=parameters, lr=lr, weight_decay=weight_decay
        )
    elif optim_type == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        assert False, "Unknown optimizer type: " + optim_type
    return optimizer
