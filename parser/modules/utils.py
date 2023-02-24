import torch


def dim_dropout(input, p, dim=-1, training=False):
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))
    if training:
        size = input.shape[dim]

        mask = torch.rand(size, device=input.device) > p
        shape = [None] * input.dim()
        shape[dim] = slice(None)
        mask = mask[shape]
        mask = mask.expand_as(input)

        return mask * input / (1. - p)
    return input

if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    print(x)
    print(dim_dropout(x, 0.5, dim=1, training=True))
    print(dim_dropout(x, 0.5, dim=1, training=False))
    print(dim_dropout(x, 0.5, dim=2, training=True))
    print(dim_dropout(x, 0.5, dim=2, training=False))