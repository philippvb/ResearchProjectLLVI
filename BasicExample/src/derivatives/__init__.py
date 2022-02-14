import torch
from torch.autograd import grad

def nth_derivative(f, wrt, n):
    """Computes the diagonal fo the n-th derivative of f with respect to wrt

    Args:
        f ([type]): The functio
        wrt ([type]): weights to compute gradient for
        n ([type]): order of the gradient

    Returns:
        [type]: gradient for the weights
    """

    for i in range(n):

        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()

    return grads


def gradient_magnitude(f, wrt):
    grads = grad(f, wrt, create_graph=True)[0]
    return torch.square(grads)
    # TODO: take g.T g


