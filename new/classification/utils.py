import torch

# These two functions can be useful in deep learning,
# for example, to monitor the gradient and parameter norms during training,
# which can help detect issues such as gradient explosion or vanishing gradients.

def get_grad_norm(parameters, norm_type=2):
    # Defines a function called get_grad_norm that takes two arguments: parameters,
    # which is an iterable containing models parameters (e.g., tensors),
    # and norm_type, which is the type of norm to compute (default is 2, corresponding to the L2 norm).

    # Filters out parameters that do not have gradients (i.e., p.grad is None).
    # This is done using a lambda function within the filter() function, which is then converted to a list.
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    # Initializes a variable total_norm to store the sum of the norms of gradients.
    total_norm = 0

    # Begins a try-except block to handle any exceptions that might occur during the computation.
    # For each parameter in the filtered list, calculate the norm of its gradient using the specified norm type,
    # raise it to the power of the norm type, and add it to total_norm.
    # After the loop, take the root of total_norm using the inverse of the norm type.
    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def get_parameter_norm(parameters, norm_type=2):
    # Defines a function called get_parameter_norm that takes the same two arguments as get_grad_norm: parameters and norm_type.

    # Initializes a variable total_norm to store the sum of the norms of the parameters.
    total_norm = 0

    # For each parameter in the list, calculate the norm using the specified norm type,
    # raise it to the power of the norm type, and add it to total_norm.
    # After the loop, take the root of total_norm using the inverse of the norm type.
    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

