"""
ChatGPT Request:
Create 20 difficult python challenges to teach me the pytorch api.
It should focus on tensor manipulation like create tensors, manipulate their shape,
and perform basic operations like addition, subtraction, and multiplication.
These should be diffcult and require more than one line of code. I want each challenge
to be a function, complete with docstrings to describe the desired coding challenge. 
Each function should start with return None and I'll update the code in it myself. Follow 
pep8 guidelines and keep strings < 100 chars per line.
"""

import torch


def challenge_01():
    """Create a 3D tensor of shape (2, 2, 2) containing
    the numbers from 0 to 7 in order."""
    return torch.arange(8).view(2, 2, 2)


def challenge_02():
    """Given a tensor of shape (4, 2, 2) extract elements
    on the diagonal and create a new tensor of shape (4,)."""
    return torch.rand(4, 2, 2).diagonal().reshape(-1)


def challenge_03():
    """Create a 2D tensor of shape(5, 3) out of a 1D tensor
    containing 15 consecutive numbers starting from 4."""
    return torch.arange(4, 19).reshape(5, 3)


def challenge_04(a, b):
    """Combine two tensors of shapes (2, 3) and (2, 4)
    into one tensor of shape (2, 7)."""
    return torch.concatenate(a, b, dim=1)


def challenge_05(a, b):
    """Given a 3D tensor of shape (2, 2, 2) insert another
    3D tensor of the same shape between the first two slices."""
    return torch.concatenate(a[:, :0], b, a[:, :, 1])


def challenge_06():
    """Replace all odd numbers in a tensor with 0 without
    altering the original shape."""
    return None


def challenge_07():
    """Return the number of positive elements in a tensor"""
    return None


def challenge_08():
    """Create a 2D tensor of shape (10, 10) filled with random
    values between 0 and 255."""
    return None


def challenge_09():
    """Create a tensor of shape (2, 3, 2, 2) filled with random
    values between 0 and 1."""
    return None


def challenge_10():
    """Create a 3-dimensional tensor of shape (3, 2, 3) and multiply
    it by a scalar of value 5."""
    return None


def challenge_11():
    """Concatenate two tensors along their rows to create a new tensor."""
    return None


def challenge_12():
    """Invert the ordering of the dimensions of a tensor."""
    return None


def challenge_13():
    """Create a tensor of shape (2, 2) filled with random values
    between 0 and 1 and add it to a tensor filled with ones of
    the same shape."""
    return None


def challenge_14():
    """Create a tensor of shape (2, 5) and subtract it from a
    tensor of shape (5, 2) and reshape it to a (2, 10) tensor."""
    return None


def challenge_15():
    """Split a tensor of shape (5, 2, 6) into four equal chunks."""
    return None


def challenge_16():
    """Compute the mean of a tensor along its first dimension."""
    return None


def challenge_17():
    """Create two tensors with the same shape filled with random
    values between 0 and 1 and compute their dot product."""
    return None


def challenge_18():
    """Given a tensor with shape (2, 2, 2, 2) multiply each
    element with its corresponding index."""
    return None


def challenge_19():
    """Return the indices of the maximum values in a tensor
    of shape (4, 2, 2)."""
    return None


def challenge_20():
    """Select elements from a tensor based on a condition
    and create a new tensor from them."""
    return None
