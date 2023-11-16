import pytest
import torch
import pdb

from s01_pytorch.t01_tensors.tensors import *


def test_challenge_01():
    """Test that challenge_1 returns a 3D tensor of shape (2, 2, 2)
    containing the numbers from 0 to 7 in order"""

    result = challenge_01()

    assert result.shape == (2, 2, 2)
    assert torch.allclose(result, torch.as_tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]))


def test_challenge_02():
    """Test that challenge_2 returns a tensor of shape (4,)
    containing elements from the diagonal of its input tensor"""

    result = challenge_02()
    assert result.shape == (4,)
    assert torch.allclose(result, torch.as_tensor([0, 3, 12, 15]))


def test_challenge_03():
    """Test that challenge_3 returns a 2D tensor of shape (5, 3)
    containing the numbers 4 to 18 in order"""

    result = challenge_03()
    assert result.shape == (5, 3)
    assert torch.allclose(
        result,
        torch.as_tensor(
            [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
        ),
    )


def test_challenge_04():
    """Test that challenge_4 returns a tensor of shape (2, 7)
    containing elements from both inputs"""

    result = challenge_04()
    assert result.shape == (2, 7)
    assert torch.allclose(
        result, torch.as_tensor([[0, 1, 2, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    )


def test_challenge_05():
    """Test that challenge_5 returns a 3D tensor of shape (2, 2, 2)
    containing the original tensor and a new one"""

    result = challenge_05()
    assert result.shape == (2, 2, 2)
    assert torch.allclose(
        result, torch.as_tensor([[[0, 1], [2, 3]], [[0, 0], [0, 0]], [[4, 5], [6, 7]]])
    )


def test_challenge_06():
    """Test that challenge_6 replaces all odd numbers with 0 without
    altering the original shape"""

    result = challenge_06()
    assert torch.allclose(result, torch.as_tensor([[[0, 2], [4, 0]], [[0, 6], [8, 0]]]))


def test_challenge_07():
    """Test that challenge_7 returns the number of positive elements
    in a tensor"""

    result = challenge_07()
    assert isinstance(result, int)
    assert result == 12


def test_challenge_08():
    """Test that challenge_8 returns a 2D tensor of shape (10, 10)
    filled with random values between 0 and 255"""

    result = challenge_08()
    assert result.shape == (10, 10)
    assert result.min() >= 0 and result.max() <= 255


def test_challenge_9():
    """Test that challenge_9 returns a tensor of shape (2, 3, 2, 2)
    filled with random values between 0 and 1"""

    result = challenge_9()
    assert result.shape == (2, 3, 2, 2)
    assert result.min() >= 0 and result.max() <= 1


def test_challenge_10():
    """Test that challenge_10 returns a 3-dimensional tensor of shape
    (3, 2, 3) multiplied by a scalar of value 5"""

    result = challenge_10()
    assert result.shape == (3, 2, 3)
    assert torch.allclose(result, 5 * torch.ones((3, 2, 3)))


def test_challenge_11():
    """Test that challenge_11 returns a new tensor that has been
    concatenated along its rows"""

    result = challenge_11()
    assert result.shape == (4, 6)


def test_challenge_12():
    """Test that challenge_12 returns a tensor whose dimensions have
    been inverted"""

    result = challenge_12()
    assert result.shape == (3, 2, 5)


def test_challenge_13():
    """Test that challenge_13 adds a random tensor of shape (2, 2)
    to a tensor filled with ones of the same shape"""

    result = challenge_13()
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.ones((2, 2)) + torch.rand((2, 2)))


@pytest.mark.parametrize("shape1, shape2", [(2, 5), (5, 2)])
def test_challenge_14(shape1, shape2):
    a = torch.rand(shape1)
    b = torch.rand(shape2)

    assert torch.sub(a, b).reshape(2, 10).shape == (2, 10)


@pytest.mark.parametrize("shape", [(5, 2, 6)])
def test_challenge_15(shape):
    a = torch.rand(shape)

    assert len(torch.split(a, 4)) == 4


@pytest.mark.parametrize("shape", [(4, 2, 2)])
def test_challenge_16(shape):
    a = torch.rand(shape)

    assert type(a.mean(dim=0)) == torch.Tensor


@pytest.mark.parametrize("shape", [(3, 2)])
def test_challenge_17(shape):
    a = torch.rand(shape)
    b = torch.rand(shape)

    assert type(torch.dot(a, b)) == float


@pytest.mark.parametrize("shape", [(2, 2, 2, 2)])
def test_challenge_18(shape):
    a = torch.rand(shape)
    idx = 0
    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            for k in range(0, len(a[i][j])):
                for l in range(0, len(a[i][j][k])):
                    a[i][j][k][l] = a[i][j][k][l] * idx
                    idx += 1

    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            for k in range(0, len(a[i][j])):
                for l in range(0, len(a[i][j][k])):
                    assert type(a[i][j][k][l] == float) or type(a[i][j][k][l] == int)


@pytest.mark.parametrize("shape", [(4, 2, 2)])
def test_challenge_19(shape):
    a = torch.rand(shape)

    assert torch.argmax(a).shape == (4, 2)


@pytest.mark.parametrize("shape", [(2, 2, 3)])
def test_challenge_20(shape):
    a = torch.arange(12).reshape(shape)

    condition = lambda x: x % 2 == 0
    new = a[torch.where(condition(a))]

    assert len(new.nonzero()) == 6
