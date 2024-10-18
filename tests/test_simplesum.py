import pytest

from tmrc.tmrc_core.utils.simple_sum import SimpleSum  

def test_simple_sum():
    # Test case 1: Normal case
    numbers = [1, 2, 3, 4]
    simple_sum = SimpleSum(numbers)
    assert simple_sum.sum() == 10

    # Test case 2: Empty list
    numbers = []
    simple_sum = SimpleSum(numbers)
    assert simple_sum.sum() == 0

    # Test case 3: Negative numbers
    numbers = [-1, -2, -3, -4]
    simple_sum = SimpleSum(numbers)
    assert simple_sum.sum() == -10

    # Test case 4: Mixed positive and negative numbers
    numbers = [-1, 2, -3, 4]
    simple_sum = SimpleSum(numbers)
    assert simple_sum.sum() == 2

    # Test case 5: Single number
    numbers = [5, 10]
    simple_sum = SimpleSum(numbers)
    assert simple_sum.sum() == 15
