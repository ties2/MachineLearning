import csv
import numpy as np
from typing import Set,Tuple, List
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision
NoneType = type(None)
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import vgg11
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
import time

"""
    This method returns the fruit name by getting the string at a specific index of the set.

    :param fruit_id: The id of the fruit to get
    :param fruits: The set of fruits to choose the id from
    :return: The string corrosponding to the index ``fruit_id``

    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   It does not print the fruit at the correct index, why is the returned result wrong?``
    | ``2   How could this be fixed?``

    This example demonstrates the issue:
    name1, name3 and name4 are expected to correspond to the strings at the indices 1, 3, and 4:
    'orange', 'kiwi' and 'strawberry'..
        **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains a bug that needs to be found.**
    try:
        # Convert set to a list to ensure consistent ordering
        sorted_fruits = sorted(list(fruits))
        # Return the fruit at the given index
        return sorted_fruits[fruit_id]
    except IndexError:
        # Handle cases where fruit_id is out of range
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

    >>> name1 = id_to_fruit(1, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name3 = id_to_fruit(3, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name4 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})
    """
def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    idx = 0
    for fruit in fruits:
        # print(fruit_id, idx, fruit)
        if fruit_id == idx:
            return fruit
        idx += 1
    raise RuntimeError(f"Fruit with id {fruit_id} does not exist")
example_list = ["apple", "orange", "melon", "kiwi", "strawberry"]
name1 = id_to_fruit(1, example_list)
name3 = id_to_fruit(3, example_list)
name4 = id_to_fruit(4, example_list)
print(name1, name3, name4)


#it used dictionary/set instead of list(using []) and sorting option for set and dictionary is not the same as list  
# The issue with the code is that sets are unordered collections of unique elements, 
# meaning that the order of elements in a set is not guaranteed. 
# To fix the issue, modify the function to take a list instead of a set as input. 
# Lists are ordered collections of elements, meaning that the order of elements is preserved