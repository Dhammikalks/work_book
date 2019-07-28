#!/bin/python -f
# Quiz Solution
# Note: You can't run code in this tab
import numpy as np
import tensorflow as ft

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return tf.nn.softmax([2.0, 1.0, 0.2])

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
