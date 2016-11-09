# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk.device import cpu, set_default_device
from cntk import Trainer
from cntk.learner import sgd
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sigmoid
from cntk.utils import ProgressPrinter

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import linear_layer

# make sure we get always the same "randomness"
np.random.seed(0)

def generate_random_data(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
    X = X.astype(np.float32)
    # converting class 0 into the vector "1 0 0",
    # class 1 into vector "0 1 0", ...
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

# Creates and trains a feedforward classification model

def log_reg():
    input_dim = 2
    num_output_classes = 2

    # Input variables denoting the features and label data
    input = input_variable((input_dim), np.float32)
    label = input_variable((num_output_classes), np.float32)

    # Instantiate the feedforward classification model
    output_dim = num_output_classes
    netout = linear_layer(input, output_dim)

    #loss = cross_entropy_with_softmax(z, label)
    ce = cross_entropy_with_softmax(netout, label)
    
    pe = classification_error(netout, label)

    # Instantiate the trainer object to drive the model training
    trainer = Trainer(netout, ce, pe, [sgd(netout.parameters, lr=0.02)])

    # Get minibatches of training data and perform model training
    minibatch_size = 64

    pp = ProgressPrinter(256)
    for i in range(2048):
        features, labels = generate_random_data(
            minibatch_size, input_dim, num_output_classes)
        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        trainer.train_minibatch({input: features, label: labels})
        pp.update_with_trainer(trainer)
    pp.epoch_summary()
    test_features, test_labels = generate_random_data(
        minibatch_size, input_dim, num_output_classes)
    avg_error = trainer.test_minibatch(
        {input: test_features, label: test_labels})
    return avg_error

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # set_default_device(cpu())

    error = log_reg()
    print(" error rate on an unseen minibatch %f" % error)
