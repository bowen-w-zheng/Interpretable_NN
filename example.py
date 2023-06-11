"""
This is an example for how to run factorization algorithms. 
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import sys
import random
import matplotlib.pyplot as plt
# create a dataset for testing

n_samples = 1000
n_features = 10
n_factors = 5
n_response_patterns = 100


# feature matrix 
F = np.random.rand(n_features, n_samples)
# mixing matrix, sparse matrix
rows = n_factors
cols = n_features
M = np.zeros((rows, cols))

# Create a set of all column indices
cols_set = list(range(cols))

for i in range(rows):
    # Choose a random number of non-zero columns for this row
    nonzero_cols = min(random.randint(1, cols//3), len(cols_set))
    # Randomly choose which columns will be non-zero from remaining set
    cols_to_populate = random.sample(cols_set, nonzero_cols)
    # Generate random probabilities that sum up to 1
    probs = np.random.dirichlet(np.ones(nonzero_cols), size=1)[0]
    # Populate the chosen columns with these probabilities
    for col_index, prob in zip(cols_to_populate, probs):
        M[i][col_index] = prob
    # Remove selected columns from the set
    for col in cols_to_populate:
        cols_set.remove(col)

# Display the matrix as an image
plt.imshow(M, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('mixing matrix: M')
plt.show()


# response matrix, for each factor, there is a pattern of response associated with it so there should be n_factor response patterns, each pattern is a vector of length n_response_pattern
# note that we expect the response matrix to be sparse 
R = np.zeros((n_factors, n_response_patterns))

for i in range(n_factors):
    # decide the number of non-zero responses for this factor, here we randomly decide it to be between 1 and half the length of the response pattern
    non_zero_responses = random.randint(1, n_response_patterns//5)
    # choose which responses will be non-zero
    responses_to_populate = random.sample(range(n_response_patterns), non_zero_responses)
    # generate random probabilities that sum up to 1 for the non-zero responses
    probs = np.random.dirichlet(np.ones(non_zero_responses), size=1)[0]
    # populate the chosen responses with these probabilities
    for response_index, prob in zip(responses_to_populate, probs):
        R[i][response_index] = prob
R = R.T

# Display the response matrix as an image
plt.imshow(R, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('response matrix: R')
plt.show()

X_true = np.dot(R, np.dot(M,F))


# run NMF


# run feature_based matrix factorization

# run network-based matrix factorization
