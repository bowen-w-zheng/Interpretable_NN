"""
Given two matrices, we want to match the the columns from one to another by computing correlation between columns
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import sys
import time

def match_columns(A, B, reps = 4):
    col_A = A.shape[1]
    col_B = B.shape[1]
    # compute the correlation matrix
    corr = np.zeros((col_A, col_B))

    for i in range(col_A):
        for j in range(col_B):
            corr[i][j] = np.corrcoef(A[:,i], B[:,j])[0][1]
    # find the best match for each column in A, if a column in B is matched, it cannot be matched again
    # permute columns of A for rep times and find the best match to get the consensus, avoiding effect of processing orders
    match = np.zeros((reps, col_A))
    for rep in range(reps):
        corr_copy = corr.copy()
        permutate_order = np.random.permutation(col_A)
        for k in permutate_order:
            match[rep, k] = np.argmax(corr_copy[k])
            # set the correlation to -1 so that it won't be matched again
            corr_copy[:,int(match[rep, k])] = -2
    return match

# test
def main():
    A = np.random.rand(4, 5)
    # duplicate A's first column 
    A = np.append(A, A[:,0].reshape(4,1), axis = 1)
    # B is permutation of A along the column dimension 
    B = A[:, np.random.permutation(5)]
    # append the first column of A to the end of B
    B = np.append(B, A[:,0].reshape(4,1), axis = 1)
    match = match_columns(A, B, reps = 10)
    print("Expecting the first and last column of A to be matched with two columns of B, and others to be matched with one column of B")
    print(match)

if __name__ == '__main__':
    main()




