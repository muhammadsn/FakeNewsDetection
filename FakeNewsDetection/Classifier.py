import numpy as np
import pandas as pd
from scipy import sparse



class Classifier:

    def __init__(self, train_sparse, labels, test_sparse):
        _train = sparse.csr_matrix(train_sparse)
        _test = sparse.csr_matrix(test_sparse)

        print(_test)