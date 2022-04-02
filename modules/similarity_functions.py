import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_soft_cossim_matrix(sentence_matrix):
    term_matrix = sentence_matrix.todense()
    return cosine_similarity(term_matrix, term_matrix)