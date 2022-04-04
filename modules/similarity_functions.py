import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

def cal_cosine_similarity(term_matrix):
    return cosine_similarity(term_matrix, term_matrix)

def cal_jaccard_score(term_matrix):
    return jaccard_score(term_matrix, term_matrix)

def cal_person_score(term_matrix):
    return np.corrcoef(term_matrix, term_matrix)