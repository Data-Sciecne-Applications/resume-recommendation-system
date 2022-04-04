import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans 
from kneed import KneeLocator
import plotly.graph_objects as go
from tkinter.tix import Tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


def train_val_test_split(x, y, train_size, val_size, test_size, random_state=41):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_size / (test_size + val_size), random_state=random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test

def tunning(model, vectorizer, crit, cv, x, y):
    Grid = GridSearchCV(model, crit, cv=cv)
    Grid.fit(vectorizer.fit_transform(x), y)
    return Grid.best_estimator_

def save_tfidf(path, tfidf_vec):
    with open(path, 'wb') as fw:
        pickle.dump(tfidf_vec, fw)
        fw.flush()
    
def load_tfidf(path):
    f_vec = open(path, 'rb')
    vec = pickle.load(f_vec, encoding='utf-8')
    return vec

def get_map_category(df, col_name, threshold):
    df_ = df.copy(deep=True)
    map_ = build_category_map() 
    # map items if in map_
    for i, item in enumerate(df_[col_name]):
        if item in map_:
            df_[col_name][i] = map_[item]
    # get counts for unique item
    counts = df_[col_name].value_counts()
    # get used items from the map
    useful = counts > threshold
    useful = [key for key in useful.keys() if useful[key] == True]
    df_ = df_[df_[col_name].isin(useful) == True]
    return df_


def build_category_map():
    category_map = dict()

    SALE = 'sales'
    ENGINEERING = 'engineering'
    MARKETING = 'marketing'
    OPEARTION = 'operation'
    IT = 'it'
    HEALTH = 'health'
    RESEARCH = 'research'
    CONSTRUCTION = 'construction'
    EDUCATION = 'education'
    CUSTOMER = 'customer'

    category_map = dict.fromkeys(['sales', 'retail'], SALE)
    category_map.update(dict.fromkeys(['engineering'], ENGINEERING))
    category_map.update(dict.fromkeys(['operations'], OPEARTION))
    category_map.update(dict.fromkeys(['it', 'development', 'product', 'information technology', 'design', 'technology', 'tech', 'designing'], IT))
    category_map.update(dict.fromkeys(['customer service'], CUSTOMER))
    category_map.update(dict.fromkeys(['r&d', 'business services', 'aerospace & defense', 'government'], RESEARCH))
    category_map.update(dict.fromkeys(['health care', 'biotech & pharmaceuticals', 'health & fitness'], HEALTH))
    category_map.update(dict.fromkeys(['construction, repair & maintenance'], CONSTRUCTION))
    category_map.update(dict.fromkeys(['education'], EDUCATION))
    return category_map

def draw_confusion_matrix(cf_matrix, labels):
    plt.figure(figsize=(len(labels),len(labels)))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.xaxis.set_ticklabels(labels, rotation=40)
    ax.yaxis.set_ticklabels(labels, rotation=40)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actal Values')
    plt.show()

def get_classification_report(unique_labels, y_test, pred, zero_division=0,has_return=True):
    cf_matrix = confusion_matrix(y_test, pred)
    draw_confusion_matrix(cf_matrix, unique_labels)
    cf_report = classification_report(y_test, pred, zero_division=zero_division)
    print(cf_report)
    if not has_return:
        return cf_report

def get_top_n_jobs_from_clf(df_jobs, pred_department, resume, vec, sim_func, n=10):
    potential_jobs = df_jobs[df_jobs['department'] == pred_department]
    temp = potential_jobs['description_combined'].append(pd.Series(resume))
    matrix = vec.transform(temp)
    term_matrix = matrix.todense()
    sim_matrix = sim_func(term_matrix)
    return np.asarray(sim_matrix[-1][np.where(sim_matrix[-1] < 1)]).argsort()[::-1][:n]

def get_top_n_jobs_from_cluster(df_jobs, pred_cluster, resume, vec, sim_func, n=10):
    potential_jobs = df_jobs[df_jobs['cluster'] == pred_cluster]
    temp = potential_jobs['description_combined'].append(pd.Series(resume))
    matrix = vec.transform(temp)
    term_matrix = matrix.todense()
    sim_matrix = sim_func(term_matrix)
    return np.asarray(sim_matrix[-1][np.where(sim_matrix[-1] < 1)]).argsort()[::-1][:n]

def get_top_n_jobs_from_cf(df_jobs, df_resume, index_similar_applicant, clf, vec, sim_func, n=1):
    cf_jobs = []
    for index in index_similar_applicant:
        similar_applicant = df_resume.iloc[index]
        pred_similar_applicant = clf.predict([similar_applicant['Resume_c']])[0]

        potential_jobs = df_jobs[df_jobs['department'] == pred_similar_applicant]
        temp = potential_jobs['description_combined'].append(pd.Series(similar_applicant['Resume_c']))
        matrix = vec.transform(temp)
        term_matrix = matrix.todense()
        sim_matrix = sim_func(term_matrix)
        job = np.asarray(sim_matrix[-1][np.where(sim_matrix[-1] < 1)]).argsort()[::-1][:n]
        cf_jobs.extend(job)
    return cf_jobs


def elbow_method(data, number):
    wcss = []
    for i in range(1, number+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    kn = KneeLocator(range(1, number+1), wcss, curve='convex', direction='decreasing')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, number+1)),
                            y=wcss))
    fig.add_vline(x=kn.knee, line_width=3, line_dash="dash", line_color="green")

    fig.update_layout(title='Elbow Method',
                      xaxis_title='Number of clusters',
                      yaxis_title='WCSS',
                      title_x=0.5,
                      height=500, 
                      width=800)
    fig.show()