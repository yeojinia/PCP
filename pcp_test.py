import plotly.graph_objects as go
import pandas as pd
from pandas import DataFrame

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.spatial import distance


dataframe = pd.read_csv("pd_speech_features_2.csv")
columns = dataframe.columns.values

corr_distance = []

" dissimilarity matrix based on Spearman's correlation coefficient "
for i in columns:
    dist = []
    for j in columns:
        corr, p_value = spearmanr(dataframe[i], dataframe[j])
        dist.append(1 - abs(corr))
    corr_distance.append(dist)

corr_dataframe = DataFrame(corr_distance)
# export_csv = corr_dataframe.to_csv('output.csv')

" parallel coordinates plot "
fig = go.Figure(data=
    go.Parcoords(
        dimensions = list([dict(range = [dataframe[idx].min(skipna = True), dataframe[idx].max(skipna = True)], values = dataframe[idx]) for idx in columns])
    )
)


" dissimilarity matrix based on Pearson correlation coefficient "
def pearson():

    distance = []
    for i in columns:
        dist = []
        for j in columns:
            corr, p_value = pearsonr(dataframe[i], dataframe[j])
            dist.append(1 - abs(corr))
        distance.append(dist)

    corr_dataframe = DataFrame(distance)


" dissimilarity matrix based on Kendall's tau "
def kendall():

    distance = []
    for i in columns:
        dist = []
        for j in columns:
            corr, p_value = kendalltau(dataframe[i], dataframe[j])
            dist.append(1 - abs(corr))
        distance.append(dist)

    corr_dataframe = DataFrame(distance)

def cosine_similarity():

    similarity = []
    for i in columns:
        sim = []
        for j in columns:
            corr, p_value = pearsonr(dataframe[i], dataframe[j])
            sim.append(corr)
        similarity.append(sim)

    corr_dataframe = DataFrame(similarity)


def isomap():

    ''' k-nearest neighbors = n_neighbors '''
    iso = Isomap(n_neighbors=7, n_components=2)
    iso.fit(corr_dataframe)
    manifold_2D = iso.transform(corr_dataframe)
    return manifold_2D

def cMDS(distance_mat):

    # Number of Dimensions
    n = len(distance_mat)

    # Compute Centering matrix C = Identity - 1/n * (11)^T
    C = np.eye(n) - np.ones((n,n))/n

    # B = -1/2 * C * D(2) * C
    B = -0.5 * (C.dot(np.power(distance_mat,2)).dot(C))

    # Eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(B)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]

    w, = np.where(eigenvals>0)
    L = np.diag(np.sqrt(eigenvals[w]))
    V = eigenvecs[:, w]
    Y = V.dot(L)

    return Y


def dendrogram():
    temp = []
    for i in range(0, len(columns)-1, 1):
        for j in range (1+i, len(columns)-1, 1):
            temp.append(corr_dataframe.values[i][j])
    ytdist = np.array(temp)
    Z = hierarchy.linkage(ytdist, method='complete')
    plt.figure()
    dn = hierarchy.dendrogram(Z)
    plt.show()

def main():

    # classical Multidimensional Scaling
    cMDS_result = cMDS(corr_dataframe)
    cMDS_dimension_graph = DataFrame(cMDS_result)
    plt.scatter(cMDS_dimension_graph[0], cMDS_dimension_graph[1], alpha = 0.5)
    plt.show()

    # isomap
    isomap_result = isomap()
    iso_dimension_graph = DataFrame(isomap_result)
    plt.scatter(iso_dimension_graph[0], iso_dimension_graph[1], alpha = 0.5)
    plt.show()

    # parallel coordinates plot
    fig.show()

    # complete linkage clustering
    dendrogram()

if __name__ == "__main__":
    main()
