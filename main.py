import numpy as np
from pandas import read_csv, DataFrame
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score


def use_df(f):
    df = read_csv('./USA/usa_elections.dat', header=0, delimiter=';')
    # df = df.fillna(0)
    # df = df.dropna(axis=1)


    def inner():
        f(df)

    return inner


@use_df
def show_dendrogram(df: DataFrame):

    data = df.drop("state.name", axis=1).values
    data = fill_with_avg(data)
    print(data)
    complete_clustering = linkage(data, method="average", metric="euclidean")
    dendrogram(complete_clustering, labels=df["state.name"].values.tolist())
    plt.show()


def fill_with_avg(data):
    missing = ~np.isfinite(data)
    mu = np.nanmean(data, 1, keepdims=1)
    data = np.where(missing, mu, data)
    return data


@use_df
def show_k_average(df: DataFrame):
    print(df)
    data = df.drop("state.name", axis=1).values

    # fill data
    data = fill_with_avg(data)

    print(data)
    wss = calculate_WSS(data, df.shape[0])
    plt.plot(range(1, len(wss) + 1), wss, color='g', linewidth='3')
    plt.xlabel("Value of K")
    plt.xticks(range(1, len(wss) + 1))
    plt.ylabel("Squared Error (Cost)")
    plt.show()
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')

    cluster.fit_predict(data)

    plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.title(f"SK Learn estimated number of clusters = {1 + np.amax(cluster.labels_)}")
    plt.show()


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k, n_init=10).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


def optimal_k(points, max_sil=0, kmax=20):
    k_optimal = 0
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        labels = kmeans.labels_
        curr_sil = silhouette_score(points, labels, metric='euclidean')
        if max_sil < curr_sil:
            max_sil = curr_sil
            k_optimal = k
    return k_optimal


def main():
    show_dendrogram()


if __name__ == '__main__':
    main()
