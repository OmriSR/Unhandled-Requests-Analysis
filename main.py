import json
import numpy as np
from nltk import ngrams
from torch import tensor
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer, util
from pandas import read_csv
from collections import defaultdict

sentencetransformer = SentenceTransformer('all-MiniLM-L6-v2')


# a toy class to couple a sentence and its embedding
class VecReq:
    def __init__(self, request, vector_req):
        self.request = request
        self.embedding = vector_req

    def __repr__(self):
        return self.request


def loadData(data_file):
    req_list = read_csv(data_file, header=0, index_col=0, squeeze=True).tolist()
    return req_list


def vectorizeData(req_list):
    return sentencetransformer.encode(req_list)


# load data and transform
def loadAndVectorize(data_file):
    reqlist = loadData(data_file)
    vecReqlist = vectorizeData(reqlist)
    return reqlist, vecReqlist


def flattenClusters(clusters, total_reqs):
    # flatten the clusters
    flat_clusters = [-1 for _ in range(total_reqs)]
    for i, cluster in enumerate(clusters):
        for req in cluster:
            flat_clusters[req] = i
    return flat_clusters


# cluster the data and return a dictionary of the coupled vector-request
def clusterTheData(data_file, min_samples):
    clusters = defaultdict(list)
    reqList, embeddingsList = loadAndVectorize(data_file)
    # cluster the data
    # labels = DBSCAN(eps=0.7, min_samples=5, metric="l2").fit_predict(embeddingsList)
    # clusters = fineTuneClusters(clusters, int(min_samples))
    labels = flattenClusters(util.community_detection(tensor(embeddingsList), threshold=0.65,
                                                      min_community_size=int(min_samples)), len(reqList))
    for i in range(len(labels)):
        clusters[labels[i]].append(VecReq(reqList[i], embeddingsList[i]))

    clusters["unclustered"] = clusters[-1]
    del clusters[-1]
    clusters = renameClusters(clusters)
    return clusters


def selectClustersRepr(vecreqdict, n_components=3):
    # select representatives for each cluster
    repr_dict = defaultdict(list)
    for k in vecreqdict.keys():
        if k == "unclustered":
            continue
        repr_dict[k] = findRepr(vecreqdict[k], n_components)
    return repr_dict


def findRepr(vecreqlist, n_components):
    # find the representative of a cluster
    # the representatives chosen are the ones with the greatest variance
    # using the power iteration method to find the eigenvectors
    # the representative is the one with the smallest distance to the eigenvectors
    embeddings = np.array([u.embedding for u in vecreqlist])
    square_mat = np.matmul(embeddings.T, embeddings)
    eigenvalues, eigenvectors = np.linalg.eig(square_mat)
    selected = []
    prev_val = None
    for egval, egvec in zip(eigenvalues, eigenvectors):
        if len(selected) == n_components:
            break
        if prev_val != egval:
            prev_val = egval
            req = vecreqlist[np.argmin(np.linalg.norm(embeddings - egvec, axis=1))].request
            if req not in selected:
                selected.append(req)

    return selected


def fineTuneClusters(clusters, min_samples):
    # deprecated, not used in the final version
    # remove clusters with less than min_samples
    # only needed when the DBSCAN algorithm is used
    clusters["unclustered"] = clusters['-1']
    del clusters['-1']
    SmallClustersPresent = True
    while SmallClustersPresent:
        SmallClustersPresent = False
        for k in clusters.keys():
            if len(clusters[k]) < min_samples:
                SmallClustersPresent = True
                clusters["unclustered"].extend(clusters[k])
                del clusters[k]
                break

    return clusters


def renameClusters(clusters):
    # rename the clusters to the most representative bi/tri-gram
    new_clusters = defaultdict(list)
    for k in clusters.keys():
        if k == "unclustered":
            new_clusters[k] = clusters[k]
            name = "unclustered"
        else:
            name = nameCluster(clusters[k])
            if name in new_clusters.keys():
                name = nameCluster(clusters[k], retry=True)
            new_clusters[name].extend(clusters[k])

        new_clusters[name].sort(key=lambda x: x.request)

    return new_clusters


def nameCluster(vecreqlist, retry=False):
    # find the median request
    embeddings = np.array([u.embedding for u in vecreqlist])
    the_med = np.median(embeddings, axis=0)
    median_request = vecreqlist[np.argmin([np.linalg.norm(the_med - embeddings, axis=1)])].request
    # break it to  bi/tri-grams
    bi_tri_grams = [ngram for ngram in ngrams(median_request.split(), 2)]
    bi_tri_grams.extend([ngram for ngram in ngrams(median_request.split(), 3)])
    bi_tri_grams_refined = []
    # remove the words "the" and "is" and remove the "?" from the end of the request
    for btg in bi_tri_grams:
        btg = " ".join(btg)
        btg = btg.replace("?", "")
        btg = btg.replace("the ", "")
        btg = btg.replace("is ", "")
        bi_tri_grams_refined.append(btg)
    # find the closest bi/tri-gram to the median request
    vec_bi_tri_grams = sentencetransformer.encode(bi_tri_grams_refined)
    if retry:
        vec_bi_tri_grams = np.delete(vec_bi_tri_grams, np.argmin([np.linalg.norm(the_med - vec_bi_tri_grams, axis=1)]),
                                     axis=0)
    return bi_tri_grams_refined[np.argmin([np.linalg.norm(the_med - vec_bi_tri_grams, axis=1)])]


def toJson(datadic, repsdic, output_file):
    res = defaultdict(list)
    # generate the json file in the required format
    for key in list(datadic.keys()):
        if key != "unclustered":
            currequests = [req.request for req in datadic[key]]
            cursubdic = dict()
            cursubdic["cluster_name"] = key
            cursubdic["representative_sentences"] = repsdic[key]
            cursubdic["requests"] = currequests
            res["cluster_list"].append(cursubdic)
    res["unclustered"] = [req.request for req in datadic["unclustered"]]
    json.dump(res, open(output_file, "w"), indent=4)


import json
import numpy as np
from nltk import ngrams
from torch import tensor
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer, util
from pandas import read_csv
from collections import defaultdict

sentencetransformer = SentenceTransformer('all-MiniLM-L6-v2')


class VecReq:
    """
    A toy class to couple a sentence and its embedding.
    """

    def __init__(self, request, vector_req):
        self.request = request
        self.embedding = vector_req

    def __repr__(self):
        return self.request


def loadData(data_file):
    """
    Load data from a CSV file and return it as a list of requests.
    """
    req_list = read_csv(data_file, header=0, index_col=0, squeeze=True).tolist()
    return req_list


def vectorizeData(req_list):
    """
    Vectorize a list of requests using the SentenceTransformer model.
    """
    return sentencetransformer.encode(req_list)


def loadAndVectorize(data_file):
    """
    Load and vectorize data from a CSV file.
    """
    reqlist = loadData(data_file)
    vecReqlist = vectorizeData(reqlist)
    return reqlist, vecReqlist


def flattenClusters(clusters, total_reqs):
    """
    Flatten a list of clusters to a list of the same length as the total number of requests,
    where each element represents the index of the cluster that the corresponding request
    belongs to.
    """
    flat_clusters = [-1 for _ in range(total_reqs)]
    for i, cluster in enumerate(clusters):
        for req in cluster:
            flat_clusters[req] = i
    return flat_clusters


def clusterTheData(data_file, min_samples):
    """
    Cluster the data in the specified CSV file and return a dictionary of the coupled vector-request.
    """
    clusters = defaultdict(list)
    reqList, embeddingsList = loadAndVectorize(data_file)
    labels = flattenClusters(
        util.community_detection(
            tensor(embeddingsList), threshold=0.65, min_community_size=int(min_samples)
        ),
        len(reqList)
    )
    for i in range(len(labels)):
        clusters[labels[i]].append(VecReq(reqList[i], embeddingsList[i]))
    clusters["unclustered"] = clusters[-1]
    del clusters[-1]
    clusters = renameClusters(clusters)
    return clusters


def selectClustersRepr(vecreqdict, n_components=3):
    """
    Select representative vectors for each cluster in the specified dictionary of the coupled vector-request.
    """
    repr_dict = defaultdict(list)
    for k in vecreqdict.keys():
        if k == "unclustered":
            continue
        repr_dict[k] = findRepr(vecreqdict[k], n_components)
    return repr_dict


def findRepr(vecreqlist, n_components):
    """
    Find the representative of a cluster.

    The representatives chosen are the ones with the greatest variance
    using the power iteration method to find the eigenvectors.
    The representative is the one with the smallest distance to the eigenvectors.

    Args:
        vecreqlist (list): A list of request objects with embeddings.
        n_components (int): The number of representatives to find.

    Returns:
        list: A list of request objects representing the clusters.
    """
    embeddings = np.array([u.embedding for u in vecreqlist])
    square_mat = np.matmul(embeddings.T, embeddings)
    eigenvalues, eigenvectors = np.linalg.eig(square_mat)
    selected = []
    prev_val = None
    for egval, egvec in zip(eigenvalues, eigenvectors):
        if len(selected) == n_components:
            break
        if prev_val != egval:
            prev_val = egval
            req = vecreqlist[np.argmin(np.linalg.norm(embeddings - egvec, axis=1))].request
            if req not in selected:
                selected.append(req)

    return selected


def renameClusters(clusters):
    """
    Rename the clusters to the most representative bi/tri-gram.

    Args:
        clusters (dict): A dictionary with keys as cluster names and values as request objects.

    Returns:
        dict: A dictionary with keys as new cluster names and values as request objects.
    """
    new_clusters = defaultdict(list)
    for k in clusters.keys():
        if k == "unclustered":
            new_clusters[k] = clusters[k]
            name = "unclustered"
        else:
            name = nameCluster(clusters[k])
            if name in new_clusters.keys():
                name = nameCluster(clusters[k], retry=True)
            new_clusters[name].extend(clusters[k])

        new_clusters[name].sort(key=lambda x: x.request)

    return new_clusters


def fineTuneClusters(clusters, min_samples):
    """
    Deprecated, not used in the final version.

    Remove clusters with less than min_samples.
    Only needed when the DBSCAN algorithm is used.

    Args:
        clusters (dict): A dictionary with keys as cluster names and values as request objects.
        min_samples (int): The minimum number of samples required to form a cluster.

    Returns:
        dict: A dictionary with keys as cluster names and values as request objects.
    """
    clusters["unclustered"] = clusters['-1']
    del clusters['-1']
    small_clusters_present = True
    while small_clusters_present:
        small_clusters_present = False
        for k in clusters.keys():
            if len(clusters[k]) < min_samples:
                small_clusters_present = True
                clusters["unclustered"].extend(clusters[k])
                del clusters[k]
                break

    return clusters


def nameCluster(vecreqlist, retry=False):
    # find the median request
    embeddings = np.array([u.embedding for u in vecreqlist])
    the_med = np.median(embeddings, axis=0)
    median_request = vecreqlist[np.argmin([np.linalg.norm(the_med - embeddings, axis=1)])].request
    # break it to  bi/tri-grams
    bi_tri_grams = [ngram for ngram in ngrams(median_request.split(), 2)]
    bi_tri_grams.extend([ngram for ngram in ngrams(median_request.split(), 3)])
    bi_tri_grams_refined = []
    # remove the words "the" and "is" and remove the "?" from the end of the request
    for btg in bi_tri_grams:
        btg = " ".join(btg)
        btg = btg.replace("?", "")
        btg = btg.replace("the ", "")
        btg = btg.replace("is ", "")
        bi_tri_grams_refined.append(btg)
    # find the closest bi/tri-gram to the median request
    vec_bi_tri_grams = sentencetransformer.encode(bi_tri_grams_refined)
    if retry:
        vec_bi_tri_grams = np.delete(vec_bi_tri_grams, np.argmin([np.linalg.norm(the_med - vec_bi_tri_grams, axis=1)]),
                                     axis=0)
    return bi_tri_grams_refined[np.argmin([np.linalg.norm(the_med - vec_bi_tri_grams, axis=1)])]


def toJson(datadic, repsdic, output_file):
    res = defaultdict(list)

    for key in datadic.keys():
        if key != "unclustered":
            currequests = [req.request for req in datadic[key]]
            cursubdic = {
                "cluster_name": key,
                "representative_sentences": repsdic[key],
                "requests": currequests
            }
            res["cluster_list"].append(cursubdic)

    res["unclustered"] = [req.request for req in datadic["unclustered"]]
    json.dump(res, open(output_file, "w"), indent=4)


def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    VeqRecDict = clusterTheData(data_file, min_size)
    ClustersReprDict = selectClustersRepr(vecreqdict=VeqRecDict, n_components=int(num_rep))
    toJson(VeqRecDict, ClustersReprDict, output_file)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    evaluate_clustering(config['example_solution_file'], config['output_file'])
