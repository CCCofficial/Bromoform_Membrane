
from scipy.spatial.distance import cdist
import hdbscan, numpy, h5py
from time import perf_counter



def trainNoiseClassifier(
        soapFile: str, fitsetAddress: str, fitSetSlice: slice = slice(None),allow_single_cluster=True,min_cluster_size=125
):
    print(f"Training HDBSCAN*")
    t1_start = perf_counter()

    with h5py.File(soapFile, "r") as fitfile:
        fitset = fitfile[fitsetAddress][fitSetSlice, :, :3].reshape(-1, 3)
        hdnc = hdbscanNoiseClassifier(
            fitset, min_cluster_size=min_cluster_size, cluster_selection_method="eom",allow_single_cluster=allow_single_cluster
        )

    t1_stop = perf_counter()
    print(f"Time for training: {t1_stop - t1_start} s")
    return hdnc


class hdbscanNoiseClassifier:
    def __init__(self, fitset, **hdbscanArgs):
        self.__fitset = fitset
        self.__clusterer = hdbscan.HDBSCAN(**hdbscanArgs).fit(self.__fitset)
        self.__clusterer.generate_prediction_data()

    @property
    def fitset(self):
        return self.__fitset

    @property
    def clusterer(self):
        return self.__clusterer

    @property
    def exemplars(self):
        if not hasattr(self, "__exemplars"):
            tree = self.__clusterer.condensed_tree_
            raw_tree = tree._raw_tree
            cluster_tree = raw_tree[raw_tree["child_size"] > 1]
            self.__exemplars = dict()
            for i, cluster_id in enumerate(tree._select_clusters()):
                # Get the leaf cluster nodes under the cluster we are considering
                leaves = hdbscan.plots._recurse_leaf_dfs(
                    cluster_tree, cluster_id)
                # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
                result = numpy.array([])
                for leaf in leaves:
                    max_lambda = raw_tree["lambda_val"][
                        raw_tree["parent"] == leaf
                    ].max()
                    points = raw_tree["child"][
                        (raw_tree["parent"] == leaf)
                        & (raw_tree["lambda_val"] == max_lambda)
                    ]
                    result = numpy.hstack((result, points))
                self.__exemplars[i] = dict(
                    points=self.fitset[result.astype(int)],
                    ids=self.fitset[result.astype(int)],
                )
        return self.__exemplars

    def min_dist_to_exemplar(self, point, clusterID):
        dists = cdist([point], self.exemplars[clusterID]["points"])
        return dists.min()

    def dist_vector(self, points):
        result = numpy.zeros(
            (points.shape[0], len(self.exemplars)), dtype=numpy.double)
        for cluster in self.exemplars:
            result[:, cluster] = cdist(points, self.exemplars[cluster]["points"]).min(
                axis=-1
            )
        return result

    def dist_membership_vector(self, point, softmax=False):
        if softmax:
            result = numpy.exp(1.0 / self.dist_vector(point))
            result[~numpy.isfinite(result)] = numpy.finfo(numpy.double).max
        else:
            result = 1.0 / self.dist_vector(point)
            result[~numpy.isfinite(result)] = numpy.finfo(numpy.double).max
        result /= result.sum()
        return result

    def predict(self, data):
        labels, strenghts = hdbscan.approximate_predict(self.__clusterer, data)
        membership_vector = hdbscan.membership_vector(self.__clusterer, data)
        labelsNoNoise = labels.copy()
        isNoise = labelsNoNoise == -1

        labelsNoNoise[isNoise] = numpy.argmax(
            self.dist_membership_vector(data[isNoise]), axis=-1
        )
        return (labels, strenghts), membership_vector, labelsNoNoise


def classifyNPs(
        hdnc: hdbscanNoiseClassifier,
        soapFile: str,
        PCAGroupAddr: str,
        outFile: str,
        whereToSave: str,
):
    """_summary_

    Args:
        hdnc (hdbscanNoiseClassifier): _description_
        soapFile (str): _description_
        PCAGroupAddr (str): _description_
        outFile (str): _description_
        whereToSave (str): _description_
    """
    print(f"Working on {soapFile} and saving on on {outFile}")
    t1_start = perf_counter()
    with h5py.File(soapFile, "r") as datafile:
        g = datafile[PCAGroupAddr]
        for k in g.keys():
            print(f"Applying prediction to trajectory {k}")
            myshape = g[k].shape
            labelshape = (myshape[0], myshape[1])
            memshape = (
                myshape[0],
                myshape[1],
                len(hdnc.clusterer.condensed_tree_._select_clusters()),
            )
            with h5py.File(outFile, "w") as classFile:
                gout = classFile.require_group(whereToSave)
                resGroup = gout.require_group(k)
                labels = resGroup.require_dataset(
                    "labels",
                    shape=labelshape,
                    dtype=numpy.int32,
                    chunks=True
                    # data=lbl.reshape(labelshape),
                )
                labelsStrength = resGroup.require_dataset(
                    "labelsStrength",
                    shape=labelshape,
                    dtype=numpy.float32,
                    chunks=True
                    # data=strg.reshape(labelshape),
                )
                labelsNN = resGroup.require_dataset(
                    "labelsNN",
                    shape=labelshape,
                    dtype=numpy.int32,
                    chunks=True
                    # data=lblNN.reshape(labelshape),
                )
                memberships = resGroup.require_dataset(
                    "memberships",
                    shape=memshape,
                    dtype=numpy.float64,
                    chunks=True
                    # data=mem.reshape(memshape),
                )
                for chunk in g[k].iter_chunks():
                    framesChunk = chunk[0]
                    print(f"classifying frames {framesChunk}")
                    (lbl, strg), mem, lblNN = hdnc.predict(
                        # g[k][framesChunk, :, :3].reshape(-1, 3)
                        g[k][(framesChunk, slice(None), slice(0, 3, 1))
                             ].reshape(-1, 3)
                    )
                    unique_cluster_labels = numpy.unique(lbl)
                    print(f"The unique cluster labels with noise that are found: {unique_cluster_labels}")
                    unique_no_noise_cluster_labels = numpy.unique(lblNN)
                    print(f"The unique cluster labels without noise using distance-based membership for noise points: {unique_cluster_labels}")

                    nframes = lbl.reshape(-1, myshape[1]).shape[0]
                    labels[framesChunk] = lbl.reshape(nframes, myshape[1])
                    labelsStrength[framesChunk] = strg.reshape(
                        nframes, myshape[1])
                    labelsNN[framesChunk] = lblNN.reshape(nframes, myshape[1])
                    memberships[framesChunk] = mem.reshape(
                        nframes, myshape[1], -1)
    t1_stop = perf_counter()
    print(f"Time for {soapFile} -> {outFile}: {t1_stop - t1_start} s")
