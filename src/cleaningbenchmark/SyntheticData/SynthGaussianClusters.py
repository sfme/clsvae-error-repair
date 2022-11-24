import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state
from scipy.stats import special_ortho_group


def _get_y_noise_lists(y_noise_tags, idxs):

    y_noise_lists = [[] for k in range(max(y_noise_tags) + 1)]

    for noise_tag, idx in zip(y_noise_tags, idxs):
        y_noise_lists[noise_tag].append(idx)

    return y_noise_lists


def from_tags_to_listoflists(y_tags):

    """ y_tags is categorical tag (numerical) vector """

    y_lists = [[] for k in range(max(y_tags) + 1)]

    for idx, class_tag in enumerate(y_tags):
        if class_tag >= 0:
            y_lists[class_tag].append(idx)

    return y_lists


def _get_y_collapsed(y_lists, n_size):

    y_lists_collapsed = np.ones(n_size, dtype=int) * -1

    for ii, class_idxs in enumerate(y_lists):
        for idx in class_idxs:
            y_lists_collapsed[idx] = ii

    return y_lists_collapsed


class SynthGaussianClusters(object):

    """Class's code inspired by PyOD package: pyod.utils.data

    random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

    """

    def __init__(
        self,
        n_samples=100,
        n_clusters=4,
        corrupt_prob=0.1,
        n_features=2,
        scale_density="same",
        size_cluster="same",
        std_scaler_cluster=1.0,
        dist=0.25,
        random_state=None,
        noise_type=None,
        noise_type_defs=None,
    ):

        self.inlier_clusters = []
        self.outlier_clusters = []

        # initialize a random state and seeds for the instance
        self.random_state = check_random_state(random_state)

        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.corrupt_prob = corrupt_prob
        self.n_features = n_features
        self.scale_density = scale_density
        self.size_cluster = size_cluster
        self.std_scaler_cluster = std_scaler_cluster
        self.dist = dist  # distance between inlier clusters
        self.noise_type = noise_type
        self.noise_type_defs = noise_type_defs  # specific noise process defs

        self.n_outliers = int(n_samples * corrupt_prob)
        self.n_inliers = n_samples - self.n_outliers

        ## Get cluster definitions
        ret = self._get_cluster_defs()
        self.clusters_size, self.clusters_density, self.n_outliers_cluster = ret

        ## Construct dataset
        # create clusters: clean (inlier) and dirty (outlier)
        (
            self.X,
            self.y_noise,
            self.y_class,
            self.X_gt,
            self.y_noise_lists,
            _,
            _,
        ) = self._make_clusters(noise_type)

        # prepare y_noise_lists before shuffle
        self.y_noise_tags = _get_y_collapsed(self.y_noise_lists, len(self.y_noise))

        # shuffle dataset arrays in unison
        (
            self.X,
            self.y_noise,
            self.y_class,
            self.X_gt,
            self.y_noise_tags,
            _,
        ) = self._shuffle_arrays(
            self.X, self.y_noise, self.y_class, self.X_gt, self.y_noise_tags
        )

        # apply changes to y_noise_lists
        self.y_noise_lists = from_tags_to_listoflists(self.y_noise_tags)

    def _shuffle_arrays(self, X, y_noise, y_class, X_gt, y_noise_tags):

        sh_idxs = self.random_state.permutation(X.shape[0])

        return (
            X[sh_idxs, :],
            y_noise[
                sh_idxs,
            ],
            y_class[
                sh_idxs,
            ],
            X_gt[sh_idxs, :],
            y_noise_tags[
                sh_idxs,
            ],
            sh_idxs,
        )

    def _get_cluster_defs(self):

        # number of inliers per cluster
        if self.size_cluster == "same":
            _sz = [int(self.n_inliers / self.n_clusters)] * (self.n_clusters - 1)
            clusters_size = _sz + [int(self.n_inliers - sum(_sz))]

        elif self.size_cluster == "different":
            if (self.n_clusters * 10) > self.n_samples:
                raise ValueError(
                    "number of samples should be at least 10 times of"
                    "the number of clusters"
                )

            if (self.n_clusters * 10) > self.n_inliers:
                raise ValueError(
                    "contamination ratio is too high, try to increase"
                    " number of samples or decrease the contamination"
                )

            _r = 1.0 / self.n_clusters  # unif. alloc of points
            _offset = self.random_state.uniform(
                _r * 0.2, _r * 0.4, size=(int(self.n_clusters / 2),)
            ).tolist()
            _offset += [
                i * -1.0 for i in _offset
            ]  # used in offsetting alloc at clusters
            clusters_size = np.round(
                np.multiply(self.n_inliers, np.add(_r, _offset))
            ).astype(int)

            if self.n_clusters % 2 == 0:  # if it is even number
                clusters_size[self.n_clusters - 1] += self.n_inliers - sum(
                    clusters_size
                )

            else:
                clusters_size = np.append(
                    clusters_size, self.n_inliers - sum(clusters_size)
                )
        else:
            raise ValueError("size should be a string of value 'same' or 'different'")

        # scale (std) of Gaussian clusters
        if self.scale_density == "same":
            # lower bound std is 0.1, and higher bound is 0.5
            clusters_density = (
                self.std_scaler_cluster
                * self.random_state.uniform(low=0.1, high=0.5, size=(1,))
            ).tolist() * self.n_clusters

        elif self.scale_density == "different":
            clusters_density = self.std_scaler_cluster * self.random_state.uniform(
                low=0.1, high=0.5, size=(self.n_clusters,)
            )
        else:
            raise ValueError(
                "density should be a string of value 'same' or 'different'"
            )

        # get number of outliers for every cluster
        _n_outliers = []
        for i in range(self.n_clusters):
            _n_outliers.append(int(round(clusters_size[i] * self.corrupt_prob)))

        _diff = int((self.n_outliers - sum(_n_outliers)) / self.n_clusters)

        for i in range(self.n_clusters - 1):
            _n_outliers[i] += _diff

        _n_outliers[self.n_clusters - 1] += self.n_outliers - sum(_n_outliers)

        self.random_state.shuffle(_n_outliers)

        return clusters_size, clusters_density, _n_outliers

    def _make_clusters(self, noise_type=None):

        """ generate clustered data, based on defs, and select a noise type """

        X_clusters, y_clusters = [], []
        X, y = (
            np.zeros([self.n_samples, self.n_features]),
            np.zeros(
                [
                    self.n_samples,
                ],
                dtype=int,
            ),
        )
        y_class = np.zeros(
            [
                self.n_samples,
            ],
            dtype=int,
        )  # cluster / class tags
        X_gt = np.zeros([self.n_samples, self.n_features])  # ground-truth X's

        y_nt_offset = 0
        y_noise_tags_acc = []
        y_noise_idxs_acc = []

        # TODO: In the future do full matrix? (grid instead, each cell is a hypercube, then randomly select a hypercube from that grid).

        # define regions of space (boxes) where (clean clusters) will be placed
        center_box = list(
            filter(
                lambda a: a != 0,
                np.linspace(
                    -np.power(self.n_samples * self.n_clusters, self.dist),
                    np.power(self.n_samples * self.n_clusters, self.dist),
                    self.n_clusters + 2,
                ),
            )
        )

        # Gen. cluster centers, at uniformly at random in center-box
        cluster_centers = []
        for jj in range(self.n_clusters):
            cluster_centers.append(
                self.random_state.uniform(
                    center_box[jj],
                    center_box[jj + 1],
                    (1, self.n_features),
                )
            )
        self.cluster_centers = np.vstack(cluster_centers)

        # Create inlier clusters
        tracker_idx = 0  # index tracker for value assignment
        for jj in range(self.n_clusters):
            inliers, outliers, ground_truth = [], [], []

            _blob, _y = make_blobs(
                n_samples=self.clusters_size[jj],
                centers=self.cluster_centers[jj].reshape(1, -1),
                cluster_std=self.clusters_density[jj],
                random_state=self.random_state,
            )
            inliers.append(_blob)

            # injection of outlier clusters
            if self.noise_type is None or self.noise_type == "type_1":
                """ just random cluster inside bounding box """
                _n_groups = self.noise_type_defs["n_groups"]
                _smin_std = self.noise_type_defs["scale_min_std"]
                _smax_std = self.noise_type_defs["scale_max_std"]

                aux_outliers, y_noise_tags = self.noise_type_1_cluster(
                    jj,
                    center_box,
                    n_groups=_n_groups,
                    scale_min_std=_smin_std,
                    scale_max_std=_smax_std,
                )
                outliers.append(aux_outliers)
                ground_truth.append(
                    np.tile(self.cluster_centers[jj, :], (aux_outliers.shape[0], 1))
                )

                y_noise_tags = y_noise_tags + y_nt_offset  #  NOTE: general case
                y_nt_offset = max(y_noise_tags) + 1

            elif self.noise_type == "type_2":
                """ additive noise with epsilon: x = x + epsilon. epsilon is given by uniform """
                _n_groups = self.noise_type_defs["n_groups"]
                _err_min = self.noise_type_defs["err_min"]
                _err_max = self.noise_type_defs["err_max"]

                aux_outliers, aux_inliers, y_noise_tags = self.noise_type_2_cluster(
                    jj, n_groups=_n_groups, err_min=_err_min, err_max=_err_max
                )
                outliers.append(aux_outliers)
                ground_truth.append(aux_inliers)

                y_noise_tags = y_noise_tags + y_nt_offset  #  NOTE: general case
                y_nt_offset = max(y_noise_tags) + 1
            else:
                raise ValueError("Cluster noising type not found !!")

            _y = np.append(_y, [1] * int(self.n_outliers_cluster[jj]))

            # generate X
            if len(np.concatenate(outliers)) > 0:
                # X noised
                crush_inliers = np.concatenate(inliers)
                crush_outliers = np.concatenate(
                    outliers
                )  # np.concat due to empty np.array sometimes
                stacked_X_temp = np.vstack((crush_inliers, crush_outliers))
                X_clusters.append(stacked_X_temp)
                tracker_idx_new = tracker_idx + stacked_X_temp.shape[0]
                X[tracker_idx:tracker_idx_new, :] = stacked_X_temp

                # X ground-truth
                stacked_Xgt_temp = np.vstack(
                    (np.concatenate(inliers), np.concatenate(ground_truth))
                )
                X_gt[tracker_idx:tracker_idx_new, :] = stacked_Xgt_temp

                # Y noise labels -- per (systematic) error group
                y_noise_idxs_acc.append(
                    list(
                        range(
                            (tracker_idx_new - crush_outliers.shape[0]), tracker_idx_new
                        )
                    )
                )
                y_noise_tags_acc.append(y_noise_tags)

            else:
                X_clusters.append(np.concatenate(inliers))

            # generate Y (inlier / outlier)
            y_clusters.append(_y)
            y[
                tracker_idx:tracker_idx_new,
            ] = _y

            # save class tags (cluster id's)
            y_class[
                tracker_idx:tracker_idx_new,
            ] = jj

            tracker_idx = tracker_idx_new

        y_noise_lists = _get_y_noise_lists(
            np.concatenate(y_noise_tags_acc), np.concatenate(y_noise_idxs_acc)
        )

        return X, y, y_class, X_gt, y_noise_lists, X_clusters, y_clusters

    def noise_type_1_cluster(
        self, index_c, center_box, n_groups=1, scale_min_std=None, scale_max_std=None
    ):

        if scale_min_std is None:
            _scale_min_std = 3.5
        else:
            _scale_min_std = scale_min_std

        if scale_max_std is None:
            _scale_max_std = 4.5
        else:
            _scale_max_std = scale_max_std

        center_box_l = center_box[index_c] * (
            1.2 + self.dist + self.clusters_density[index_c]
        )
        center_box_r = center_box[index_c + 1] * (
            1.2 + self.dist + self.clusters_density[index_c]
        )

        _y_noise_tags = []
        outliers_acc = []

        if n_groups > 1:
            # for now equal partition of size between systematic groups #TODO: do unequal later?
            _sz = [int(self.n_outliers_cluster[index_c] / n_groups)] * (n_groups - 1)
            _clusters_size = _sz + [int(self.n_outliers_cluster[index_c] - sum(_sz))]
        else:
            _clusters_size = [self.n_outliers_cluster[index_c]]

        for group_idx in range(0, n_groups):
            if _clusters_size[group_idx] > 0:
                _outliers = make_blobs(
                    n_samples=_clusters_size[group_idx],
                    centers=1,
                    cluster_std=self.random_state.uniform(
                        self.clusters_density[index_c] * _scale_min_std,
                        self.clusters_density[index_c] * _scale_max_std,
                        size=(1,)[0],
                    ),
                    center_box=(center_box_l, center_box_r),
                    n_features=self.n_features,
                    random_state=self.random_state,
                )[0]

                _y_noise_tags.append([group_idx] * _outliers.shape[0])
                outliers_acc.append(_outliers)

        _y_noise_tags = np.concatenate(_y_noise_tags)
        outliers = np.concatenate(outliers_acc)

        return outliers, _y_noise_tags

    def noise_type_2_cluster(self, index_c, n_groups=1, err_min=None, err_max=None):

        # NOTE: Haar random rotation matrix, to generate additive error around hypersphere of norm epsilon

        if err_min is None:
            _err_min = self.clusters_density[index_c] * 3
            # *0.8 min_const*sigma_cluster
        else:
            _err_min = self.clusters_density[index_c] * err_min

        if err_max is None:
            _err_max = self.clusters_density[index_c] * 4.5
            # *2  max_const*sigma_cluster
        else:
            _err_max = self.clusters_density[index_c] * err_max

        if n_groups > 1:
            # for now equal partition of size between systematic groups #TODO: do unequal later?
            _sz = [int(self.n_outliers_cluster[index_c] / n_groups)] * (n_groups - 1)
            _clusters_size = _sz + [int(self.n_outliers_cluster[index_c] - sum(_sz))]
        else:
            _clusters_size = [self.n_outliers_cluster[index_c]]

        _y_noise_tags = []
        outliers_acc = []
        inliers_acc = []

        # Create ground-truth (clean) datapoints, group is cluster
        for group_idx in range(0, n_groups):
            if _clusters_size[group_idx] > 0:
                _inliers = make_blobs(
                    n_samples=_clusters_size[group_idx],
                    centers=self.cluster_centers[index_c].reshape(1, -1),
                    cluster_std=self.clusters_density[index_c],
                    random_state=self.random_state,
                )[0]

                add_err = self.random_state.uniform(low=_err_min, high=_err_max)

                # to init transformation, though O(N) Haar distribution does the work
                v = np.random.randn(self.n_features)
                v /= np.linalg.norm(v)
                v = v * add_err  # norm of vector is add_err

                # rotation matrix
                rnd_rot_mtx = special_ortho_group.rvs(self.n_features)

                # additive noise -- systematic for the group
                _outliers = _inliers + np.dot(rnd_rot_mtx, v)

                _y_noise_tags.append([group_idx] * _outliers.shape[0])
                inliers_acc.append(_inliers)
                outliers_acc.append(_outliers)

        _y_noise_tags = np.concatenate(_y_noise_tags)

        inliers = np.concatenate(inliers_acc)
        outliers = np.concatenate(outliers_acc)

        return outliers, inliers, _y_noise_tags
