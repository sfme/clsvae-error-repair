import numpy as np
from skimage import draw as skidraw
import itertools


class SimpleShapesWithStripes(object):

    """
    Image dataset, simple shapes in white, with added white stripes.
    Size: 28x28

    """

    def __init__(
        self,
        n_samples=600,
        corrupt_prob=0.1,
        random_state=None,
        combs_on=True,
    ):
        self.img_size = (28, 28)

        # flag that controls if noise combinations is in dataset
        self.combs_on = combs_on

        # initialize a random generator, using seed
        self.random_gen = np.random.default_rng(random_state)

        self.n_samples = n_samples
        self.corrupt_prob = corrupt_prob
        self.n_outliers = int(n_samples * corrupt_prob)
        self.n_inliers = n_samples - self.n_outliers

        self.probs_clean_class = np.ones(4) * 1.0 / 4
        # self.probs_dirty_class # defined in _add_noise

        ## contruct dataset
        (
            self.X,
            self.y_noise,
            self.y_class,
            self.X_gt,
            self.y_noise_lists,
            self.y_noise_lists_combs,
        ) = self._build_dataset()

    ## Clean Shapes
    @staticmethod
    def _base_rectangle():

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        start = (1, 1)
        end = (12, 6)
        base_xx, base_yy = skidraw.rectangle(start, end=end, shape=img_shape)
        base_xx = base_xx.flatten()
        base_yy = base_yy.flatten()
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @staticmethod
    def _base_triangle():

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        tri_verts = np.array(((0, 3), (17, 0), (6, 15)))
        base_xx, base_yy = skidraw.polygon(
            tri_verts[:, 0], tri_verts[:, 1], shape=img_shape
        )
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @staticmethod
    def _base_disk():

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        disk_center = (8, 8)
        disk_radius = 8
        base_xx, base_yy = skidraw.disk(disk_center, disk_radius, shape=img_shape)
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @staticmethod
    def _base_ellipse():

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        base_xx, base_yy = skidraw.ellipse(
            8, 8, 4, 10, shape=img_shape, rotation=3 * np.pi / 4.0
        )
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @classmethod
    def _base_shapes(cls, list_fbase=None):
        """
        - assumes 28x28 image
        - binary image
        """

        if list_fbase is None:
            _list_fbase = [
                cls._base_rectangle,
                cls._base_triangle,
                cls._base_disk,
                cls._base_ellipse,
            ]
        else:
            _list_fbase = list_fbase

        l_img_bases = []
        l_bbox_bases = []
        l_idxs_bases = []

        for func_shape in _list_fbase:
            _img_idxs, _bbox_shape, _img_base = func_shape()
            l_img_bases.append(_img_base)
            l_bbox_bases.append(_bbox_shape)
            l_idxs_bases.append(_img_idxs)

        img_bases = np.stack(l_img_bases, axis=0)

        return l_idxs_bases, l_bbox_bases, img_bases

    def _get_rnd_imgs(self, n_imgs, idxs_shape, bbox_shape, img_size=(28, 28)):
        """
        - Generate images through randomly translated base shape, given as arg.

        """

        imgs = np.zeros((n_imgs, *img_size), dtype=np.uint8)
        _n_pixels = len(idxs_shape["x"])

        # get random shift (translation) of base image
        _shift_xx = self.random_gen.integers(
            img_size[0] - (bbox_shape["x"][1] + 1) + 1, size=n_imgs
        )
        _shift_yy = self.random_gen.integers(
            img_size[1] - (bbox_shape["y"][1] + 1) + 1, size=n_imgs
        )
        # NOTE: +1 inside brackets: no shape will touch "edges" of image.

        _pos_xx = _shift_xx.reshape(-1, 1) + idxs_shape["x"].reshape(1, -1)
        _pos_xx = _pos_xx.flatten()
        _pos_yy = _shift_yy.reshape(-1, 1) + idxs_shape["y"].reshape(1, -1)
        _pos_yy = _pos_yy.flatten()

        _img_range = np.arange(n_imgs).reshape(-1, 1).repeat(_n_pixels, axis=1)
        _img_range = _img_range.flatten()

        imgs[_img_range, _pos_xx, _pos_yy] = 1

        return imgs

    ## Dirty Patterns
    @staticmethod
    def _noise_strip_vert():
        """ noise: vertical strip """

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        base_xx, base_yy = skidraw.rectangle(
            (0, 20), extent=(28, 1), shape=img_base.shape
        )
        base_xx = base_xx.flatten()
        base_yy = base_yy.flatten()
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @staticmethod
    def _noise_strip_horz():
        """ noise: horizontal strip """

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        base_xx, base_yy = skidraw.rectangle(
            (15, 0), extent=(1, 28), shape=img_base.shape
        )
        base_xx = base_xx.flatten()
        base_yy = base_yy.flatten()
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @staticmethod
    def _noise_strip_diag_up():
        """ noise: diagonal strip top left corner """

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        diag_cord = np.array([(15, 0), (0, 15), (0, 16), (16, 0)])
        base_xx, base_yy = skidraw.polygon(
            diag_cord[:, 0], diag_cord[:, 1], shape=img_base.shape
        )
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @staticmethod
    def _noise_strip_diag_down():
        """ noise: diagonal strip bottom right corner """

        img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        diag_cord = np.array([(28, 10), (10, 28), (11, 28), (28, 11)])
        base_xx, base_yy = skidraw.polygon(
            diag_cord[:, 0], diag_cord[:, 1], shape=img_base.shape
        )
        img_base[base_xx, base_yy] = 1

        img_idxs = {"x": None, "y": None}
        img_idxs["x"] = base_xx
        img_idxs["y"] = base_yy

        bbox_shape = {"x": None, "y": None}
        # e.g. bbox_shape["x"] -> (start, length)
        bbox_shape["x"] = (min(base_xx.flatten()), max(base_xx.flatten()) + 1)
        bbox_shape["y"] = (min(base_yy.flatten()), max(base_yy.flatten()) + 1)

        return img_idxs, bbox_shape, img_base

    @classmethod
    def _base_noises(cls):
        """
        - assumes 28x28 image
        - binary image
        - returns base noise patterns (images)

        #NOTE: may chnage in future!
        """

        _list_fbase = [
            cls._noise_strip_vert,
            cls._noise_strip_horz,
            cls._noise_strip_diag_up,
            cls._noise_strip_diag_down,
        ]

        l_img_bases = []
        l_bbox_bases = []
        l_idxs_bases = []

        for func_shape in _list_fbase:
            _img_idxs, _bbox_shape, _img_base = func_shape()
            l_img_bases.append(_img_base)
            l_bbox_bases.append(_bbox_shape)
            l_idxs_bases.append(_img_idxs)

        img_bases = np.stack(l_img_bases, axis=0)

        return l_idxs_bases, l_bbox_bases, img_bases

    def _make_clean_dataset(self):

        # NOTE: Assert, idxs_bases len() is equal to self.probs_clean_class (i.e. 4 classes)

        # empty data structs
        imgs = np.zeros((self.n_samples, *self.img_size), dtype=np.uint8)
        y_class = np.empty(self.n_samples, dtype=int)

        # define partitions
        parts = self.probs_clean_class * self.n_samples
        parts = np.floor(parts).astype(int)

        diff_int = int(self.n_samples - parts.sum())

        if diff_int > 0:
            pick_idxs = self.random_gen.choice(
                np.arange(len(parts)), size=diff_int, replace=False
            )
            parts[pick_idxs] += 1

        idxs_bases, bbox_bases, _ = self._base_shapes()

        offset = 0
        for img_cls, part_size in enumerate(parts):
            _rnd_imgs = self._get_rnd_imgs(
                part_size, idxs_bases[img_cls], bbox_bases[img_cls]
            )
            imgs[offset : (offset + part_size), :, :] = _rnd_imgs
            y_class[offset : (offset + part_size)] = img_cls

            offset += part_size

        # shuffle dataset
        new_idxs = self.random_gen.permutation(self.n_samples)
        imgs = imgs[new_idxs, :, :]
        y_class = y_class[new_idxs]

        # flatten images
        imgs = imgs.reshape(imgs.shape[0], -1)

        return imgs, y_class

    def _add_noise(self, X_gt):

        dirtied_imgs = X_gt.copy()
        dirtied_imgs = dirtied_imgs.reshape(self.n_samples, *self.img_size)

        _, _, noise_bases = self._base_noises()
        n_noises = noise_bases.shape[0]

        noised_idxs = self.random_gen.permutation(self.n_samples)[: self.n_outliers]

        y_noise = np.zeros(self.n_samples, dtype=int)
        y_noise[noised_idxs] = 1

        ## Define Noising

        # pick 1 noise at a time (single noise)
        _1_noise_list = np.arange(n_noises).tolist()

        # pick 2 noises at a time (combinations noise)
        if self.combs_on:
            _2_noises_list = list(itertools.combinations(np.arange(n_noises), 2))
        else:
            _2_noises_list = []

        # total possibilities of noise types (inc. combinations)
        _total_noise_list = _1_noise_list + _2_noises_list
        n_partitions = len(_total_noise_list)

        # (about) uniform probability for each noise event #TODO: in future non-uniform?
        _prob_list = np.ones(n_partitions) * (1.0 / n_partitions)
        self.probs_dirty_class = _prob_list

        # get partitions (sizes) for each noise type
        parts = _prob_list * self.n_outliers
        parts = np.floor(parts).astype(int)

        diff_int = int(self.n_outliers - parts.sum())

        if diff_int > 0:
            pick_idxs = self.random_gen.choice(
                np.arange(len(parts)), size=diff_int, replace=False
            )
            parts[pick_idxs] += 1

        # get y_noise_lists
        y_noise_lists = [[] for _ in range(n_noises)]
        offset = 0
        for noise_classes, part_size in zip(_total_noise_list, parts):
            if isinstance(noise_classes, int):
                # only one index / class
                y_noise_lists[noise_classes].extend(
                    noised_idxs[offset : (offset + part_size)].tolist()
                )
            else:
                for noise_cls in noise_classes:
                    y_noise_lists[noise_cls].extend(
                        noised_idxs[offset : (offset + part_size)].tolist()
                    )
            offset += part_size

        # add noise patterns to image
        for noise_tag, _cur_noised_idxs in enumerate(y_noise_lists):
            dirtied_imgs[_cur_noised_idxs, :, :] = np.logical_or(
                dirtied_imgs[_cur_noised_idxs, :, :], noise_bases[noise_tag, :, :]
            )

        # get y_noise_lists_combs (use after is optional)
        # -> register combinations of noise patterns as labels, e.g. use in stratisfied sampling.
        if self.combs_on:
            y_noise_lists_combs = [[] for _ in range(n_partitions)]
            offset = 0
            for noise_tag, part_size in enumerate(parts):
                y_noise_lists_combs[noise_tag].extend(
                    noised_idxs[offset : (offset + part_size)].tolist()
                )
                offset += part_size
        else:
            y_noise_lists_combs = None

        # flatten images
        dirtied_imgs = dirtied_imgs.reshape(self.n_samples, -1)

        return dirtied_imgs, y_noise, y_noise_lists, y_noise_lists_combs

    def _build_dataset(self):

        X_gt, y_class = self._make_clean_dataset()
        X, y_noise, y_noise_lists, y_noise_lists_combs = self._add_noise(X_gt)

        return X, y_noise, y_class, X_gt, y_noise_lists, y_noise_lists_combs
