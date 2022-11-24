"""
This module defines a collection of systematic
statistical noise classes.
"""

import numpy as np
from skimage import draw as skidraw
import itertools
from collections import namedtuple

import warnings

from . import NoiseModel

"""
This model implements Missing Values that
are not at random, i.e., disproportionately
affect high-value features.
"""


class MissingSystematicNoiseModel(NoiseModel.NoiseModel):
    """
    K is the number of features to corrupt
    p is the fraction of the top records to
    corrupt (within the selected set)
    """

    def __init__(self, shape, probability=0, feature_importance=[], k=1, p=0.1):

        super(MissingSystematicNoiseModel, self).__init__(
            shape, probability, feature_importance, True
        )
        self.k = k
        self.p = p

    def corrupt(self, X):
        hvfeature = self.feature_importance[0]
        means = np.mean(X, axis=0)
        Ns = np.shape(X)[0]
        ps = np.shape(X)[1]
        Y = X

        for i in np.argsort(X[:, hvfeature]):
            if np.random.rand(1, 1) < self.p:
                a = np.random.choice(self.feature_importance[0 : self.k], 1)
                Y[i, a] = means[a]

        return Y


"""
This model implements noise that resembles entity resolution noise
"""


class ERNoiseModel(NoiseModel.NoiseModel):
    """
    Applies ER Noise to a subset of ks features
    p is the error rate
    """

    def __init__(self, shape, probability=0, feature_importance=[], ks=[], z=3):

        super(ERNoiseModel, self).__init__(
            shape, probability, feature_importance, False
        )
        self.ks = ks
        self.z = z

    def corrupt(self, X):
        Ns = np.shape(X)[0]
        ps = np.shape(X)[1]
        for i in range(0, Ns):
            ern = np.random.zipf(self.z, (1, 1))
            eri = np.random.choice(range(0, Ns), ern)
            for j in eri:
                X[j, self.ks] = X[i, self.ks]
        return X


######## Image Noise Models: Systematic ########


class ImageSystematicSimpleShapes(object):
    def __init__(
        self,
        img_data_shape,
        img_prob_noise,
        min_val=0.0,
        max_val=1.0,
        prob_min=0.5,
        pixel_val_fixed=None,
        number_blocks=1,
        rand_blocks=False,
        side_len=4,
        std_shift=(10, 10),
        use_other_patterns=False,
        random_state=None,
        combs_on=False,
    ):

        # initialize a random generator, using seed
        self.random_gen = np.random.default_rng(random_state)

        self.n_samples = img_data_shape[0]
        self.img_size = img_data_shape[1:]

        self.corrupt_prob = img_prob_noise  # probability of noise instance
        # self.n_outliers = int(self.n_samples * self.corrupt_prob)
        # self.n_inliers = self.n_samples - self.n_outliers

        self.pixel_min = min_val
        self.pixel_max = max_val
        self.prob_min = prob_min
        self.pixel_val_fixed = pixel_val_fixed

        self.number_blocks = number_blocks
        self.rand_blocks = rand_blocks

        self.side_len = side_len
        self.std_shift = std_shift

        self.other_patterns = use_other_patterns

        self.combs_on = combs_on

        # when using user defined shift or side_len, check correctness
        if self.side_len > max(self.img_size[0], self.img_size[1]):
            raise ValueError(
                "Error: side_len is bigger than image size, make it smaller!"
            )
        if ((self.side_len + self.std_shift[0]) > self.img_size[0]) or (
            (self.side_len + self.std_shift[1]) > self.img_size[1]
        ):
            raise ValueError(
                "Error: side_len plus std_shift is bigger than image size, make it smaller!"
            )

        self._init_noise_defs()

    def _init_noise_defs(self):
        """ define noising process """

        (
            self.idxs_noise_bases,
            self.bbox_noise_bases,
            self.img_noise_bases,
            self.pixel_vals_bases,
        ) = self._base_noises()
        self.n_noises = self.img_noise_bases.shape[0]

        # pick 1 noise at a time (single noise)
        _1_noise_list = np.arange(self.n_noises).tolist()

        # pick 2 noises at a time (combinations noise)
        if self.combs_on:
            _2_noises_list = list(itertools.combinations(np.arange(self.n_noises), 2))
        else:
            _2_noises_list = []

        # total possibilities of noise types (inc. combinations)
        self.total_noise_list = _1_noise_list + _2_noises_list
        self.n_noise_partitions = len(self.total_noise_list)

        # (about) uniform probability for each noise event #TODO: in future non-uniform?
        self.probs_dirty_class = np.ones(self.n_noise_partitions) * (
            1.0 / self.n_noise_partitions
        )

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

    @staticmethod
    def _noise_square_block_base(img_shape=(28, 28), side_len=4, add_shift=(0, 0)):
        """ noise: square block of pixels, fixed (not random?) for now. """

        # img_shape = (28, 28)
        img_base = np.zeros(img_shape, dtype=np.uint8)

        start = (0 + add_shift[0], 0 + add_shift[1])
        end = (side_len - 1 + add_shift[0], side_len - 1 + add_shift[1])

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

    def _noise_square_blocks(self, side_len=4, std_shift=(0, 0)):

        blocks_all_idxs_set = set()
        blocks_list = list()

        if self.number_blocks == 1 and (not self.rand_blocks):
            # non-random single square block (systematic error) dataset
            _img_idxs, _img_bbox, _img_base = self._noise_square_block_base(
                self.img_size, side_len, std_shift
            )
            return [_img_idxs], [_img_bbox], [_img_base]

        else:
            # random single or several square block (systematic error) dataset
            cnt = 0
            jj = 0  # number of tries before giving up
            while (cnt < self.number_blocks) and (jj < 250):
                # add random shift
                eps_x = np.random.randint(0, self.img_size[0] - side_len)
                eps_y = np.random.randint(0, self.img_size[1] - side_len)
                eps_shift = (eps_x, eps_y)
                (
                    _cur_img_idxs,
                    _cur_img_bbox,
                    _cur_img_base,
                ) = self._noise_square_block_base(self.img_size, side_len, eps_shift)

                _img_idxs_set = set(zip(_cur_img_idxs["x"], _cur_img_idxs["y"]))

                if len(_img_idxs_set & blocks_all_idxs_set) == 0:
                    blocks_all_idxs_set.update(_img_idxs_set)
                    blocks_list.append((_cur_img_idxs, _cur_img_bbox, _cur_img_base))
                    cnt += 1
                # else: retry other square block placement

                jj += 1

            if cnt < self.number_blocks:
                warnings.warn(
                    f"Was not able to place all {self.number_blocks} noise patterns in image, only {cnt} were placed instead."
                )

            return tuple(map(list, zip(*blocks_list)))

    def _base_noises(self):

        l_idxs_bases, l_bbox_bases, l_img_bases = self._noise_square_blocks(
            self.side_len, self.std_shift
        )

        if self.other_patterns:
            _list_fbase = [
                self._noise_strip_vert,
                self._noise_strip_horz,
                self._noise_strip_diag_up,
                self._noise_strip_diag_down,
            ]

            for func_shape in _list_fbase:
                _img_idxs, _bbox_shape, _img_base = func_shape()
                l_img_bases.append(_img_base)
                l_bbox_bases.append(_bbox_shape)
                l_idxs_bases.append(_img_idxs)

        img_bases = np.stack(l_img_bases, axis=0)

        # define pixel value per error type
        # TODO: in future Gaussian noise as well to fill in shapes, as an option?
        l_pixel_vals = []
        for _ in range(len(l_idxs_bases)):
            l_pixel_vals.append(self._get_pixel_val())

        return l_idxs_bases, l_bbox_bases, img_bases, l_pixel_vals

    def _get_pixel_val(self):

        if self.pixel_val_fixed is not None:
            pixel_val = self.pixel_val_fixed

        else:
            # random value (either black / white)
            pixel_val = np.random.choice(
                [self.pixel_min, self.pixel_max], p=[self.prob_min, 1 - self.prob_min]
            )

        return pixel_val

    def _fill_pixels(self, arr_noised, inst_idxs, x_idxs, y_idxs, pixel_val):

        _temp_arr = arr_noised[inst_idxs]
        _temp_arr[:, x_idxs, y_idxs] = pixel_val

        arr_noised[inst_idxs] = _temp_arr

    def _add_noise(self, X_gt):

        n_samples = X_gt.shape[0]

        dirtied_imgs = X_gt.copy()
        dirtied_imgs = dirtied_imgs.reshape(n_samples, *self.img_size)

        n_outliers = int(n_samples * self.corrupt_prob)
        # n_inliers = n_samples - n_outliers

        # get partitions (sizes) for each noise type
        noise_parts = self.probs_dirty_class * n_outliers
        noise_parts = np.floor(noise_parts).astype(int)

        diff_int = int(n_outliers - noise_parts.sum())

        if diff_int > 0:
            pick_idxs = self.random_gen.choice(
                np.arange(len(noise_parts)), size=diff_int, replace=False
            )
            noise_parts[pick_idxs] += 1

        noised_idxs = self.random_gen.permutation(n_samples)[:n_outliers]

        y_noise = np.zeros(n_samples, dtype=int)
        y_noise[noised_idxs] = 1

        ## Apply Noising

        # get y_noise_lists
        y_noise_lists = [[] for _ in range(self.n_noises)]
        offset = 0
        for noise_classes, part_size in zip(self.total_noise_list, noise_parts):
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

        # insert noise patterns to image --> straight assignment to pixels (replaced)
        for noise_tag, _cur_noised_idxs in enumerate(y_noise_lists):
            _x_idxs = self.idxs_noise_bases[noise_tag]["x"]
            _y_idxs = self.idxs_noise_bases[noise_tag]["y"]
            _pixel_val = self.pixel_vals_bases[noise_tag]

            self._fill_pixels(
                dirtied_imgs, _cur_noised_idxs, _x_idxs, _y_idxs, _pixel_val
            )

        # get y_noise_lists_combs (use after is optional)
        # -> register combinations of noise patterns as (unique) labels, e.g. use in stratisfied sampling.
        if self.combs_on:
            y_noise_lists_combs = [[] for _ in range(self.n_noise_partitions)]
            offset = 0
            for noise_tag, part_size in enumerate(noise_parts):
                y_noise_lists_combs[noise_tag].extend(
                    noised_idxs[offset : (offset + part_size)].tolist()
                )
                offset += part_size
        else:
            y_noise_lists_combs = None

        return dirtied_imgs, y_noise, y_noise_lists, y_noise_lists_combs

    def apply(self, X_gt):

        """ call func to noise X_gt; it returns noised copy of image data and related structs """

        X, y_noise, y_noise_lists, y_noise_lists_combs = self._add_noise(X_gt)

        return X, y_noise, y_noise_lists, y_noise_lists_combs
