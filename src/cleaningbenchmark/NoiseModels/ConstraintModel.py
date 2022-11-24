"""
This module defines a collection of
constraint based errors
"""

import numpy
import pandas as pd
from . import NoiseModel
from pandasql import sqldf


def to_str_CFD_SQL(set_LHS, tuple_RHS, all_rule=True):

    cfd_condition_str = ""

    # go through LHS for current CFD
    for att_name, att_value in set_LHS:
        cfd_condition_str = cfd_condition_str + "[{}]='{}' and ".format(
            att_name, att_value
        )

    if all_rule:
        # go through RHS for current CFD
        cfd_condition_str = cfd_condition_str + "[{}]='{}'".format(
            tuple_RHS[0], tuple_RHS[1]
        )
    else:
        # LHS only (without RHS of CFD)
        cfd_condition_str = cfd_condition_str[:-5]

    return cfd_condition_str


class RowConstraint(object):
    def __init__(self, name):
        self.name = name

    def corrupt(self, row):
        """
    This function should be overwritten
    """
        return row


class FDRowConstraint(object):
    def __init__(self, indep_cols, dep_col):
        """
    indep_cols is the list of cols on the left side of the functional dep
    dep_col is the right side of the functional dep
    """
        self.indep_cols = indep_cols
        self.dep_cols = dep_cols


class cCFDRowConstraint(object):
    """
    LHS is the left hand side of the constant conditional functional dep (indep. cols)
    RHS is the right hand side of the constant conditional functional dep (dep. cols)
    """

    def __init__(self, LHS_set, RHS_tuple, support=-10, conf=-10):

        # set of tuples for LHS (attribute_name, attribute_value)
        self.LHS = LHS_set
        # tuple for RHS, attribute=tuple[0] and value=tuple[1]
        self.RHS = RHS_tuple
        # support value for the CFD (database definition, in a format ratio 0.XX)
        self.support = support
        # confidence (for approximate CFD rules only, otherwise 1)
        self.confidence = conf


class FDNoiseModel(NoiseModel.NoiseModel):
    """
  fd is the list of functional dependencies
  """

    def __init__(self, shape, probability=0, feature_importance=[], fds=[]):

        super(FDNoiseModel, self).__init__(shape, probability, feature_importance, True)
        self.k = k
        self.p = p

    def corrupt(self, X):
        hvfeature = self.feature_importance[0]
        means = numpy.mean(X, axis=0)
        Ns = numpy.shape(X)[0]
        ps = numpy.shape(X)[1]
        Y = X

        for i in numpy.argsort(X[:, hvfeature]):
            if numpy.random.rand(1, 1) < self.p:
                a = numpy.random.choice(self.feature_importance[0 : self.k], 1)
                Y[i, a] = means[a]

        return Y


class cCFDNoiseModel(NoiseModel.NoiseModel):
    """
  ccfd is the list of constant conditional functional dependencies.

  This function only runs with Pandas DataFrame.
  """

    def __init__(
        self,
        shape,
        categories_dict,
        probability=0,
        feature_importance=[],
        ccfds=[],
        p_row=0.01,
        alpha_prob=1.0,
    ):

        super(cCFDNoiseModel, self).__init__(
            shape, probability, feature_importance, True
        )

        self.p = p_row  # probability of row being used
        self.ccfds = ccfds  # list of constant CFDs to be broken
        self.categories_dict = categories_dict
        self.alpha_prob = alpha_prob

    def apply(self, X):

        """
    Must receive pandas DataFrame X, with proper definition of
    column data types.
    """

        return self.corrupt(X), X

    def corrupt(self, X):

        """
    Note that here X is a pandas DataFrame
    """

        # create new corrupted data
        Y = X.copy()

        # get means and standard deviations
        # (will be used as noise for the numericas, does not distort statistics)
        means_col = X.mean()
        stds_col = X.std()

        # add auxiliary index
        X["indexcol"] = X.index

        # break the cCFDs in the list
        for ccfd in self.ccfds:

            ## Get Rows which hold the cCFD constraint
            cfd_cond_str = to_str_CFD_SQL(ccfd.LHS, ccfd.RHS)
            sql_query = "SELECT {} FROM {} WHERE {};".format(
                "indexcol", "X", cfd_cond_str
            )
            df_res = sqldf(sql_query, locals())

            # Get categories and respective probabilities in the dataset,
            # if ccfd.RHS[0] feature is categorical
            if X[ccfd.RHS[0]].dtype.name == "category":
                cats = [
                    t for t in self.categories_dict[ccfd.RHS[0]] if t != ccfd.RHS[1]
                ]
                cats_probs = X[ccfd.RHS[0]].value_counts()[cats].values
                cats_probs = cats_probs ** self.alpha_prob / float(
                    cats_probs.sum() ** self.alpha_prob
                )

            ## Insert Right Hand Side Noise (to violate the constraint)
            for row_idx in df_res["indexcol"]:

                if numpy.random.rand() <= self.p:

                    # is categorical
                    if X[ccfd.RHS[0]].dtype.name == "category":
                        # choose other categories according to their proportion in the dataset
                        idx_cat = numpy.random.choice(len(cats), 1, False, cats_probs)[
                            0
                        ]
                        Y.set_value(row_idx, ccfd.RHS[0], cats[idx_cat])

                    # is integer
                    elif X[ccfd.RHS[0]].dtype.name in ["int16", "int32", "int64"]:
                        # noise the cell using the mean of column (with a fraction of the standard deviation)
                        Y.set_value(
                            row_idx,
                            ccfd.RHS[0],
                            int(means_col[ccfd.RHS[0]] + 0.10 * stds_col[ccfd.RHS[0]]),
                        )

                    # is float
                    elif X[ccfd.RHS[0]].dtype.name in ["float16", "float32", "float64"]:
                        # noise the cell using the mean of column (with a fraction of the standard deviation)
                        Y.set_value(
                            row_idx,
                            ccfd.RHS[0],
                            float(
                                means_col[ccfd.RHS[0]] + 0.10 * stds_col[ccfd.RHS[0]]
                            ),
                        )

                    # Add Typo if none of above
                    else:
                        # noise the cell using standard typo (e.g. unique/rare)
                        Y.set_value(
                            row_idx, ccfd.RHS[0], NoiseModel.generate_typo(ccfd.RHS[1])
                        )

        # Testing
        # for ccfd in self.ccfds:
        #  # Get Rows which hold the cCFD constraint
        #  cfd_cond_str = to_str_CFD_SQL(ccfd.LHS, ccfd.RHS)
        #  sql_query = "SELECT {} FROM {} WHERE {};".format("count(*)", "Y", cfd_cond_str)
        #  df_res = sqldf(sql_query, locals())
        #  print df_res

        # drop auxiliary index
        X.drop("indexcol", axis=1, inplace=True)

        # cast category to object (since adding noise, may change the definition of categories available to feature)
        Y[self.categories_dict.keys()] = Y[self.categories_dict.keys()].apply(
            lambda x: x.astype("object")
        )

        return Y

