from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X might not be a pandas DataFrame (e.g. a np.array)
        df = pd.DataFrame(X)

        # Compute and store the means/standard-deviations for each column (e.g. 'c1' and 'c2'),
        # for each group (e.g. 'A', 'B', 'C').
        # (Our solution uses a dictionary)
        self.groups = df[df.columns.tolist()[0]].unique().tolist()
        self.indizes = df.columns.tolist()[1:]
        self.grp_identifier = df.columns.tolist()[0]
        self.grps_ = df.groupby(self.grp_identifier).agg(["std", "mean"])
        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")

        # Hint: Define a helper function here!

        df = pd.DataFrame(X)
        for group in self.groups:
            for index in self.indizes:
                df.loc[df[self.grp_identifier] == group, index] = (df.loc[df[self.grp_identifier] == group, index] -
                                                                   self.grps_[index, "mean"][group]) / \
                                                                  self.grps_[index, "std"][group]
        return df.drop(self.grp_identifier, axis=1).fillna(0.0)