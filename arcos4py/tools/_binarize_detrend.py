"""Module containing binarization and detrending classes.

Example:
    >>> from arcos4py.tools import binData
    >>> binarizer = binData(biasMet="lm", polyDeg=1)
    >>> data_rescaled = binarizer.run(data, colMeas="ERK_KTR", colGroup="trackID")
"""

from __future__ import annotations

from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import median_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, minmax_scale

from ..tools._arcos4py_deprecation import handle_deprecated_params


class detrender:
    """Smooth and de-trend input data.

    First, a short-term median filter with size smooth_k is applied
    to remove fast noise from the time series. smooth_k is applied
    The subsequent de-trending can be performed with a long-term median filter
    with the size bias_k {bias_method = "runmed"}
    or by fitting a polynomial of degree polynomial_degree {bias_method = "lm"}.

    Attributes:
        smooth_k (int): Representing the size of the short-term median smoothing filter.
        bias_k (int): Representing the size of the long-term de-trending median filter.
        peak_threshold (float): Threshold for rescaling of the de-trended signal.
        polynomial_degree (int): Sets the degree of the polynomial for lm fitting..
        bias_method (str): Indicating de-trending method, one of ['runmed', 'lm', 'none'].
        n_jobs (int): Number of paralell workers to spawn, -1 uses all available cpus.biasMet (str):
            Indicating de-trending method, one of ['runmed', 'lm', 'none'].
    """

    def __init__(
        self,
        smooth_k: int = 3,
        bias_k: int = 51,
        peak_threshold: float = 0.2,
        polynomial_degree: int = 1,
        bias_method: str = "runmed",
        n_jobs: int = 1,
        **kwargs,
    ) -> None:
        """Smooth and de-trend input data.

        Arguments:
            smooth_k (int): Representing the size of the short-term median smoothing filter.
            bias_k (int): Representing the size of the long-term de-trending median filter.ian filter.
            peak_threshold (float): Threshold for rescaling of the de-trended signal.
            polynomial_degree (int): Sets the degree of the polynomial for lm fitting.
            bias_method (str): Indicating de-trending method, one of ['runmed', 'lm', 'none'].
            n_jobs (int): Number of parallel workers to spawn, -1 uses all available CPUs.
        """
        # handle deprecated parameters
        param_mapping = {
            "smoothK": "smooth_k",
            "biasK": "bias_k",
            "peakThr": "peak_threshold",
            "polyDeg": "polynomial_degree",
            "biasMet": "bias_method",
        }
        # allowed_kwargs
        allowed_kwargs = param_mapping.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")
        updated_kwargs = handle_deprecated_params(param_mapping, **kwargs)

        # update the parameters
        smooth_k = updated_kwargs.get("smooth_k", smooth_k)
        bias_k = updated_kwargs.get("bias_k", bias_k)
        peak_threshold = updated_kwargs.get("peak_threshold", peak_threshold)
        polynomial_degree = updated_kwargs.get("polynomial_degree", polynomial_degree)
        bias_method = updated_kwargs.get("bias_method", bias_method)

        # check if biasmethod contains one of these three types
        biasMet_types = ["runmed", "lm", "none"]
        if bias_method not in biasMet_types:
            raise ValueError(f"Invalid bias method. Expected one of: {biasMet_types}")

        self.smooth_k = smooth_k
        self.bias_k = bias_k
        self.peak_threshold = peak_threshold
        self.polynomial_degree = polynomial_degree
        self.bias_method = bias_method
        self.n_jobs = n_jobs

    def _detrend_runnmed(self, x, filter_size, endrule_mode):
        local_smoothing = median_filter(input=x, size=filter_size, mode=endrule_mode)
        return local_smoothing

    # solution a bit janky but only median filter with changing window size?
    def _detrend_global_runmed(self, x, filter_size):
        x_series = pd.Series(x)
        global_smoothing = x_series.rolling(filter_size, min_periods=1).median().to_numpy()
        return global_smoothing

    def _detrend_lm(self, x, polynomial_degree):
        y = np.linspace(1, x.size, x.size).astype(int).reshape((-1, 1))
        transformer = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        data_ = transformer.fit_transform(y)
        model = LinearRegression().fit(X=data_, y=x)
        predicted_value = model.predict(data_)
        return predicted_value

    def _run_detrend(self, x: np.ndarray) -> np.ndarray:
        if x.size:
            local_smoothed = self._detrend_runnmed(x, self.smooth_k, "nearest")
            if self.bias_method != "none":
                if self.bias_method == "runmed":
                    global_smoothed = self._detrend_global_runmed(
                        x=local_smoothed,
                        filter_size=self.bias_k,
                    )
                elif self.bias_method == "lm":
                    global_smoothed = self._detrend_lm(local_smoothed, self.polynomial_degree)

                local_smoothed = np.subtract(local_smoothed, global_smoothed)
                local_smoothed = np.clip(local_smoothed, 0, None)
                if (local_smoothed.max() - local_smoothed.min()) > self.peak_threshold:
                    local_smoothed = np.divide(local_smoothed, local_smoothed.max())
                local_smoothed = np.nan_to_num(local_smoothed)
        else:
            local_smoothed = np.array([])

        return local_smoothed

    def detrend(self, x: np.ndarray, group_index: int, meas_index: int) -> np.ndarray:
        """Run detrinding on input data.

        The method applies detrending to each group defined in group_col and
        outputs it into the resc_column.

        Arguments:
            x (np.ndarray): Time series data for smoothing.
            group_index (int | None): Index of id column in x.
            meas_index (int): Index of measurement column in x.

        Returns (np.ndarray): Dataframe containing rescaled column.
        """
        meas_array = x[:, meas_index].astype('float64')
        try:
            group_array = x[:, group_index].astype('int64')
        except ValueError:
            try:
                group_array = x[:, group_index].astype('float64')
            except ValueError:
                group_array = x[:, group_index].astype('U6')

        grouped_array = np.split(meas_array, np.unique(group_array, axis=0, return_index=True)[1][1:])
        out = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_detrend)(
                x=x,
            )
            for x in grouped_array
        )
        out_list = [item for sublist in out for item in sublist]
        return np.array(out_list)


class binData(detrender):
    """Smooth, de-trend, and binarise the input data.

    First a short-term median filter with size smoothK
    is applied to remove fast noise from the time series.
    If the de-trending method is set to "none",
    smoothing is applied on globally rescaled time series.
    The subsequent de-trending can be performed with a long-term median filter
    with the size biasK {biasMet = "runmed"}
    or by fitting a polynomial of degree polyDeg {biasMet = "lm"}.

    After de-trending,
    if the global difference between min/max is greater than the threshold
    the signal is rescaled to the (0,1) range.
    The final signal is binarised using the binThr threshold.

    Attributes:
        smoothK (int): Size of the short-term median smoothing filter.
        biasK (int): Size of the long-term de-trending median filter.
        peakThr (float): Threshold for rescaling of the de-trended signal.
        binThr (float): Threshold for binarizing the de-trended signal.
        polyDeg (int): Sets the degree of the polynomial for lm fitting.
        biasMet (str): De-trending method, one of ['runmed', 'lm', 'none'].
    """

    def __init__(
        self,
        smooth_k: int = 3,
        bias_k: int = 51,
        peak_threshold: float = 0.2,
        binarization_threshold: float = 0.1,
        polynomial_degree: int = 1,
        bias_method: str = "runmed",
        n_jobs: int = 1,
        **kwargs,
    ) -> None:
        """Smooth, de-trend, and binarise the input data.

        Arguments:
            smooth_k (int): Size of the short-term median smoothing filter.
            bias_k (int): Size of the long-term de-trending median filter.
            peak_threshold (float): Threshold for rescaling of the de-trended signal.
            binarization_threshold (float): Threshold for binarizing the de-trended signal.
            polynomial_degree (int): Sets the degree of the polynomial for lm fitting.
            bias_method (str): De-trending method, one of ['runmed', 'lm', 'none'].
            n_jobs (int): Number of jobs to run in parallel.
        """
        super().__init__(smooth_k, bias_k, peak_threshold, polynomial_degree, bias_method, n_jobs, **kwargs)
        self.binarization_threshold = binarization_threshold

    def _rescale_data(self, x: np.ndarray, measurement_index: int, feature_range: tuple = (0, 1)) -> np.ndarray:
        """Rescale data to a given range."""
        meas_array = x[:, measurement_index].astype('float64')
        rescaled = minmax_scale(meas_array, feature_range=feature_range)
        x[:, measurement_index] = rescaled
        return x

    def _bin_data(self, x: np.ndarray) -> np.ndarray:
        bin = (x >= self.binarization_threshold).astype(np.int_)
        return bin

    def run(
        self, x: pd.DataFrame, group_column: str | None, measurement_column: str, frame_column: str, **kwargs
    ) -> pd.DataFrame:
        """Runs binarization and detrending.

        If the bias_method is 'none', first it rescales the data to between [0,1], then
        local smoothing is applied to the measurement by groups, followed by
        binarization.

        If bias_method is one of ['lm', 'runmed'], first the data is detrended locally with a
        median filter and then detrended globally, for 'lm' with a linear model and for 'runmed' with a
        median filter.
        Followed by binarization of the data.

        Arguments:
            x (DataFrame): The time-series data for smoothing, detrending and binarization.
            group_column (str | None): Object id column in x. Detrending and rescaling is performed on a per-object basis.
                If None, no detrending is performed, only rescaling and bias method is ignored.
            measurement_column (str): Measurement column in x on which detrending and rescaling is performed.
            frame_column (str): Frame column in Time-series data. Used for sorting.
            **kwargs (Any): Additional keyword arguments. Includes old parameters for backwards compatibility.
                - GroupCol (str): Object id column in x. Detrending and rescaling is performed on a per-object basis.
                - colMeas (str): Measurement column in x on which detrending and rescaling is performed.
                - colFrame (str): Frame column in Time-series data. Used for sorting.

        Returns:
            DataFrame: Dataframe containing binarized data, rescaled data and the original columns.
        """
        # handle deprecated parameters
        param_mapping = {
            "GroupCol": "group_column",
            "colMeas": "measurement_column",
            "colFrame": "frame_column",
        }
        # allowed_kwargs
        allowed_kwargs = param_mapping.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Invalid keyword argument: {key}")
        updated_kwargs = handle_deprecated_params(param_mapping, **kwargs)

        # update the parameters
        group_column = updated_kwargs.get("group_column", group_column)
        measurement_column = updated_kwargs.get("measurement_column", measurement_column)
        frame_column = updated_kwargs.get("frame_column", frame_column)

        if group_column is None:
            return self._run_without_groupcol(x, measurement_column, frame_column)
        else:
            return self._run_with_groupcol(x, group_column, measurement_column, frame_column)

    def _run_without_groupcol(self, x, measurement_column, frame_column):
        col_resc = f"{measurement_column}.resc"
        col_bin = f"{measurement_column}.bin"
        cols = [measurement_column]
        x = x.sort_values(frame_column)
        data_np = x[cols].to_numpy()

        if self.bias_method == "none":
            rescaled_data = self._rescale_data(data_np, measurement_index=0)
            binarized_data = self._bin_data(rescaled_data)
        else:
            warn("No detrending is performed, only rescaling. To run detrending, set colGroup.")
            rescaled_data = self._rescale_data(data_np, measurement_index=0)
            binarized_data = self._bin_data(rescaled_data)

        x[col_resc] = rescaled_data[:, 0]
        x[col_bin] = binarized_data[:, 0]
        return x

    def _run_with_groupcol(self, x, group_column, measurement_column, frame_column):
        col_resc = f"{measurement_column}.resc"
        col_bin = f"{measurement_column}.bin"
        col_fact = f'{group_column}_factorized'
        cols = [col_fact, measurement_column]
        x = x.sort_values([group_column, frame_column])
        # factorize column in order to prevent numpy grouping error in detrending
        value, _ = x[group_column].factorize()
        x[col_fact] = value

        data_np = x[cols].to_numpy()

        if self.bias_method == "none":
            rescaled_data = self._rescale_data(data_np, measurement_index=1)
            detrended_data = self.detrend(rescaled_data, 0, 1)
            binarized_data = self._bin_data(detrended_data)
        else:
            detrended_data = self.detrend(data_np, group_index=0, meas_index=1)
            binarized_data = self._bin_data(detrended_data)

        x = x.drop([col_fact], axis=1)
        x[col_resc] = detrended_data
        x[col_bin] = binarized_data
        return x
