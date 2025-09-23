"""Main Module of arcos4py.

This module contains the ARCOS class, which implements most functionallity of arcos4py
to prepare data and to detect and track collective events.

Example:
    >>> from arcos4py import ARCOS
    >>> ts = ARCOS(data,["x"], 'time', 'id', 'meas', 'clTrackID')
    >>> ts.interpolate_measurements()
    >>> ts.clip_meas(clip_low = 0.001, clip_high=0.999)
    >>> ts.bin_measurements(
            smooth_k = 3,
            bias_k = 51,
            peak_threshold = 0.2,
            binarization_threshold = 0.1,
            polynomial_degree = 1,
            bias_method = "runmed")
    >>> events_df = ts.trackCollev(eps = 1, min_clustersize = 1, n_prev = 1)
"""

from __future__ import annotations

import warnings
from typing import Union

import pandas as pd

from .tools._arcos4py_deprecation import handle_deprecated_params
from .tools._binarize_detrend import binData
from .tools._cleandata import clipMeas, interpolation
from .tools._detect_events import track_events_dataframe


class ARCOS:
    """Detects and tracks collective events in a tracked time-series dataset.

    Requires binarized measurement column, that can be generated with the
    bin_measurements method.
    Tracking makes use of the dbscan algorithm, which is applied to every frame
    and subsequently connects collective events between frames located
    within eps distance of each other.

    Attributes:
        data (DataFrame): Data of tracked time-series in "long format". Can be used to
            acess modified dataframe at any point.
        position_columns (list): List containing position column names strings inside data e.g.
            At least one dimension is required.
        frame_column (str): Indicating the frame column in input_data.
        obj_id_column (str): Indicating the track id/id column in input_data.
        measurement_column (str): Indicating the measurement column in input_data.
        clid_column (str): Indicating the column name containing the collective event ids.
        binarized_measurement_column (str | None): Name of the binary column.
            This is generated based on the name of the measurement_column after binarization.
            Optionally can be set in order to provide a already binarized column to skip ARCOS binarization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        position_columns: list = ["x"],
        frame_column: str = 'time',
        obj_id_column: str | None = 'id',
        measurement_column: str = 'meas',
        clid_column: str = 'clTrackID',
        n_jobs: int = 1,
        **kwargs,
    ) -> None:
        """Constructs class with provided arguments.

        Arguments:
            data (DataFrame): Input Data of tracked time-series in "long format" containing position columns,
                a measurement and an object ID column.
            position_columns (list): List ontaining position column names strings inside data e.g.
                At least one dimension is required.
            frame_column (str): Indicating the frame column in input_data.
            obj_id_column (str): Indicating the track id/object id column in input_data. If None, the data is assumed to
                not have a tracking column. Binarization can only be performed without detrending.
            measurement_column (str): Indicating the measurement column in input_data.
            clid_column (str): Indicating the column name containing the collective event ids.
            n_jobs (str): Number of workers to spawn, -1 uses all available cpus.
            kwargs (Any): Additional keyword arguments. Includes old parameter names for backwards compatibility.
                - posCols: List containing position column names strings inside data e.g.
        """
        # allowed kwargs
        allowed_kwargs = ["posCols", "id_column"]
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"__init__() got an unexpected keyword argument '{key}'")
        # Handle deprecated parameters
        param_mapping = {
            "posCols": "position_columns",
            "id_column": "obj_id_column",
        }
        updated_kwargs = handle_deprecated_params(param_mapping, **kwargs)

        # Assign updated kwargs to class attributes
        position_columns = updated_kwargs.get("position_columns", position_columns)
        obj_id_column = updated_kwargs.get("obj_id_column", obj_id_column)

        self.data = data
        self.position_columns = position_columns
        self.frame_column = frame_column
        self.obj_id_column = obj_id_column
        self.measurement_column = measurement_column
        self.clid_column = clid_column
        self.n_jobs = n_jobs

        self.binarized_measurement_column: Union[str, None] = None
        # to check if no measurement was provided assign None
        if self.obj_id_column is None:
            self.data = self.data.sort_values(by=[self.frame_column])
        else:
            self.data = self.data.sort_values(by=[self.frame_column, self.obj_id_column])
        self._check_col()
        if self.measurement_column is not None:
            self.resc_col = f"{self.measurement_column}.resc"
            self.binarized_measurement_column = f"{self.measurement_column}.bin"

    def __repr__(self) -> pd.DataFrame:
        """Set __repr__ to return self.data."""
        return repr(self.data)

    def _check_col(self):
        """Checks that self.cols contains all required columns."""
        columns = self.data.columns
        input_columns = [
            self.frame_column,
            self.obj_id_column,
            self.measurement_column,
        ]
        input_columns = [col for col in input_columns if col is not None]
        if not all(item in columns for item in input_columns):
            raise ValueError(f"Columns {input_columns} do not match with column in dataframe.")

    def interpolate_measurements(self) -> pd.DataFrame:
        """Interpolates NaN's in place in measurement column.

        Returns:
            Dataframe with interpolated measurement column.
        """
        meas_interp = interpolation(self.data).interpolate()
        self.data = meas_interp
        return self.data

    def clip_measurements(self, clip_low: float = 0.001, clip_high: float = 0.999) -> pd.DataFrame:
        """Clip measurement column to upper and lower quantiles defined in clip_low and clip_high.

        Arguments:
            clip_low (float): Lower clipping boundary (quantile).

            clip_high (float): Upper clipping boundary (quantile).

        Returns:
            Dataframe with in place clipped measurement column.
        """
        meas_column = self.data[self.measurement_column].to_numpy()
        meas_clipped = clipMeas(meas_column).clip(clip_low, clip_high)
        self.data[self.measurement_column] = meas_clipped
        return self.data

    def clip_meas(self, clip_low: float = 0.001, clip_high: float = 0.999) -> pd.DataFrame:
        """Clip measurement column to upper and lower quantiles defined in clip_low and clip_high.

        Arguments:
            clip_low (float): Lower clipping boundary (quantile).

            clip_high (float): Upper clipping boundary (quantile).

        Returns:
            Dataframe with in place clipped measurement column.
        """
        # Issue a deprecation warning
        warnings.warn(
            "The 'clip_meas' method is deprecated and will be removed in a future version.\
            Please use 'clip_measurements' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.clip_measurements(clip_low, clip_high)

    def bin_measurements(
        self,
        smooth_k: int = 3,
        bias_k: int = 51,
        peak_threshold: float = 0.2,
        binarization_threshold: float = 0.1,
        polynomial_degree: int = 1,
        bias_method: str = "runmed",
        **kwargs,
    ) -> pd.DataFrame:
        r"""Smooth, de-trend, and binarise the input data.

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
        The final signal is binarised using the binThr threshold

        Arguments:
            smooth_k (int): Size of the short-term median smoothing filter.
            bias_k (int): Size of the long-term de-trending median filter
            peak_threshold (float): Threshold for rescaling of the de-trended signal.
            binarization_threshold (float): Threshold for binary classification.
            polynomial_degree (int): Sets the degree of the polynomial for lm fitting.
            bias_method (str): De-trending method, one of ['runmed', 'lm', 'none'].
                If no id_column is provided, only 'none' is allowed.
            **kwargs (Any): Additional keyword arguments. Includes old parameter names for backwards compatibility.
                - smoothK: Size of the short-term median smoothing filter.
                - biasK: Size of the long-term de-trending median filter
                - peakThr: Threshold for rescaling of the de-trended signal.
                - binThr: Threshold for binary classification.
                - polyDeg: Sets the degree of the polynomial for lm fitting.
                - biasMet: De-trending method, one of ['runmed', 'lm', 'none'].

        Returns:
            DataFrame with detrended/smoothed and binarized measurement column.
        """
        # allowed kwargs
        param_mapping = {
            "smoothK": "smooth_k",
            "biasK": "bias_k",
            "peakThr": "peak_threshold",
            "binThr": "binarization_threshold",
            "polyDeg": "polynomial_degree",
            "biasMet": "bias_method",
        }
        # allowed kwargs
        allowed_kwargs = param_mapping.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"bin_measurements() got an unexpected keyword argument '{key}'")

        updated_kwargs = handle_deprecated_params(param_mapping, **kwargs)

        smooth_k = updated_kwargs.get("smooth_k", smooth_k)
        bias_k = updated_kwargs.get("bias_k", bias_k)
        peak_threshold = updated_kwargs.get("peak_threshold", peak_threshold)
        binarization_threshold = updated_kwargs.get("binarization_threshold", binarization_threshold)
        polynomial_degree = updated_kwargs.get("polynomial_degree", polynomial_degree)
        bias_method = updated_kwargs.get("bias_method", bias_method)

        self.data = binData(
            smooth_k,
            bias_k,
            peak_threshold,
            binarization_threshold,
            polynomial_degree,
            bias_method,
            n_jobs=self.n_jobs,
        ).run(
            self.data,
            measurement_column=self.measurement_column,
            group_column=self.obj_id_column,
            frame_column=self.frame_column,
        )
        return self.data

    def trackCollev(
        self,
        eps: float = 1,
        eps_prev: Union[float, None] = None,
        min_clustersize: int = 1,
        n_prev: int = 1,
        clustering_method: str = "dbscan",
        linking_method: str = "nearest",
        min_samples: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Detects and tracks collective events in a tracked time-series dataset.

        Makes use of the dbscan algorithm,
        applies this to every timeframe and subsequently connects
        collective events between frames located within eps distance of each other.

        Arguments:
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighbourhood of the other.
                This is not a maximum bound on the distances of points within a cluster.
            eps_prev (float | None): Frame to frame distance, value is used to connect
                collective events across multiple frames.If "None", same value as eps is used.
            min_clustersize (int): The minimum size for a cluster to be identified as a collective event
            n_prev (int): Number of previous frames the tracking
                algorithm looks back to connect collective events
            clustering_method (str): Clustering method, one of ['dbscan', 'hdbscan'].
            min_samples (int | None): The number of samples (or total weight) in a neighbourhood for a
                point to be considered as a core point. This includes the point itself.
                Only used if clustering_method is 'hdbscan'. If None, min_samples =  min_clustersize.
            linking_method (str): Linking method, one of ['nearest', 'transportation'].
            **kwargs (Any): Additional keyword arguments. Includes old parameter names for backwards compatibility.
                - epsPrev: Frame to frame distance, value is used to connect
                    collective events across multiple frames.
                - minClsz: The minimum size for a cluster to be identified as a collective event
                - nPrev: Number of previous frames the tracking
                    algorithm looks back to connect collective events
                - clusteringMethod: Clustering method, one of ['dbscan', 'hdbscan'].
                - minSamples: The number of samples (or total weight) in a neighbourhood for a
                    point to be considered as a core point. This includes the point itself.
                    Only used if clustering_method is 'hdbscan'. If None, min_samples =  min_clustersize.
                - linkingMethod: Linking method, one of ['nearest', 'transportation'].

        Returns:
            DataFrame with detected collective events across time.
        """
        warnings.warn(
            "The 'trackCollev' method is deprecated and will be removed in a future version.\
                Please use 'track_collective_events' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.track_collective_events(
            eps, eps_prev, min_clustersize, n_prev, clustering_method, linking_method, min_samples, **kwargs
        )

    def track_collective_events(
        self,
        eps: float = 1,
        eps_prev: Union[float, None] = None,
        min_clustersize: int = 1,
        n_prev: int = 1,
        clustering_method: str = "dbscan",
        linking_method: str = "nearest",
        min_samples: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Detects and tracks collective events in a tracked time-series dataset.

        Makes use of the dbscan algorithm,
        applies this to every timeframe and subsequently connects
        collective events between frames located within eps distance of each other.

        Arguments:
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighbourhood of the other.
                This is not a maximum bound on the distances of points within a cluster.
            eps_prev (float | None): Frame to frame distance, value is used to connect
                collective events across multiple frames.If "None", same value as eps is used.
            min_clustersize (int): The minimum size for a cluster to be identified as a collective event
            n_prev (int): Number of previous frames the tracking
                algorithm looks back to connect collective events
            clustering_method (str): Clustering method, one of ['dbscan', 'hdbscan'].
            min_samples (int | None): The number of samples (or total weight) in a neighbourhood for a
                point to be considered as a core point. This includes the point itself.
                Only used if clustering_method is 'hdbscan'. If None, min_samples =  min_clustersize.
            linking_method (str): Linking method, one of ['nearest', 'transportation'].
            **kwargs (Any): Additional keyword arguments. Includes old parameter names for backwards compatibility.
                - epsPrev: Frame to frame distance, value is used to connect
                    collective events across multiple frames.
                - minClsz: The minimum size for a cluster to be identified as a collective event
                - nPrev: Number of previous frames the tracking
                    algorithm looks back to connect collective events
                - clusteringMethod: Clustering method, one of ['dbscan', 'hdbscan'].
                - minSamples: The number of samples (or total weight) in a neighbourhood for a
                    point to be considered as a core point. This includes the point itself.
                    Only used if clustering_method is 'hdbscan'. If None, min_samples =  min_clustersize.
                - linkingMethod: Linking method, one of ['nearest', 'transportation'].

        Returns:
            DataFrame with detected collective events across time.
        """
        param_mapping = {
            "epsPrev": "eps_prev",
            "minClsz": "min_clustersize",
            "nPrev": "n_prev",
            "clusteringMethod": "clustering_method",
            "minSamples": "min_samples",
            "linkingMethod": "linking_method",
        }
        # allowed kwargs
        allowed_kwargs = param_mapping.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"track_collective_events() got an unexpected keyword argument '{key}'")
        updated_kwargs = handle_deprecated_params(param_mapping, **kwargs)

        eps_prev = updated_kwargs.get("eps_prev", eps_prev)
        min_clustersize = updated_kwargs.get("min_clustersize", min_clustersize)
        n_prev = updated_kwargs.get("n_prev", n_prev)
        clustering_method = updated_kwargs.get("clustering_method", clustering_method)
        min_samples = updated_kwargs.get("min_samples", min_samples)
        linking_method = updated_kwargs.get("linking_method", linking_method)

        data_events = track_events_dataframe(
            X=self.data,
            position_columns=self.position_columns,
            frame_column=self.frame_column,
            id_column=self.obj_id_column,
            binarized_measurement_column=self.binarized_measurement_column,
            eps=eps,
            eps_prev=eps_prev,
            min_clustersize=min_clustersize,
            n_prev=n_prev,
            clid_column=self.clid_column,
            linking_method=linking_method,
            clustering_method=clustering_method,
            min_samples=min_samples,
            n_jobs=self.n_jobs,
        )

        return data_events

    @property
    def bin_col(self) -> str | None:
        """Return the name of the binarized measurement column."""
        # Issue a deprecation warning
        warnings.warn(
            "The 'bin_col' attribute is deprecated and will be removed in a future version.\
            Please use 'binarized_measurement_column' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.binarized_measurement_column

    @bin_col.setter
    def bin_col(self, value: str | None) -> None:
        """Set the name of the binarized measurement column."""
        # Issue a deprecation warning
        warnings.warn(
            "The 'bin_col' attribute is deprecated and will be removed in a future version.\
            Please use 'binarized_measurement_column' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.binarized_measurement_column = value

    @property
    def posCols(self) -> list:
        """Return the position columns."""
        # Issue a deprecation warning
        warnings.warn(
            "The 'posCols' attribute is deprecated and will be removed in a future version.\
            Please use 'position_columns' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.position_columns

    @posCols.setter
    def posCols(self, value: list) -> None:
        """Set the position columns."""
        # Issue a deprecation warning
        warnings.warn(
            "The 'posCols' attribute is deprecated and will be removed in a future version.\
            Please use 'position_columns' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.position_columns = value

    @property
    def id_column(self) -> str | None:
        """Return the name of the id column."""
        # Issue a deprecation warning
        warnings.warn(
            "The 'id_column' attribute is deprecated and will be removed in a future version.\
            Please use 'obj_id_column' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.obj_id_column

    @id_column.setter
    def id_column(self, value: str | None) -> None:
        """Set the name of the id column."""
        # Issue a deprecation warning
        warnings.warn(
            "The 'id_column' attribute is deprecated and will be removed in a future version.\
            Please use 'obj_id_column' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.obj_id_column = value
