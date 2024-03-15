"""Tools for resampling data and bootstrapping."""

from __future__ import annotations

from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from arcos4py import ARCOS
from arcos4py.tools import calcCollevStats, filterCollev

from ..tools._arcos4py_deprecation import handle_deprecated_params
from ._resampling import resample_data


def bootstrap_arcos(
    df: pd.DataFrame,
    position_columns: list,
    frame_column: str,
    obj_id_column: str,
    measurement_column: str,
    method: str | list[str] = 'shuffle_tracks',
    smooth_k: int = 3,
    bias_k: int = 51,
    peak_threshold: float = 0.2,
    binarization_threshold: float = 0.1,
    polynomial_degree: int = 1,
    bias_method: str = "runmed",
    eps: float = 2,
    eps_prev: int | None = None,
    min_clustersize: int = 1,
    n_prev: int = 1,
    min_duration: int = 1,
    min_total_size: int = 1,
    stats_metric: str | list[str] = ["total_size", "duration"],
    pval_alternative: str = "greater",
    finite_correction: bool = True,
    n: int = 100,
    seed: int = 42,
    allow_duplicates: bool = False,
    max_tries: int = 100,
    show_progress: bool = True,
    verbose: bool = False,
    parallel_processing: bool = True,
    plot: bool = True,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap data using the ARCOS algorithm.

    Arguments:
        df: DataFrame containing the data to be bootstrapped.
        position_columns: List of column names containing the x and y coordinates.
        frame_column: Name of the column containing the frame number.
        obj_id_column: Name of the column containing the track id.
        measurement_column: Name of the column containing the measurement.
        method: Method used for bootstrapping. Can be "shuffle_tracks", 'shuffle_timepoints', 'shift_timepoints',
            'shuffle_binary_blocks', 'shuffle_coordinates_timepoint or a list of methods,
            which will be applied in order of index.
        smooth_k: Smoothing kernel size.
        bias_k: Bias kernel size.
        peak_threshold: Threshold for peak detection.
        binarization_threshold: Threshold for binarization.
        polynomial_degree: Degree of the polynomial used for bias correction.
        bias_method: Bias correction method. Can be 'none', 'runmed', 'lm'
        eps: Epsilon parameter for DBSCAN.
        eps_prev: Parameter for linking tracks. If None, eps is used.
        min_clustersize: Minimum cluster size.
        n_prev: Number of previous frames to consider for linking.
        min_duration: Minimum duration of a track.
        min_total_size: Minimum size of a track.
        stats_metric: Metric to calculate. Can be "duration", "total_size", "min_size", "max_size" or a list of metrics.
            Default is ["duration", "total_size"].
        pval_alternative: Alternative hypothesis for the p-value calculation. Can be "less" or "greater".
        finite_correction: Correct p-values for finite sampling. Default is True.
        n: Number of bootstraps.
        seed: Seed for the random number generator.
        allow_duplicates: If False, resampling will check if the resampled data contains duplicates.
            If True, duplicates will be allowed.
        max_tries: Maximum number of tries to resample data without duplicates.
        show_progress: Show a progress bar.
        verbose: Print additional information.
        parallel_processing: Use parallel processing.
        plot: Plot the distribution of the bootstrapped data.
        **kwargs (Any): Additional keyword arguments. Includes deprecated parameters.
            - id_column: Deprecated. Use obj_id_column instead.
            - meas_column: Deprecated. Use measurement_column instead.
            - smoothK: Deprecated. Use smooth_k instead.
            - biasK: Deprecated. Use bias_k instead.
            - peakThr: Deprecated. Use peak_threshold instead.
            - binThr: Deprecated. Use binarization_threshold instead.
            - polyDeg: Deprecated. Use polynomial_degree instead.
            - biasMet: Deprecated. Use bias_method instead.
            - epsPrev: Deprecated. Use eps_prev instead.
            - minClsz: Deprecated. Use min_clustersize instead.
            - min_size: Deprecated. Use min_total_size instead.
            - paralell_processing: Deprecated. Use parallel_processing instead.

    Returns:
        DataFrame containing the bootstrapped data.
    """
    map_deprecated_params = {
        "id_column": "obj_id_column",
        "meas_column": "measurement_column",
        "smoothK": "smooth_k",
        "biasK": "bias_k",
        "peakThr": "peak_threshold",
        "binThr": "binarization_threshold",
        "polyDeg": "polynomial_degree",
        "biasMet": "bias_method",
        "epsPrev": "eps_prev",
        "minClsz": "min_clustersize",
        "min_size": "min_total_size",
        "paralell_processing": "parallel_processing",
    }

    # check allowed kwargs
    allowed_kwargs = map_deprecated_params.keys()
    for key in kwargs:
        if key not in allowed_kwargs:
            raise ValueError(f"Got an unexpected keyword argument '{key}'")

    updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

    # Assigning the parameters
    obj_id_column = updated_kwargs.get("obj_id_column", obj_id_column)
    measurement_column = updated_kwargs.get("measurement_column", measurement_column)
    smooth_k = updated_kwargs.get("smooth_k", smooth_k)
    bias_k = updated_kwargs.get("bias_k", bias_k)
    peak_threshold = updated_kwargs.get("peak_threshold", peak_threshold)
    binarization_threshold = updated_kwargs.get("binarization_threshold", binarization_threshold)
    polynomial_degree = updated_kwargs.get("polynomial_degree", polynomial_degree)
    bias_method = updated_kwargs.get("bias_method", bias_method)
    eps_prev = updated_kwargs.get("eps_prev", eps_prev)
    min_clustersize = updated_kwargs.get("min_clustersize", min_clustersize)
    min_total_size = updated_kwargs.get("min_total_size", min_total_size)
    parallel_processing = updated_kwargs.get("parallel_processing", parallel_processing)

    if not isinstance(stats_metric, list):
        stats_metric = [stats_metric]

    for stats_m in stats_metric:
        if stats_m not in [
            "duration",
            "total_size",
            "min_size",
            "max_size",
        ]:
            raise ValueError(f"Invalid metric: {stats_metric}")

    if pval_alternative not in ["less", "greater"]:
        raise ValueError(f"Invalid alternative hypothesis: {pval_alternative}")

    clid_name = 'clid'

    if isinstance(method, str):
        print(f'Resampling data using method "{method}"...')
    elif isinstance(method, list):
        print(f'Resampling data using methods "{method}"...')

    df_resampled = resample_data(
        data=df,
        position_columns=position_columns,
        frame_column=frame_column,
        obj_id_column=obj_id_column,
        measurement_column=measurement_column,
        method=method,
        n=n,
        seed=seed,
        allow_duplicates=allow_duplicates,
        max_tries=max_tries,
        show_progress=show_progress,
        verbose=verbose,
        parallel_processing=parallel_processing,
    )

    iterations = df_resampled['iteration'].unique()

    print(f'Running ARCOS and calculating "{stats_metric}"...')

    stats_df, stats_df_mean = calculate_arcos_stats(
        df_resampled=df_resampled,
        position_columns=position_columns,
        frame_column=frame_column,
        obj_id_column=obj_id_column,
        measurement_column=measurement_column,
        smooth_k=smooth_k,
        bias_k=bias_k,
        peak_threshold=peak_threshold,
        binarization_threshold=binarization_threshold,
        polynomial_degree=polynomial_degree,
        bias_method=bias_method,
        eps=eps,
        eps_prev=eps_prev,
        min_clustersize=min_clustersize,
        n_prev=n_prev,
        min_duration=min_duration,
        min_total_size=min_total_size,
        stats_metric=stats_metric,
        show_progress=show_progress,
        parallel_processing=parallel_processing,
        clid_column=clid_name,
        iterations=iterations,
    )
    df_p = calculate_pvalue(stats_df_mean, stats_metric, pval_alternative, finite_correction, plot)
    return stats_df, df_p


def calculate_pvalue(
    stats_df_mean: pd.DataFrame,
    stats_metric: str | list[str],
    pval_alternative: str,
    finite_correction: bool,
    plot: bool,
    **plot_kwargs,
):
    """Calculates the p-value with the given alternative hypothesis.

    Arguments:
        stats_df_mean (DataFrame): DataFrame containing the bootstrapped data.
        stats_metric (str | list[str]): Metric to calculate.
            Can be "duration", "total_size", "min_size", "max_size" or a list of metrics.
            Default is ["duration", "total_size"].
        pval_alternative (str): Alternative hypothesis for the p-value calculation.
            Can be "less", "greater" or both which will return p values for both alternatives.
        finite_correction (bool): Correct p-values for finite sampling. Default is True.
        plot (bool): Plot the distribution of the bootstrapped data.

    Returns:
        DataFrame (pd.DataFrame): containing the p-values.
    """
    if finite_correction:
        pval = stats_df_mean[stats_metric].agg(lambda x: _p_val_finite_sampling(x, pval_alternative))
    else:
        pval = stats_df_mean[stats_metric].agg(lambda x: _p_val_infinite_sampling(x, pval_alternative))
    pval.name = 'p_value'

    if isinstance(stats_metric, list):
        _stats_metric = stats_metric
    else:
        _stats_metric = [stats_metric]

    mean_control = stats_df_mean[stats_metric].iloc[0]
    stats_df_mean = stats_df_mean[stats_df_mean['bootstrap_iteration'] != 0].reset_index(drop=True)

    if plot:
        fig, axis = plt.subplots(1, len(_stats_metric))
        try:
            iter(axis)
        except TypeError:
            axis = [axis]
        for idx, (ax, stats_col) in enumerate(zip(axis, _stats_metric)):
            # sns.kdeplot(stats_df_mean[stats_col], ax=ax, shade=True, sharey=True)
            sns.histplot(stats_df_mean[stats_col], ax=ax, kde=True, stat='density', common_norm=False, **plot_kwargs)
            # ax.hist(stats_df_mean[stats_col], alpha=0.5)
            ax.set_title(stats_col)
            ax.vlines(mean_control[stats_col], ymin=0, ymax=ax.get_ylim()[1], color='red', ls='--')
            ax.set_xlabel('Value')
            if len(axis) > 1 and idx == 0:
                ax.set_ylabel('Density')
            else:
                ax.set_ylabel('')
            x_pos = ax.get_xlim()[0] + ((ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.7)
            y_pos = ax.get_ylim()[0] + ((ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.7)
            ax.text(
                x_pos,
                y_pos,
                f'p-value\n{pval[stats_col].values[0]:.3f}',
                ha='center',
                va='center',
                color='red',
            )
        fig.suptitle(f'Bootstrapped metrics: pval_alternative {pval.index[0]}')
        return pval, fig, axis
    return pval


def calculate_arcos_stats(
    df_resampled: pd.DataFrame,
    iterations: list[int],
    position_columns: list,
    frame_column: str,
    obj_id_column: str,
    measurement_column: str,
    smooth_k: int = 3,
    bias_k: int = 51,
    peak_threshold: float = 0.2,
    binarization_threshold: float = 0.1,
    polynomial_degree: int = 1,
    bias_method: str = "runmed",
    eps: float = 2,
    eps_prev: int | None = None,
    min_clustersize: int = 1,
    n_prev: int = 1,
    min_duration: int = 1,
    min_total_size: int = 1,
    stats_metric: list[str] = ['duration', 'total_size'],
    show_progress: bool = True,
    parallel_processing: bool = True,
    clid_column: str = 'clid',
    **kwargs,
):
    """Calculate the bootstrapped statistics.

    Arguments:
        df_resampled (DataFrame): Dataframe with resampled data.
        iterations (list[int]): List of iteration names, or range.
        position_columns (list): List of position columns..
        frame_column (str): Name of the frame column.
        obj_id_column (str): Name of the id column.
        measurement_column (str): Name of the measurement column.
        smooth_k (int, optional): Smoothing kernel size for local detrending. Defaults to 3.
        bias_k (int, optional): Bias kernel size for large scale detrending (used with biasMet='runmed'). Defaults to 51.
        peak_threshold (float, optional): Peak threshold used for rescaling (used with biasMet='runmed'). Defaults to 0.2.
        binarization_threshold (float, optional): Threshold for binarizing measurements after detrending. Defaults to 0.1.
        polynomial_degree (int, optional): Polynomial degree used for detrending (used with biasMet='lm'). Defaults to 1.
        bias_method (str, optional): Bias method, can be 'none', 'runmed', 'lm'. Defaults to "runmed".
        eps (float, optional): Epsilon used for culstering active entities. Defaults to 2.
        eps_prev (int, optional): Epsilon used for linking together culsters across time. Defaults to None.
        min_clustersize (int, optional): Minimum cluster size. Defaults to 1.
        n_prev (int, optional): Number of previous frames to consider when tracking clusters. Defaults to 1.
        min_duration (int, optional): Minimum duration of detected event. Defaults to 1.
        min_total_size (int, optional): Minimum size, minimum size of detected event. Defaults to 1.
        stats_metric (list[str], optional): List of metrics to calculate. Defaults to ['duration', 'total_size'].
        show_progress (bool, optional): Show progress bar. Defaults to True.
        parallel_processing (bool, optional): Use paralell processing, uses the joblib package. Defaults to True.
        clid_column (str, optional): Name of the cluster id column. Defaults to 'clid'.
        **kwargs (Any): Additional keyword arguments. Includes deprecated parameters.
            - posCols: Deprecated. Use position_columns instead.
            - id_column: Deprecated. Use obj_id_column instead.
            - meas_column: Deprecated. Use measurement_column instead.
            - smoothK: Deprecated. Use smooth_k instead.
            - biasK: Deprecated. Use bias_k instead.
            - peakThr: Deprecated. Use peak_threshold instead.
            - binThr: Deprecated. Use binarization_threshold instead.
            - polyDeg: Deprecated. Use polynomial_degree instead.
            - biasMet: Deprecated. Use bias_method instead.
            - min_size: Deprecated. Use min_total_size instead.
            - paralell_processing: Deprecated. Use parallel_processing instead.

    Returns:
        DataFrame (pd.DataFrame): Dataframe with the bootstrapped statistics.
        DataFrame (pd.DataFrame): Dataframe with mean statistics.
    """
    map_deprecated_params = {
        "posCols": "position_columns",
        "id_column": "obj_id_column",
        "meas_column": "measurement_column",
        "smoothK": "smooth_k",
        "biasK": "bias_k",
        "peakThr": "peak_threshold",
        "binThr": "binarization_threshold",
        "polyDeg": "polynomial_degree",
        "biasMet": "bias_method",
        "min_size": "min_total_size",
        "paralell_processing": "parallel_processing",
        "clid_name": "clid_column",
    }

    # check allowed kwargs
    allowed_kwargs = map_deprecated_params.keys()
    for key in kwargs:
        if key not in allowed_kwargs:
            raise ValueError(f"Got an unexpected keyword argument '{key}'")

    updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

    # Assigning the parameters
    position_columns = updated_kwargs.get("position_columns", position_columns)
    obj_id_column = updated_kwargs.get("obj_id_column", obj_id_column)
    measurement_column = updated_kwargs.get("measurement_column", measurement_column)
    smooth_k = updated_kwargs.get("smooth_k", smooth_k)
    bias_k = updated_kwargs.get("bias_k", bias_k)
    peak_threshold = updated_kwargs.get("peak_threshold", peak_threshold)
    binarization_threshold = updated_kwargs.get("binarize_threshold", binarization_threshold)
    polynomial_degree = updated_kwargs.get("polynomial_degree", polynomial_degree)
    bias_method = updated_kwargs.get("bias_method", bias_method)
    min_total_size = updated_kwargs.get("min_total_size", min_total_size)
    parallel_processing = updated_kwargs.get("parallel_processing", parallel_processing)
    clid_column = updated_kwargs.get("clid_column", clid_column)

    if parallel_processing:
        from joblib import Parallel, delayed

        stats_df_list = Parallel(n_jobs=-1)(
            delayed(_apply_arcos)(
                i_iter=i_iter,
                df_resampled=df_resampled,
                position_columns=position_columns,
                frame_column=frame_column,
                obj_id_column=obj_id_column,
                measurement_column=measurement_column,
                smooth_k=smooth_k,
                bias_k=bias_k,
                peak_threshold=peak_threshold,
                binarization_threshold=binarization_threshold,
                polynomial_degree=polynomial_degree,
                bias_method=bias_method,
                eps=eps,
                eps_prev=eps_prev,
                min_clustersize=min_clustersize,
                n_prev=n_prev,
                min_duration=min_duration,
                min_total_size=min_total_size,
                clid_column=clid_column,
            )
            for i_iter in tqdm(iterations, disable=not show_progress)
        )
    else:
        stats_df_list = []
        for i_iter in tqdm(iterations, disable=not show_progress):
            stats_df = _apply_arcos(
                i_iter=i_iter,
                df_resampled=df_resampled,
                position_columns=position_columns,
                frame_column=frame_column,
                obj_id_column=obj_id_column,
                measurement_column=measurement_column,
                smooth_k=smooth_k,
                bias_k=bias_k,
                peak_threshold=peak_threshold,
                binarization_threshold=binarization_threshold,
                polynomial_degree=polynomial_degree,
                bias_method=bias_method,
                eps=eps,
                eps_prev=eps_prev,
                min_clustersize=min_clustersize,
                n_prev=n_prev,
                min_duration=min_duration,
                min_total_size=min_total_size,
                clid_column=clid_column,
            )
            stats_df_list.append(stats_df)

    stats_df = pd.concat(stats_df_list, ignore_index=True)

    stats_df_indexer = ['bootstrap_iteration'] + stats_metric
    stats_df_mean: pd.DataFrame = (
        stats_df[stats_df_indexer].groupby(['bootstrap_iteration']).agg(['mean']).reset_index()
    )
    stats_df_mean = stats_df_mean.droplevel(level=1, axis=1)
    # for bootstrap iteratoins that did not detect any events, set the metric to 0
    stats_df_mean[stats_metric] = stats_df_mean[stats_metric].fillna(0)
    return stats_df, stats_df_mean


def _apply_arcos(
    i_iter: int,
    df_resampled: pd.DataFrame,
    position_columns: list[str],
    frame_column: str,
    obj_id_column: str,
    measurement_column: str,
    smooth_k: int,
    bias_k: int,
    peak_threshold: float,
    binarization_threshold: float,
    polynomial_degree: int,
    bias_method: str,
    eps: float,
    eps_prev: int | None,
    min_clustersize: int,
    n_prev: int,
    min_duration: int,
    min_total_size: int,
    clid_column: str,
) -> pd.DataFrame:
    df_i_iter = df_resampled.loc[df_resampled['iteration'] == i_iter]
    arcos_instance = ARCOS(
        data=df_i_iter,
        posCols=position_columns,
        frame_column=frame_column,
        obj_id_column=obj_id_column,
        measurement_column=measurement_column,
        clid_column=clid_column,
    )
    arcos_instance.interpolate_measurements()
    arcos_instance.bin_measurements(
        smooth_k=smooth_k,
        bias_k=bias_k,
        peak_threshold=peak_threshold,
        binarization_threshold=binarization_threshold,
        polynomial_degree=polynomial_degree,
        bias_method=bias_method,
    )
    df_arcos = arcos_instance.trackCollev(eps=eps, eps_prev=eps_prev, min_clustersize=min_clustersize, n_prev=n_prev)

    df_arcos_filtered = filterCollev(
        data=df_arcos, frame_column=frame_column, clid_column=clid_column, obj_id_column=obj_id_column
    ).filter(min_duration, min_total_size)

    if i_iter == 0 and df_arcos.empty:
        raise ValueError('No events detected in control, consider changing parameters')

    stats_df = calcCollevStats().calculate(
        df_arcos_filtered, frame_column, clid_column, obj_id_column, position_columns
    )
    stats_df['bootstrap_iteration'] = i_iter
    return stats_df


def _p_val_finite_sampling(x: pd.DataFrame, alternative: str = 'greater') -> pd.Series:
    orig = x[0]
    df_test = x[1:]
    if alternative == 'greater':
        return pd.Series({'greater': (1 + sum(df_test >= orig)) / (len(df_test) + 1)})
    elif alternative == 'less':
        return pd.Series({'less': (1 + sum(df_test <= orig)) / (len(df_test) + 1)})
    elif alternative == 'both':
        warn(
            'Combined p-values will not add up to 1 due to the fact that greater\
                and equal and less and equal are not mutually exclusive.'
        )
        return pd.Series(
            {
                'greater': (1 + sum(df_test >= orig)) / (len(df_test) + 1),
                'less': (1 + sum(df_test <= orig)) / (len(df_test) + 1),
            }
        )
    else:
        raise ValueError(f'alternative must be one of "greater", "less" or "both". Got {alternative}')


def _p_val_infinite_sampling(x: pd.DataFrame, alternative: str = 'greater') -> pd.Series:
    orig = x[0]
    df_test = x[1:]
    if alternative == 'greater':
        return pd.Series({'greater': sum(df_test >= orig) / len(df_test)})
    elif alternative == 'less':
        return pd.Series({'less': sum(df_test <= orig) / len(df_test)})
    elif alternative == 'both':
        warn(
            'Combined p-values will not add up to 1 due to the fact that greater and equal and less and equal are not mutually exclusive.'  # noqa: E501
        )
        return pd.Series({'greater': sum(df_test >= orig) / len(df_test), 'less': sum(df_test <= orig) / len(df_test)})
    else:
        raise ValueError(f'alternative must be one of "greater", "less", or "both". Got {alternative}')
