"""Tools for resampling data and bootstrapping."""

from __future__ import annotations

from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from arcos4py import ARCOS
from arcos4py.tools import calcCollevStats, filterCollev

from ._resampling import resample_data


def bootstrap_arcos(
    df: pd.DataFrame,
    posCols: list,
    frame_column: str,
    id_column: str,
    meas_column: str,
    method: str | list[str] = 'shuffle_tracks',
    smoothK: int = 3,
    biasK: int = 51,
    peakThr: float = 0.2,
    binThr: float = 0.1,
    polyDeg: int = 1,
    biasMet: str = "runmed",
    eps: float = 2,
    epsPrev: int | None = None,
    minClsz: int = 1,
    nPrev: int = 1,
    min_duration: int = 1,
    min_size: int = 1,
    stats_metric: str | list[str] = ["total_size", "duration"],
    pval_alternative: str = "greater",
    finite_correction: bool = True,
    n: int = 100,
    seed: int = 42,
    allow_duplicates: bool = False,
    max_tries: int = 100,
    show_progress: bool = True,
    verbose: bool = False,
    paralell_processing: bool = True,
    plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap data using the ARCOS algorithm.

    Arguments:
        df: DataFrame containing the data to be bootstrapped.
        posCols: List of column names containing the x and y coordinates.
        frame_column: Name of the column containing the frame number.
        id_column: Name of the column containing the track id.
        meas_column: Name of the column containing the measurement.
        method: Method used for bootstrapping. Can be "shuffle_tracks", 'shuffle_timepoints', 'shift_timepoints',
            'shuffle_binary_blocks', 'shuffle_coordinates_timepoint or a list of methods,
            which will be applied in order of index.
        smoothK: Smoothing kernel size.
        biasK: Bias kernel size.
        peakThr: Threshold for peak detection.
        binThr: Threshold for binarization.
        polyDeg: Degree of the polynomial used for bias correction.
        biasMet: Bias correction method. Can be 'none', 'runmed', 'lm'
        eps: Epsilon parameter for DBSCAN.
        epsPrev: Parameter for linking tracks. If None, eps is used.
        minClsz: Minimum cluster size.
        nPrev: Number of previous frames to consider for linking.
        min_duration: Minimum duration of a track.
        min_size: Minimum size of a track.
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

    Returns:
        DataFrame containing the bootstrapped data.
    """
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
        posCols=posCols,
        frame_column=frame_column,
        id_column=id_column,
        meas_column=meas_column,
        method=method,
        n=n,
        seed=seed,
        allow_duplicates=allow_duplicates,
        max_tries=max_tries,
        show_progress=show_progress,
        verbose=verbose,
        paralell_processing=paralell_processing,
    )

    iterations = df_resampled['iteration'].unique()

    print(f'Running ARCOS and calculating "{stats_metric}"...')

    stats_df, stats_df_mean = calculate_arcos_stats(
        df_resampled=df_resampled,
        posCols=posCols,
        frame_column=frame_column,
        id_column=id_column,
        meas_column=meas_column,
        smoothK=smoothK,
        biasK=biasK,
        peakThr=peakThr,
        binThr=binThr,
        polyDeg=polyDeg,
        biasMet=biasMet,
        eps=eps,
        epsPrev=epsPrev,
        minClsz=minClsz,
        nPrev=nPrev,
        min_duration=min_duration,
        min_size=min_size,
        stats_metric=stats_metric,
        show_progress=show_progress,
        paralell_processing=paralell_processing,
        clid_name=clid_name,
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
        pval = stats_df_mean[['total_size', 'duration']].agg(lambda x: _p_val_infinite_sampling(x, pval_alternative))
    pval.name = 'p_value'

    if isinstance(stats_metric, list):
        _stats_metric = stats_metric
    else:
        _stats_metric = [stats_metric]

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
            ax.vlines(stats_df_mean[stats_col].iloc[0], ymin=0, ymax=ax.get_ylim()[1], color='red', ls='--')
            ax.set_xlabel('Value')
            if len(axis) > 1 and idx == 0:
                ax.set_ylabel('Density')
            else:
                ax.set_ylabel('')
            x_pos = ax.get_xlim()[0] + ((ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.8)
            y_pos = ax.get_ylim()[0] + ((ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.8)
            ax.text(
                x_pos,
                y_pos,
                f'p-value: \n {pval.loc[:,stats_col].round(3).to_string()}',
                ha='center',
                va='center',
                color='red',
            )
        fig.suptitle('Bootstrapped metrics')
        plt.show()
    return pval


def calculate_arcos_stats(
    df_resampled: pd.DataFrame,
    iterations: list[int],
    posCols: list,
    frame_column: str,
    id_column: str,
    meas_column: str,
    smoothK: int = 3,
    biasK: int = 51,
    peakThr: float = 0.2,
    binThr: float = 0.1,
    polyDeg: int = 1,
    biasMet: str = "runmed",
    eps: float = 2,
    epsPrev: int | None = None,
    minClsz: int = 1,
    nPrev: int = 1,
    min_duration: int = 1,
    min_size: int = 1,
    stats_metric: list[str] = ['duration', 'total_size'],
    show_progress: bool = True,
    paralell_processing: bool = True,
    clid_name: str = 'clid',
):
    """Calculate the bootstrapped statistics.

    Arguments:
        df_resampled (DataFrame): Dataframe with resampled data.
        iterations (list[int]): List of iteration names, or range.
        posCols (list): List of position columns..
        frame_column (str): Name of the frame column.
        id_column (str): Name of the id column.
        meas_column (str): Name of the measurement column.
        smoothK (int, optional): Smoothing kernel size for local detrending. Defaults to 3.
        biasK (int, optional): Bias kernel size for large scale detrending (used with biasMet='runmed'). Defaults to 51.
        peakThr (float, optional): Peak threshold used for rescaling (used with biasMet='runmed'). Defaults to 0.2.
        binThr (float, optional): Threshold for binarizing measurements after detrending. Defaults to 0.1.
        polyDeg (int, optional): Polynomial degree used for detrending (used with biasMet='lm'). Defaults to 1.
        biasMet (str, optional): Bias method, can be 'none', 'runmed', 'lm'. Defaults to "runmed".
        eps (float, optional): Epsilon used for culstering active entities. Defaults to 2.
        epsPrev (int, optional): Epsilon used for linking together culsters across time. Defaults to None.
        minClsz (int, optional): Minimum cluster size. Defaults to 1.
        nPrev (int, optional): Number of previous frames to consider when tracking clusters. Defaults to 1.
        min_duration (int, optional): Minimum duration of detected event. Defaults to 1.
        min_size (int, optional): Minimum size, minimum size of detected event. Defaults to 1.
        stats_metric (list[str], optional): List of metrics to calculate. Defaults to ['duration', 'total_size'].
        show_progress (bool, optional): Show progress bar. Defaults to True.
        paralell_processing (bool, optional): Use paralell processing, uses the joblib package. Defaults to True.
        clid_name (str, optional): Name of the cluster id column. Defaults to 'clid'.

    Returns:
        DataFrame (pd.DataFrame): Dataframe with the bootstrapped statistics.
        DataFrame (pd.DataFrame): Dataframe with mean statistics.
    """
    if paralell_processing:
        from joblib import Parallel, delayed

        stats_df_list = Parallel(n_jobs=-1)(
            delayed(_apply_arcos)(
                i_iter=i_iter,
                df_resampled=df_resampled,
                posCols=posCols,
                frame_column=frame_column,
                id_column=id_column,
                meas_column=meas_column,
                smoothK=smoothK,
                biasK=biasK,
                peakThr=peakThr,
                binThr=binThr,
                polyDeg=polyDeg,
                biasMet=biasMet,
                eps=eps,
                epsPrev=epsPrev,
                minClsz=minClsz,
                nPrev=nPrev,
                min_dur=min_duration,
                min_size=min_size,
                clid_name=clid_name,
            )
            for i_iter in tqdm(iterations, disable=not show_progress)
        )
    else:
        stats_df_list = []
        for i_iter in tqdm(iterations, disable=not show_progress):
            stats_df = _apply_arcos(
                i_iter=i_iter,
                df_resampled=df_resampled,
                posCols=posCols,
                frame_column=frame_column,
                id_column=id_column,
                meas_column=meas_column,
                smoothK=smoothK,
                biasK=biasK,
                peakThr=peakThr,
                binThr=binThr,
                polyDeg=polyDeg,
                biasMet=biasMet,
                eps=eps,
                epsPrev=epsPrev,
                minClsz=minClsz,
                nPrev=nPrev,
                min_dur=min_duration,
                min_size=min_size,
                clid_name=clid_name,
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
    posCols: list[str],
    frame_column: str,
    id_column: str,
    meas_column: str,
    smoothK: int,
    biasK: int,
    peakThr: float,
    binThr: float,
    polyDeg: int,
    biasMet: str,
    eps: float,
    epsPrev: int | None,
    minClsz: int,
    nPrev: int,
    min_dur: int,
    min_size: int,
    clid_name: str,
) -> pd.DataFrame:
    df_i_iter = df_resampled.loc[df_resampled['iteration'] == i_iter]
    arcos_instance = ARCOS(
        data=df_i_iter,
        posCols=posCols,
        frame_column=frame_column,
        id_column=id_column,
        measurement_column=meas_column,
        clid_column=clid_name,
    )
    arcos_instance.interpolate_measurements()
    arcos_instance.bin_measurements(
        smoothK=smoothK, biasK=biasK, peakThr=peakThr, binThr=binThr, polyDeg=polyDeg, biasMet=biasMet
    )
    df_arcos = arcos_instance.trackCollev(eps=eps, epsPrev=epsPrev, minClsz=minClsz, nPrev=nPrev)

    df_arcos_filtered = filterCollev(
        data=df_arcos, frame_column=frame_column, collid_column=clid_name, obj_id_column=id_column
    ).filter(min_dur, min_size)

    if i_iter == 0 and df_arcos.empty:
        raise ValueError('No events detected in control, consider changing parameters')

    stats_df = calcCollevStats().calculate(df_arcos_filtered, frame_column, clid_name, id_column, posCols)
    stats_df['bootstrap_iteration'] = i_iter
    return stats_df


def _p_val_finite_sampling(x: pd.DataFrame, alternative: str = 'greater') -> pd.Series:
    orig = x[0]
    df_test = x[1:]
    if alternative == 'greater':
        return pd.Series({'>=': (1 + sum(df_test >= orig)) / (len(df_test) + 1)})
    elif alternative == 'less':
        return pd.Series({'<=': (1 + sum(df_test <= orig)) / (len(df_test) + 1)})
    elif alternative == 'both':
        warn(
            'Combined p-values will not add up to 1 due to the fact that greater\
                and equal and less and equal are not mutually exclusive.'
        )
        return pd.Series(
            {
                '>=': (1 + sum(df_test >= orig)) / (len(df_test) + 1),
                '<=': (1 + sum(df_test <= orig)) / (len(df_test) + 1),
            }
        )
    else:
        raise ValueError(f'alternative must be one of "greater", "less" or "both". Got {alternative}')


def _p_val_infinite_sampling(x: pd.DataFrame, alternative: str = 'greater') -> pd.Series:
    orig = x[0]
    df_test = x[1:]
    if alternative == 'greater':
        return pd.Series({'>=': sum(df_test >= orig) / len(df_test)})
    elif alternative == 'less':
        return pd.Series({'<=': sum(df_test <= orig) / len(df_test)})
    elif alternative == 'both':
        warn(
            'Combined p-values will not add up to 1 due to the fact that greater and equal and less and equal are not mutually exclusive.'  # noqa: E501
        )
        return pd.Series({'>=': sum(df_test >= orig) / len(df_test), '<=': sum(df_test <= orig) / len(df_test)})
    else:
        raise ValueError(f'alternative must be one of "greater", "less", or "both". Got {alternative}')
