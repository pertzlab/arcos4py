"""Module to plot different metrics generated by arcos4py functions.

Examples:
    >>> # Data Plots
    >>> from arcos4py.plotting import dataPlots
    >>> data_plots = dataPlots(df,'time', 'meas', 'track_id')
    >>> hist = data_plots.histogram()
    >>> dens = data_plots.density_plot()
    >>> xt_plot = data_plots.position_t_plot({'x'}, n=20)

    >>> # Detrended vs original plot
    >>> from arcos4py.plotting import plotOriginalDetrended
    >>> arcosPlots = plotOriginalDetrended(data, 'time', 'meas', 'detrended', 'id')
    >>> plot = arcosPlots(data, 'time', 'meas', 'detrended', 'id')
    >>> plot.plot_detrended()

    >>> # Stats Plot
    >>> from arcos4py.plotting import statsPlots
    >>> coll_dur_size_scatter = statsPlots(stats).plot_events_duration('total_size','duration')

    >>> # Noodle Plot
    >>> from arcos4py.plotting import NoodlePlot
    >>> ndl = NoodlePlot(df,"collid", 'track_id', 'time', 'x', 'y')
    >>> ndl_plot = ndl.plot('x')
"""

from __future__ import annotations

from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..tools._arcos4py_deprecation import handle_deprecated_params

TAB20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]


class dataPlots:
    """Plot different metrics of input data.

    Attributes:
        data (Dataframe): containing ARCOS data.
        frame_column (str): name of frame column in data.
        measurement_column (str): name of measurement column in data.
        obj_id_column (str): name of track id column.
    """

    def __init__(self, data: pd.DataFrame, frame_column: str, measurement_column: str, obj_id_column: str, **kwargs):
        """Plot different metrics such as histogram, position-t and density.

        Arguments:
            data (Dataframe): containing ARCOS data.
            frame_column (str): name of frame column in data.
            measurement_column (str): name of measurement column in data.
            obj_id_column (str): name of track id column.
            **kwargs (Any): Additional keyword arguments. Includes deprecated parameters.
                - id (str): Deprecated. Use obj_id_column instead.
                - frame (str): Deprecated. Use frame_column instead.
                - measurement (str): Deprecated. Use measurement_column instead.
        """
        map_deprecated_params = {
            "id": "obj_id_column",
            "frame": "frame_column",
            "measurement": "measurement_column",
        }

        # check allowed kwargs
        allowed_kwargs = map_deprecated_params.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Got an unexpected keyword argument '{key}'")

        updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

        # Assigning the parameters
        obj_id_column = updated_kwargs.get("obj_id_column", obj_id_column)
        frame_column = updated_kwargs.get("frame_column", frame_column)
        measurement_column = updated_kwargs.get("measurement_column", measurement_column)

        self.data = data
        self.obj_id = obj_id_column
        self.frame_column = frame_column
        self.measurement_column = measurement_column

    def position_t_plot(self, position_columns: set[str] = {'x'}, n: int = 20, **kwargs) -> Union[plt.Figure, Any]:
        """Plots X and Y over T to visualize tracklength.

        Arguments:
            position_columns (set): containing names of position columns in data.
            n (int): number of samples to plot.
            **kwargs (Any): Additional keyword arguments. Includes deprecated parameters.
                - posCol (set): Deprecated. Use position_columns instead.

        Returns:
            fig (matplotlib.figure.Figure): Matplotlib figure object of density plot.
            axes (matplotlib.axes.Axes): Matplotlib axes of density plot.
        """
        map_deprecated_params = {
            "posCol": "position_columns",
        }

        # check allowed kwargs
        allowed_kwargs = map_deprecated_params.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Got an unexpected keyword argument '{key}'")

        updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

        # Assigning the parameters
        position_columns = updated_kwargs.get("position_columns", position_columns)

        sample = pd.Series(self.data[self.obj_id].unique()).sample(n)
        pd_from_r_df = self.data.loc[self.data[self.obj_id].isin(sample)]
        fig, axes = plt.subplots(1, len(position_columns), figsize=(6, 3))
        for _, df in pd_from_r_df.groupby(self.obj_id):
            for index, value in enumerate(position_columns):
                if len(position_columns) > 1:
                    df.plot(x=self.frame_column, y=value, ax=axes[index], legend=None)
                else:
                    df.plot(x=self.frame_column, y=value, ax=axes, legend=None)
        if len(position_columns) > 1:
            for index, value in enumerate(position_columns):
                axes[index].set_title(value)
        else:
            axes.set_title(value)
        return fig, axes

    def density_plot(self, *args, **kwargs):
        """Density plot of measurement.

        Uses Seaborn distplot to plot measurement density.

        Arguments:
            *args (Any): arguments passed on to seaborn histplot function.
            **kwargs (Any): keyword arguments passed on to seaborn histplot function.

        Returns:
            FacetGrid (seaborn.FacetGrid): Seaborn FacetGrid of density density plot.
        """
        plot = sns.displot(
            self.data[self.measurement_column],
            kind="kde",
            palette="pastel",
            label=self.measurement_column,
            *args,
            **kwargs,
        )
        # Plot formatting
        plt.legend(prop={'size': 10})
        plt.title('Density Plot of Measurement')
        plt.xlabel('Measurement')
        plt.ylabel('Density')
        return plot

    def histogram(self, bins: str = 'auto', *args, **kwargs) -> plt.Axes:
        """Histogram of tracklenght.

        Uses seaborn histplot function to plot tracklenght histogram.

        Arguments:
            bins (str): number or width of bins in histogram
            *args (Any): arguments passed on to seaborn histplot function.
            **kwargs (Any): keyword arguments passed on to seaborn histplot function.

        Returns:
            AxesSubplot: Matplotlib AxesSubplot of histogram.
        """
        # Draw histogram
        track_length = self.data.groupby(self.obj_id).size()
        axes = sns.histplot(track_length, label="Track Length", bins=bins, *args, **kwargs)
        # Plot formatting
        plt.title('Track length Histogram')
        axes.set_xlabel('Track Length')
        axes.set_ylabel('Count')
        return axes


class plotOriginalDetrended:
    """Plot original and detrended data.

    Attributes:
        data (DataFrame): containing ARCOS data.
        frame_column (str): name of frame column in data.
        measurement_column (str): name of measurement column in data.
        detrended_column (str): name of detrended column in data.
        obj_id_column (str): name of track id column.
        seed (int): seed for random number generator.

    Methods:
        plot_detrended: plot detrended data.
        plot_original: plot original data.
        plot_original_and_detrended: plot original and detrended data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        frame_column: str,
        measurement_column: str,
        detrended_column: str,
        obj_id_column: str,
        seed: int = 42,
        **kwargs,
    ):
        """Constructs class with given parameters."""
        map_deprecated_params = {
            "id": "obj_id_column",
            "frame": "frame_column",
            "detrended": "detrended_column",
            "measurement": "measurement_column",
        }

        # check allowed kwargs
        allowed_kwargs = map_deprecated_params.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Got an unexpected keyword argument '{key}'")

        updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

        # Assigning the parameters
        obj_id_column = updated_kwargs.get("obj_id_column", obj_id_column)
        frame_column = updated_kwargs.get("frame_column", frame_column)
        measurement_column = updated_kwargs.get("measurement_column", measurement_column)

        self.data = data
        self.frame_column = frame_column
        self.measurement_column = measurement_column
        self.detrended_column = detrended_column
        self.obj_id_column = obj_id_column
        self.seed = seed

    def _prepare_data(self, n_samples: int):
        rng_gen = np.random.default_rng(seed=self.seed)
        vals = rng_gen.choice(self.data[self.obj_id_column].unique(), n_samples, replace=False)  # noqa: F841
        filtered_data = self.data.query(f"{self.obj_id_column} in @vals")
        return filtered_data.groupby(self.obj_id_column)

    def _plot_data(self, grouped, ncols, nrows, plotsize, plot_columns, labels, add_binary_segments=False):
        fig, axes2d = plt.subplots(nrows=nrows, ncols=ncols, figsize=plotsize, sharey=True)
        max_val = 0
        for (name, group), ax in zip(grouped, axes2d.flatten()):
            for column, label in zip(plot_columns, labels):
                ax.plot(group[self.frame_column], group[column], label=label)
                max_val = group[column].max() if group[column].max() > max_val else max_val
            ax.set_title(f"Track {name}")

        if add_binary_segments:
            for (name, group), ax in zip(grouped, axes2d.flatten()):
                self._add_binary_segments(group, ax, max_val)

        fig.supxlabel('Time Point')
        fig.supylabel('Measurement')
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, loc='upper right')

        return fig, axes2d

    def _add_binary_segments(self, group, ax, max_val):
        x_val = group[group[f"{self.measurement_column}.bin"] != 0][self.frame_column]
        y_val = np.repeat(max_val, x_val.size)
        indices = np.where(np.diff(x_val) != 1)[0] + 1
        x_split = np.split(x_val, indices)
        y_split = np.split(y_val, indices)
        for idx, (x_val, y_val) in enumerate(zip(x_split, y_split)):
            if idx == 0:
                ax.plot(x_val, y_val, color="red", lw=2, label="bin")
            else:
                ax.plot(x_val, y_val, color="red", lw=2)

    def plot_detrended(
        self,
        n_samples: int = 25,
        subplots: tuple = (5, 5),
        plotsize: tuple = (20, 10),
        add_binary_segments: bool = False,
    ) -> tuple[plt.Figure, Any]:
        """Plots detrended data.

        Arguments:
            n_samples (int): number of samples to plot.
            subplots (tuple): number of subplots in x and y direction.
            plotsize (tuple): size of the plot.
            add_binary_segments (bool): if True, binary segments are added to the plot.

        Returns:
            fig (matplotlib.figure.Figure): Matplotlib figure object of plot.
            axes (matplotlib.axes.Axes): Matplotlib axes of plot.
        """
        grouped = self._prepare_data(n_samples)
        return self._plot_data(
            grouped, subplots[0], subplots[1], plotsize, [self.detrended_column], ["detrended"], add_binary_segments
        )

    def plot_original(
        self,
        n_samples: int = 25,
        subplots: tuple = (5, 5),
        plotsize: tuple = (20, 10),
        add_binary_segments: bool = False,
    ) -> tuple[plt.Figure, Any]:
        """Plots original data.

        Arguments:
            n_samples (int): number of samples to plot.
            subplots (tuple): number of subplots in x and y direction.
            plotsize (tuple): size of the plot.
            add_binary_segments (bool): if True, binary segments are added to the plot.

        Returns:
            fig (matplotlib.figure.Figure): Matplotlib figure object of plot.
            axes (matplotlib.axes.Axes): Matplotlib axes of plot.
        """
        grouped = self._prepare_data(n_samples)
        return self._plot_data(
            grouped,
            subplots[0],
            subplots[1],
            plotsize,
            [self.measurement_column],
            ["original"],
            add_binary_segments,
        )

    def plot_original_and_detrended(
        self,
        n_samples: int = 25,
        subplots: tuple = (5, 5),
        plotsize: tuple = (20, 10),
        add_binary_segments: bool = False,
    ) -> tuple[plt.Figure, Any]:
        """Plots original and detrended data.

        Arguments:
            n_samples (int): number of samples to plot.
            subplots (tuple): number of subplots in x and y direction.
            plotsize (tuple): size of the plot.
            add_binary_segments (bool): if True, binary segments are added to the plot.

        Returns:
            fig (matplotlib.figure.Figure): Matplotlib figure object of plot.
            axes (matplotlib.axes.Axes): Matplotlib axes of plot.
        """
        grouped = self._prepare_data(n_samples)
        return self._plot_data(
            grouped,
            subplots[0],
            subplots[1],
            plotsize,
            [self.measurement_column, self.detrended_column],
            ["original", "detrended"],
            add_binary_segments,
        )


class statsPlots:
    """Plot data generated by the stats module.

    Attributes:
        data (DataFrame): containing ARCOS stats data.
    """

    def __init__(self, data: pd.DataFrame):
        """Plot detrended vs original data.

        Arguments:
            data (DataFrame): containing ARCOS stats data.
        """
        self.data = data

    def plot_events_duration(self, total_size: str, duration: str, point_size: int = 40, *args, **kwargs) -> plt.Axes:
        """Scatterplot of collective event duration.

        Arguments:
            total_size (str): name of total size column.
            duration (str):, name of column with collective event duration.
            point_size (int): scatterplot point size.
            *args (Any): Arguments passed on to seaborn scatterplot function.
            **kwargs (Any): Keyword arguments passed on to seaborn scatterplot function.

        Returns:
            Axes (matplotlib.axes.Axes): Matplotlib Axes object of scatterplot
        """
        if self.data.empty:
            raise ValueError("Dataframe is empty")
        plot = sns.scatterplot(x=self.data[total_size], y=self.data[duration], s=point_size, *args, **kwargs)
        return plot


class NoodlePlot:
    """Create Noodle Plot of cell tracks, colored by collective event id.

    Attributes:
        df (pd.DataFrame): DataFrame containing collective events from arcos.
        colev (str): Name of the collective event column in df.
        trackid (str): Name of the track column in df.
        frame (str): Name of the frame column in df.
        posx (str): Name of the X coordinate column in df.
        posy (str): Name of the Y coordinate column in df.
        posz (str): Name of the Z coordinate column in df,
            or None if no z column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        clid_column: str,
        obj_id_column: str,
        frame_column: str,
        posx: str,
        posy: str,
        posz: Union[str, None] = None,
        **kwargs,
    ):
        """Constructs class with given parameters.

        Arguments:
            df (pd.DataFrame): DataFrame containing collective events from arcos.
            clid_column (str): Name of the collective event column in df.
            obj_id_column (str): Name of the track column in df.
            frame_column (str): Name of the frame column in df.
            posx (str): Name of the X coordinate column in df.
            posy (str): Name of the Y coordinate column in df.
            posz (str | None): Name of the Z coordinate column in df,
                or None if no z column.
            **kwargs (Any): Additional keyword arguments for plot. Includes deprecated parameters.
                - colev (str): Deprecated. Use clid_column instead.
                - trackid (str): Deprecated. Use obj_id_column instead.
                - frame (str): Deprecated. Use frame_column instead.
        """
        map_deprecated_params = {
            "colev": "clid_column",
            "trackid": "obj_id_column",
            "frame": "frame_column",
        }

        # allowed matplotlib kwargs
        allowed_kwargs = [
            "alpha",
            "animated",
            "c",
            "label",
            "linewidth",
            "linestyle",
            "marker",
            "markersize",
            "markeredgecolor",
            "markerfacecolor",
            "markerfacecoloralt",
            "markeredgewidth",
            "path_effects",
            "picker",
            "pickradius",
            "solid_capstyle",
            "solid_joinstyle",
            "transform",
            "visible",
            "zorder",
        ]

        # check allowed kwargs
        allowed_kwargs_2 = map_deprecated_params.keys()
        for key in kwargs:
            if key not in allowed_kwargs and key not in allowed_kwargs_2:
                raise ValueError(f"Got an unexpected keyword argument '{key}'")

        updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

        # Assigning the parameters
        clid_column = updated_kwargs.pop("clid_column", clid_column)
        obj_id_column = updated_kwargs.pop("obj_id_column", obj_id_column)
        frame_column = updated_kwargs.pop("frame_column", frame_column)

        self.df = df
        self.clid_column = clid_column
        self.obj_id_column = obj_id_column
        self.frame_column = frame_column
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.plot_kwargs = updated_kwargs

    def _prepare_data_noodleplot(
        self,
        df: pd.DataFrame,
        color_cylce: list[str],
        clid_column: str,
        obj_id_column: str,
        frame_column: str,
        posx: str,
        posy: str,
        posz: Union[str, None] = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """From arcos collective event data,\
        generates a list of numpy arrays, one for each event.

        Arguments:
            df (pd.DataFrame): DataFrame containing collective events from arcos.
            color_cylce (list[str]): list of colors used to color trackid's
                for individual collective events.
            colev (str): Name of the collective event column in df.
            trackid (str): Name of the track column in df.
            frame: (str): Name of the frame column in df.
            posx (str): Name of the X coordinate column in df.
            posy (str): Name of the Y coordinate column in df.
            posz (str): Name of the Z coordinate column in df,
                or None if no z column.

        Returns:
            grouped_array (list[np.ndarray]): List of collective events data
            colors (np.ndarray): colors for each collective event.
        """
        df = df.copy()
        # factorize trackid to get unique values and make sure they are nummeric
        df[obj_id_column] = df[obj_id_column].factorize()[0]
        # sort by collective event and trackid
        df = df.sort_values([clid_column, obj_id_column])
        if posz:
            array = df[[clid_column, obj_id_column, frame_column, posx, posy, posz]].to_numpy()
        else:
            array = df[[clid_column, obj_id_column, frame_column, posx, posy]].to_numpy()
        # generate goroups for each unique value
        grouped_array = np.split(array, np.unique(array[:, 0], axis=0, return_index=True)[1][1:])
        # make collids sequential
        seq_colids = np.concatenate(
            [np.repeat(i, value.shape[0]) for i, value in enumerate(grouped_array)],
            axis=0,
        )
        array_seq_colids = np.column_stack((array, seq_colids))
        # split sequential collids array by trackid and collid
        grouped_array = np.split(
            array_seq_colids,
            np.unique(array_seq_colids[:, :2], axis=0, return_index=True)[1][1:],
        )
        # generate colors for each collective event, wrap arround the color cycle
        colors = np.take(np.array(color_cylce), [i + 1 for i in np.unique(seq_colids)], mode="wrap")
        return grouped_array, colors

    def _create_noodle_plot(self, grouped_data: list[np.ndarray], colors: np.ndarray):
        """Plots the noodle plot."""
        fig, ax = plt.subplots()
        ax.set_xlabel("Time Point")
        ax.set_ylabel("Position")
        for dat in grouped_data:
            if dat.size == 0:
                continue
            ax.plot(
                dat[:, 2],
                dat[:, self.projection_index],
                c=colors[int(dat[0, -1])],
                **self.plot_kwargs,
            )
        return fig, ax

    def plot(self, projection_axis: str, color_cylce: list[str] = TAB20):
        """Create Noodle Plot of cell tracks, colored by collective event id.

        Arguments:
            projection_axis (str): Specify with witch coordinate the noodle
                plot should be drawn. Has to be one of the posx, posy or posz arguments
                passed in during the class instantiation process.
            color_cylce (list[str]): List of hex color values or string names
                (i.e. ['red', 'yellow']) used to color collecitve events. Cycles through list.

        Returns:
            fig (matplotlib.figure.Figure): Matplotlib figure object for the noodle plot.
            axes (matplotlib.axes.Axes): Matplotlib axes for the nooble plot.
        """
        if self.df.empty:
            raise ValueError("Dataframe is empty")
        if projection_axis not in [self.posx, self.posy, self.posz]:
            raise ValueError(f"projection_axis has to be one of {[self.posx, self.posy, self.posz]}")
        if projection_axis == self.posx:
            self.projection_index = 3
        elif projection_axis == self.posy:
            self.projection_index = 4
        elif projection_axis == self.posz:
            self.projection_index = 5
        grpd_data, colors = self._prepare_data_noodleplot(
            self.df,
            color_cylce,
            self.clid_column,
            self.obj_id_column,
            self.frame_column,
            self.posx,
            self.posy,
            self.posz,
        )
        fig, axes = self._create_noodle_plot(grpd_data, colors)
        return fig, axes
