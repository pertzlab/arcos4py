"""Module to filter collective events.

Example:
    >>> from arcos4py.tools import filterCollev
    >>> f = filterCollev(data, 'time', 'collid')
    >>> df = f.filter(min_duration = 9, min_total_size = 10)
"""

import pandas as pd

from ._arcos4py_deprecation import handle_deprecated_params
from ._stats import calcCollevStats


class filterCollev:
    """Select Collective events that last longer than coll_duration\
    and have a larger total size than coll_total_size.

    Attributes:
        data (Dataframe): With detected collective events.
        frame_column (str): Indicating the frame column in data.
        collid_column (str): Indicating the collective event id column in data.
        obj_id_column (str): Inidicating the object identifier column such as cell track id.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        frame_column: str = "time",
        clid_column: str = "collid",
        obj_id_column: str = "trackID",
        **kwargs,
    ):
        """Constructs filterCollev class with Parameters.

        Arguments:
            data (Dataframe): With detected collective events.
            frame_column (str): Indicating the frame column in data.
            clid_column (str): Indicating the collective event id column in data.
            obj_id_column (str): Inidicating the object identifier column such as cell track id.
            **kwargs (Any): Additional keyword arguments. Includes deprecated parameters.
                - collid_column (str): Deprecated. Use clid_column instead.
        """
        map_deprecated_params = {
            "collid_column": "clid_column",
        }

        # check allowed kwargs
        allowed_kwargs = map_deprecated_params.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Got an unexpected keyword argument '{key}'")

        updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

        # Assigning the parameters
        clid_column = updated_kwargs.get("clid_column", clid_column)

        self.data = data
        self.frame_column = frame_column
        self.clid_column = clid_column
        self.obj_id_column = obj_id_column

    def _filter_collev(
        self,
        data: pd.DataFrame,
        clid_stats: pd.DataFrame,
        clid_column: str,
        min_duration: int,
        min_total_size: int,
    ):
        clid_stats = clid_stats[(clid_stats["duration"] >= min_duration) & (clid_stats["total_size"] >= min_total_size)]
        data = data[data[clid_column].isin(clid_stats[clid_column])]
        return data

    def filter(self, min_duration: int = 9, min_total_size: int = 10, **kwargs) -> pd.DataFrame:
        """Filter collective events.

        Method to filter collective events according to the
        parameters specified in the object instance.

        Arguments:
            min_duration (int): Minimal duration of collective events to be selected.
            min_total_size (int): Minimal total size of collective events to be selected.
            **kwargs (Any): Additional keyword arguments. Includes deprecated parameters.
                - coll_duration (int): Deprecated. Use min_duration instead.
                - coll_total_size (int): Deprecated. Use min_total_size instead.

        Returns:
             Returns pandas dataframe containing filtered collective events
        """
        map_deprecated_params = {
            "coll_duration": "min_duration",
            "coll_total_size": "min_total_size",
        }

        # check allowed kwargs
        allowed_kwargs = map_deprecated_params.keys()
        for key in kwargs:
            if key not in allowed_kwargs:
                raise ValueError(f"Got an unexpected keyword argument '{key}'")

        updated_kwargs = handle_deprecated_params(map_deprecated_params, **kwargs)

        # Assigning the parameters
        min_duration = updated_kwargs.get("min_duration", min_duration)
        min_total_size = updated_kwargs.get("min_total_size", min_total_size)

        if self.data.empty:
            return self.data
        stats = calcCollevStats()
        stats_df = stats.calculate(self.data, self.frame_column, self.clid_column, self.obj_id_column)

        filtered_df = self._filter_collev(
            data=self.data,
            clid_stats=stats_df,
            clid_column=self.clid_column,
            min_duration=min_duration,
            min_total_size=min_total_size,
        )
        return filtered_df
