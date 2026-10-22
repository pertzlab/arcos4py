# Co-activation event detection

`detect_coactivation_events` finds collective events directly in the continuous signal of tracked cells (or
tiles of an image), without binarising each trace first and without distance, size or memory parameters.
The only parameter is `epsilon`, the number of false events you are willing to expect per movie (default 1).

## When to use it

| | `binData` + `track_events_dataframe` | `detect_coactivation_events` |
|---|---|---|
| Input | binarised measurement | raw measurement |
| Parameters | detrending, binarisation threshold, `eps`, `eps_prev`, `min_clustersize`, `n_prev` | `epsilon` |
| Neighbourhood | distance `eps` | Delaunay neighbours per frame |
| Event size threshold | `min_clustersize` | calibrated so that chance groups this large occur in fewer than `epsilon` movies |
| Waves that touch | joined if within `eps` | kept separate if they started at different times |

Use it when the signal is continuous (e.g. ERK activity), when the parameters of the binarised route are hard
to choose, or when the same event definition should apply across movies with different noise levels.

## Usage

```python
from arcos4py.tools import detect_coactivation_events

events = detect_coactivation_events(
    df,                          # long table: one row per cell and frame
    position_columns=["x", "y"],
    frame_column="t",
    obj_id_column="track_id",
    measurement_column="erk",    # continuous signal, not binarised
)
# rows that belong to an event, event id in "collid" -- the same format as track_events_dataframe
```

With the `ARCOS` class (no `bin_measurements` needed):

```python
from arcos4py import ARCOS

ts = ARCOS(df, ["x", "y"], "t", "track_id", "erk", "collid")
events = ts.track_coactive_events(epsilon=1)
```

Image stacks (T, Y, X) are processed as square tiles on a fixed grid (tile size chosen from the data):

```python
from arcos4py.tools import detect_coactivation_events_image

labels = detect_coactivation_events_image(stack)   # (T, Y, X) int32, 0 = background
```

To apply one event definition to several movies (replicates, conditions), calibrate once:

```python
from arcos4py.tools import calibrate_coactivation

calib = calibrate_coactivation([df1, df2, df3], ["x", "y"], "t", "track_id", "erk")
events = detect_coactivation_events(df4, ["x", "y"], "t", "track_id", "erk", calibration=calib)
```

`return_report=True` also returns a `CoactivationReport` with what was chosen for the movie (co-activation level,
onset window, minimum event size, baseline window, which events touch, merges).

## How it works

1. **Activity.** For every cell, dR = 3-frame mean of the signal minus a baseline estimated only from frames
   where the cell is at rest: drift shared by all cells (bleaching, focus; each cell with its own scale) plus
   the cell's own slow drift (a robust local line). A pulse of the whole field is kept, not taken for drift.
2. **Co-activation score.** L = min(own dR, mean dR of the cell's neighbours in that frame): high only if a cell
   and its neighbourhood rise together.
3. **Null.** Every cell's dR is circularly shifted in time by its own random offset. Each cell keeps its pulses
   and noise; only the coincidence between neighbours is destroyed.
4. **Pulses and onset groups.** Runs of frames with L above a null quantile are pulses. Pulses of neighbouring
   cells are linked if their onsets differ by at most delta frames, the largest lag at which such pairs are at
   least twice as common as in the null.
5. **Significance.** A group is an event if chance groups that large are expected in fewer than `epsilon` movies
   (over a small ladder of levels and density cores, Bonferroni-corrected). Events are then grown into weaker
   co-active pulses that are onset-linked to them.
6. **Membership.** Full width at half maximum of each event; each member's first frame is read on the unsmoothed
   signal.
7. **Births.** Inside an event, activations that started separately and are larger than chance stay separate
   events (frame-by-frame tracking, as in ARCOS). `allow_merges=True` joins two waves of one onset group from
   their first contact on, with a lineage table in the report.
8. **Early members removed.** A member that started more than k frames before every neighbouring member of its
   event (k from the movie's own lag statistics) is most often a pulse of its own that touched the wave.

## Performance

The heavy steps (null copies, pulse graph, lag statistics, running medians, cluster sizes, frame-by-frame
tracking, membership) are compiled with numba; the first call after installation includes compilation (cached
afterwards). On one core, a movie of 1,600 tracked cells x 200 frames takes about 5 s, and a 160 x 160 px
image stack of 250 frames about 13 s (three tile sizes tried).

## Limitations

- Waves that start within a few cells of another wave's front at the same time can be merged into it.
- When a whole field switches off at once, a few cells lingering for a frame extend the event by that frame.
