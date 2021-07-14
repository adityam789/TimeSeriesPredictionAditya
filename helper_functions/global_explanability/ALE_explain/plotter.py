from typing import List, Literal, Optional, Union, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import count

# no_type_check is needed because exp is a generic explanation and so mypy doesn't know that the
# attributes actually exist... As a side effect the type information does not show up in the static
# docs. Will need to re-think this.
@no_type_check
def plotter_ALE(exp,
             features: Union[List[Union[int, str]], Literal['all']] = 'all',
             targets: Union[List[Union[int, str]], Literal['all']] = 'all',
             n_cols: int = 3,
             sharey: str = 'all',
             constant: bool = False,
             ax: Union['plt.Axes', np.ndarray, None] = None,
             line_kw: Optional[dict] = None,
             fig_kw: Optional[dict] = None) -> 'np.ndarray':
    """
    Plot ALE curves on matplotlib axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the `ALE.explain` method.
    features
        A list of features for which to plot the ALE curves or `all` for all features.
        Can be a mix of integers denoting feature index or strings denoting entries in
        `exp.feature_names`. Defaults to 'all'.
    targets
        A list of targets for which to plot the ALE curves or `all` for all targets.
        Can be a mix of integers denoting target index or strings denoting entries in
        `exp.target_names`. Defaults to 'all'.
    n_cols
        Number of columns to organize the resulting plot into.
    sharey
        A parameter specifying whether the y-axis of the ALE curves should be on the same scale
        for several features. Possible values are `all`, `row`, `None`.
    constant
        A parameter specifying whether the constant zeroth order effects should be added to the
        ALE first order effects.
    ax
        A `matplotlib` axes object or a numpy array of `matplotlib` axes to plot on.
    line_kw
        Keyword arguments passed to the `plt.plot` function.
    fig_kw
        Keyword arguments passed to the `fig.set` function.

    Returns
    -------
    An array of matplotlib axes with the resulting ALE plots.

    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # line_kw and fig_kw values
    default_line_kw = {'markersize': 3, 'marker': 'o', 'label': None}
    if line_kw is None:
        line_kw = {}
    line_kw = {**default_line_kw, **line_kw}

    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}
    fig_kw = {**default_fig_kw, **fig_kw}

    if features == 'all':
        features = range(0, len(exp.feature_names))
    else:
        for ix, f in enumerate(features):
            if isinstance(f, str):
                try:
                    f = np.argwhere(exp.feature_names == f).item()
                except ValueError:
                    raise ValueError(f"Feature name {f} does not exist.")
            features[ix] = f
    n_features = len(features)

    if targets == 'all':
        targets = range(0, len(exp.target_names))
    else:
        for ix, t in enumerate(targets):
            if isinstance(t, str):
                try:
                    t = np.argwhere(exp.target_names == t).item()
                except ValueError:
                    raise ValueError(f"Target name {t} does not exist.")
            targets[ix] = t

    # make axes
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(ax, plt.Axes) and n_features != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_features)
        n_rows = math.ceil(n_features / n_cols)

        axes = np.empty((n_rows, n_cols), dtype=np.object)
        axes_ravel = axes.ravel()
        # gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=ax.get_subplotspec())
        gs = GridSpec(n_rows, n_cols)
        for i, spec in zip(range(n_features), gs):
            # determine which y-axes should be shared
            if sharey == 'all':
                cond = i != 0
            elif sharey == 'row':
                cond = i % n_cols != 0
            else:
                cond = False

            if cond:
                axes_ravel[i] = fig.add_subplot(spec, sharey=axes_ravel[i - 1])
                continue
            axes_ravel[i] = fig.add_subplot(spec)

    else:  # array-like
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)
        if ax.size < n_features:
            raise ValueError(
                f"Expected ax to have {n_features} axes, got {ax.size}")
        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    # make plots
    for ix, feature, ax_ravel in \
            zip(count(), features, axes_ravel):
        _ = _plot_one_ale_num(exp=exp,
                              feature=feature,
                              targets=targets,
                              constant=constant,
                              ax=ax_ravel,
                              legend=not ix,  # only one legend
                              line_kw=line_kw)

    # if explicit labels passed, handle the legend here as the axis passed might be repeated
    if line_kw['label'] is not None:
        axes_ravel[0].legend()

    fig.set(**fig_kw)
    # TODO: should we return just axes or ax + axes
    return axes


@no_type_check
def _plot_one_ale_num(exp,
                      feature: int,
                      targets: List[int],
                      constant: bool = False,
                      ax: 'plt.Axes' = None,
                      legend: bool = True,
                      line_kw: dict = None) -> 'plt.Axes':
    """
    Plots the ALE of exactly one feature on one axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import transforms

    if ax is None:
        ax = plt.gca()

    # add zero baseline
    ax.axhline(0, color='grey')

    lines = ax.plot(
        exp.feature_values[feature],
        exp.ale_values[feature][:, targets] + constant * exp.constant_value,
        **line_kw
    )

    # add decile markers to the bottom of the plot
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(exp.feature_deciles[feature][1:], 0, 0.0005, transform=trans)

    ax.set_xlabel(exp.feature_names[feature])
    ax.set_ylabel('ALE')

    if legend:
        # if no explicit labels passed, just use target names
        if line_kw['label'] is None:
            ax.legend(lines, exp.target_names[targets])

    return ax
