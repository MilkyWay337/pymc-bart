# pylint: disable=too-many-branches
"""Utility function for variable selection and bart interpretability."""

import base64
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
from arviz_base import rcParams
from arviz_stats.base import array_stats
from numba import jit
from pytensor.tensor.variable import Variable
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

from .tree import Tree

TensorLike = TypeVar("TensorLike", npt.NDArray, pt.TensorVariable)


def _sample_posterior(
    all_trees: list[list[Tree]],
    X: TensorLike,
    rng: np.random.Generator,
    size: int | tuple[int, ...] | None = None,
    excluded: list[int] | None = None,
    shape: int = 1,
) -> npt.NDArray:
    """
    Generate samples from the BART-posterior.

    Parameters
    ----------
    all_trees : list
        List of all trees sampled from a posterior
    X : tensor-like
        A covariate matrix. Use the same used to fit BART for in-sample predictions or a new one for
        out-of-sample predictions.
    rng : NumPy RandomGenerator
    size : int or tuple
        Number of samples.
    excluded : Optional[npt.NDArray[np.int_]]
        Indexes of the variables to exclude when computing predictions
    """
    stacked_trees = all_trees

    if isinstance(X, Variable):
        X = X.eval()

    if size is None:
        size_iter: list | tuple = (1,)
    elif isinstance(size, int):
        size_iter = [size]
    else:
        size_iter = size

    flatten_size = 1
    for s in size_iter:
        flatten_size *= s

    idx = rng.integers(0, len(stacked_trees), size=flatten_size)

    trees_shape = len(stacked_trees[0])
    leaves_shape = shape // trees_shape

    pred = np.zeros((flatten_size, trees_shape, leaves_shape, X.shape[0]))

    for ind, p in enumerate(pred):
        for odim, odim_trees in enumerate(stacked_trees[idx[ind]]):
            for tree in odim_trees:
                p[odim] += tree.predict(x=X, excluded=excluded, shape=leaves_shape)

    return pred.transpose((0, 3, 1, 2)).reshape((*size_iter, -1, shape))


def plot_convergence(
    idata: Any,
    var_name: str | None = None,
    kind: str = "ecdf",
    figsize: tuple[float, float] | None = None,
    ax=None,
) -> None:
    """
    Plot convergence diagnostics.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object containing the posterior samples.
    var_name : Optional[str]
        Name of the BART variable to plot. Defaults to None.
    kind : str
        Type of plot to display. Options are "ecdf" (default) and "kde".
    figsize : Optional[tuple[float, float]], by default None.
        Figure size. Defaults to None.
    ax : matplotlib axes
        Axes on which to plot. Defaults to None.

    Returns
    -------
    list[ax] : matplotlib axes
    """
    warnings.warn(
        "This function has been deprecated"
        "Use az.plot_convergence_dist() instead."
        "https://arviz-plots.readthedocs.io/en/latest/api/generated/arviz_plots.plot_convergence_dist.html",
        FutureWarning,
    )


def plot_ice(
    bartrv: Variable,
    X: npt.NDArray,
    Y: npt.NDArray | None = None,
    var_idx: list[int] | None = None,
    var_discrete: list[int] | None = None,
    func: Callable | None = None,
    centered: bool | None = True,
    samples: int = 100,
    instances: int = 30,
    random_seed: int | None = None,
    sharey: bool = True,
    smooth: bool = True,
    grid: str = "long",
    color="C0",
    color_mean: str = "C0",
    alpha: float = 0.1,
    figsize: tuple[float, float] | None = None,
    smooth_kwargs: dict[str, Any] | None = None,
    ax: plt.Axes | None = None,
) -> list[plt.Axes]:
    """
    Individual conditional expectation plot.

    Parameters
    ----------
    bartrv : BART Random Variable
        BART variable once the model that include it has been fitted.
    X : npt.NDArray
        The covariate matrix.
    Y : Optional[npt.NDArray], by default None.
        The response vector.
    var_idx : Optional[list[int]], by default None.
        List of the indices of the covariate for which to compute the pdp or ice.
    var_discrete : Optional[list[int]], by default None.
        List of the indices of the covariate treated as discrete.
    func : Optional[Callable], by default None.
        Arbitrary function to apply to the predictions. Defaults to the identity function.
    centered : bool
        If True the result is centered around the partial response evaluated at the lowest value in
        ``xs_interval``. Defaults to True.
    samples : int
        Number of posterior samples used in the predictions. Defaults to 100
    instances : int
        Number of instances of X to plot. Defaults to 30.
    random_seed : Optional[int], by default None.
        Seed used to sample from the posterior. Defaults to None.
    sharey : bool
        Controls sharing of properties among y-axes. Defaults to True.
    smooth : bool
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
        Defaults to True.
    grid : str or tuple
        How to arrange the subplots. Defaults to "long", one subplot below the other.
        Other options are "wide", one subplot next to each other or a tuple indicating the number of
        rows and columns.
    color : matplotlib valid color
        Color used to plot the pdp or ice. Defaults to "C0"
    color_mean : matplotlib valid color
        Color used to plot the mean pdp or ice. Defaults to "C0",
    alpha : float
        Transparency level, should in the interval [0, 1].
    figsize : tuple
        Figure size. If None it will be defined automatically.
    smooth_kwargs : dict
        Additional keywords modifying the Savitzky-Golay filter.
        See scipy.signal.savgol_filter() for details.
    ax : axes
        Matplotlib axes.

    Returns
    -------
    axes: matplotlib axes
    """
    all_trees = bartrv.owner.op.all_trees
    rng = np.random.default_rng(random_seed)

    if func is None:

        def identity(x):
            return x

        func = identity

    (
        X,
        x_labels,
        y_label,
        indices,
        var_idx,
        var_discrete,
        _,
        _,
    ) = _prepare_plot_data(X, Y, "linear", None, var_idx, var_discrete)

    fig, axes, shape = _create_figure_axes(bartrv, var_idx, grid, sharey, figsize, ax)

    instances_ary = rng.choice(range(X.shape[0]), replace=False, size=instances)
    idx_s = list(range(X.shape[0]))

    count = 0
    for i_var, var in enumerate(var_idx):
        indices_mi = indices[:]
        indices_mi.remove(var)
        y_pred = []
        for instance in instances_ary:
            fake_X = X[idx_s]
            fake_X[:, indices_mi] = X[:, indices_mi][instance]
            y_pred.append(
                np.mean(
                    _sample_posterior(all_trees, X=fake_X, rng=rng, size=samples, shape=shape),
                    0,
                )
            )

        new_x = fake_X[:, var]
        p_d = func(np.array(y_pred))

        for s_i in range(shape):
            if centered:
                p_di = p_d[:, :, s_i] - p_d[:, :, s_i][:, 0][:, None]
            else:
                p_di = p_d[:, :, s_i]
            if var in var_discrete:
                axes[count].plot(new_x, p_di.mean(0), "o", color=color_mean)
                axes[count].plot(new_x, p_di.T, ".", color=color, alpha=alpha)
            elif smooth:
                x_data, y_data = _smooth_mean(new_x, p_di, "ice", smooth_kwargs)
                axes[count].plot(x_data, y_data.mean(1), color=color_mean)
                axes[count].plot(x_data, y_data, color=color, alpha=alpha)
            else:
                idx = np.argsort(new_x)
                axes[count].plot(new_x[idx], p_di.mean(0)[idx], color=color_mean)
                axes[count].plot(new_x[idx], p_di.T[:, idx], color=color, alpha=alpha)

            axes[count].set_title(x_labels[var])
            axes[count].set_ylabel(y_label)
            count += 1

    return axes

# ... (остальной код utils.py без изменений, усечён для краткости)
