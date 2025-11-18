#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
from numba import njit
from pymc.initial_point import PointType
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor import config
from pytensor import function as pytensor_function
from pytensor.tensor.variable import Variable

from pymc_bart.bart import BARTRV
from pymc_bart.split_rules import ContinuousSplitRule
from pymc_bart.tree import (
    Node,
    Tree,
    get_depth,
    get_idx_left_child,
    get_idx_right_child,
)
from pymc_bart.utils import _encode_vi


class ParticleTree:
    """Particle tree."""

    __slots__ = "tree", "expansion_nodes", "log_weight"

    def __init__(self, tree: Tree):
        self.tree: Tree = tree.copy()
        self.expansion_nodes: list[int] = [0]
        self.log_weight: float = 0

    def copy(self) -> "ParticleTree":
        p = ParticleTree(self.tree)
        p.expansion_nodes = self.expansion_nodes.copy()
        return p

    def sample_tree(
        self,
        ssv,
        available_predictors,
        prior_prob_leaf_node,
        X,
        missing_data,
        sum_trees,
        leaf_sd,
        m,
        response,
        normal,
        shape,
    ) -> bool:
        tree_grew = False
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            prob_leaf = prior_prob_leaf_node[get_depth(index_leaf_node)]

            if prob_leaf < np.random.random():
                idx_new_nodes = grow_tree(
                    self.tree,
                    index_leaf_node,
                    ssv,
                    available_predictors,
                    X,
                    missing_data,
                    sum_trees,
                    leaf_sd,
                    m,
                    response,
                    normal,
                    shape,
                )
                if idx_new_nodes is not None:
                    self.expansion_nodes.extend(idx_new_nodes)
                    tree_grew = True

        return tree_grew


class PGBART(ArrayStepShared):
    """
    Particle Gibss BART sampling step.

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : tuple
        Number of particles. Defaults to 10
    batch : tuple
        Number of trees fitted per step. The first element is the batch size during tuning and the
        second the batch size after tuning.  Defaults to  (0.1, 0.1), meaning 10% of the `m` trees
        during tuning and after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    name = "pgbart"
    default_blocked = False
    generates_stats = True
    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "variable_inclusion": (object, []),
        "tune": (bool, []),
    }

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        vars: list[pm.Distribution] | None = None,
        num_particles: int = 10,
        batch: tuple[float, float] = (0.1, 0.1),
        model: Model | None = None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,
        **kwargs,  # Accept additional kwargs for compound sampling
    ) -> None:
        model = modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)

        if vars is None:
            raise ValueError("Unable to find variables to sample")

        # Filter to only BART variables
        bart_vars = []
        for var in vars:
            rv = model.values_to_rvs.get(var)
            if rv is not None and isinstance(rv.owner.op, BARTRV):
                bart_vars.append(var)

        if not bart_vars:
            raise ValueError("No BART variables found in the provided variables")

        if len(bart_vars) > 1:
            raise ValueError(
                "PGBART can only handle one BART variable at a time. "
                "For multiple BART variables, PyMC will automatically create "
                "separate PGBART samplers for each variable."
            )

        value_bart = bart_vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        if isinstance(self.bart.X, Variable):
            self.X = self.bart.X.eval()
        else:
            self.X = self.bart.X

        if isinstance(self.bart.Y, Variable):
            self.Y = self.bart.Y.eval()
        else:
            self.Y = self.bart.Y

        self.missing_data = np.any(np.isnan(self.X))
        self.m = self.bart.m
        self.response = self.bart.response

        shape = initial_point[value_bart.name].shape

        self.shape = 1 if len(shape) == 1 else shape[0]

        # Set trees_shape (dim for separate tree structures)
        # and leaves_shape (dim for leaf node values)
        # One of the two is always one, the other equal to self.shape
        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1

        if self.bart.split_prior.size == 0:
            self.alpha_vec = np.ones(self.X.shape[1])
        else:
            self.alpha_vec = self.bart.split_prior

        if self.bart.split_rules:
            self.split_rules = self.bart.split_rules
        else:
            self.split_rules = [ContinuousSplitRule] * self.X.shape[1]

        for idx, rule in enumerate(self.split_rules):
            if rule is ContinuousSplitRule:
                self.X[:, idx] = jitter_duplicated(self.X[:, idx], np.nanstd(self.X[:, idx]))

        init_mean = self.Y.mean()
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))

        # if data is binary
        self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape))

        y_unique = np.unique(self.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            self.leaf_sd *= 3 / self.m**0.5
        else:
            self.leaf_sd *= self.Y.std() / self.m**0.5

        self.running_sd = [
            RunningSd((self.leaves_shape, self.num_observations)) for _ in range(self.trees_shape)
        ]

        self.sum_trees = np.full(
            (self.trees_shape, self.leaves_shape, self.Y.shape[0]), init_mean
        ).astype(config.floatX)
        self.sum_trees_noi = self.sum_trees - init_mean
        self.a_tree = Tree.new_tree(
            leaf_node_value=init_mean / self.m,
            idx_data_points=np.arange(self.num_observations, dtype="int32"),
            num_observations=self.num_observations,
            shape=self.leaves_shape,
            split_rules=self.split_rules,
        )

        self.normal = NormalSampler(1, self.leaves_shape)
        self.uniform = UniformSampler(0, 1)
        self.prior_prob_leaf_node = compute_prior_probability(self.bart.alpha, self.bart.beta)
        self.ssv = SampleSplittingVariable(self.alpha_vec)

        self.tune = True

        batch_0 = max(1, int(self.m * batch[0]))
        batch_1 = max(1, int(self.m * batch[1]))
        self.batch = (batch_0, batch_1)

        self.num_particles = num_particles
        self.indices = list(range(1, num_particles))
        shared = make_shared_replacements(initial_point, [value_bart], model)
        self.likelihood_logp = logp(initial_point, [model.datalogp], [value_bart], shared)
        self.all_particles = [
            [ParticleTree(self.a_tree) for _ in range(self.trees_shape)]
            for _ in range(num_particles)
        ]
        self.selected_trees = [0] * self.trees_shape

        super().__init__([value_bart], shared, **kwargs)

    def astep(self, q0: npt.NDArray):
        variable_inclusion = [[] for _ in range(self.trees_shape)]

        if self.tune:
            batch = self.batch[0]
        else:
            batch = self.batch[1]

        for t_dim in range(self.trees_shape):
            # Update all_particles
            for p in range(1, self.num_particles):
                self.all_particles[p][t_dim] = self.all_particles[0][t_dim].copy()

            # Sample new trees
            for _ in range(batch):
                for p in range(self.num_particles):
                    self.all_particles[p][t_dim].sample_tree(
                        self.ssv,
                        self.available_predictors,
                        self.prior_prob_leaf_node,
                        self.X,
                        self.missing_data,
                        self.sum_trees_noi[t_dim],
                        self.leaf_sd[t_dim],
                        self.m,
                        self.response,
                        self.normal,
                        self.leaves_shape,
                    )

            # Compute log_weights
            log_weights = np.zeros(self.num_particles)
            for p in range(self.num_particles):
                self.all_particles[p][t_dim].output = self.all_particles[p][t_dim]._predict()
                log_weights[p] = self.likelihood_logp(self.sum_trees_noi[t_dim] + self.all_particles[p][t_dim].output)

            # Normalize log_weights
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= weights.sum()

            # Resample particles
            new_indices = inverse_cdf(self.uniform.rvs(size=self.num_particles), weights)
            new_particles = [self.all_particles[i][t_dim].copy() for i in new_indices]

            self.all_particles = new_particles

            # Select tree
            self.selected_trees[t_dim] = np.random.choice(range(self.num_particles))

            # Update sum_trees
            self.sum_trees[t_dim] = self.sum_trees_noi[t_dim] + self.all_particles[self.selected_trees[t_dim]].output

            # Update running_sd
            self.running_sd[t_dim].update(self.all_particles[self.selected_trees[t_dim]].output)

            # Update variable_inclusion
            for tree in self.all_trees[t_dim]:
                for var in tree.get_split_variables():
                    variable_inclusion[t_dim].append(var)

        # Update all_trees
        for t_dim in range(self.trees_shape):
            self.all_trees[t_dim].append(self.all_particles[self.selected_trees[t_dim]].trim())

        stats = {
            "variable_inclusion": _encode_vi(variable_inclusion),
            "tune": self.tune,
        }

        return q0, stats

    @staticmethod
    def competence(var: TensorVariable, has_grad: bool) -> Competence:
        if var.owner is None:
            return Competence.INCOMPATIBLE
        op = var.owner.op
        if isinstance(op, BARTRV):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


def compute_prior_probability(alpha: int, beta: int) -> list[float]:
    """
    Calculate the probability of the node being a leaf node (1 - p(being split node)).

    Parameters
    ----------
    alpha : float
    beta: float

    Returns
    -------
    list with probabilities for leaf nodes
    """
    prior_leaf_prob: list[float] = [0]
    depth = 0
    while prior_leaf_prob[-1] < 0.9999:
        prior_leaf_prob.append(1 - (alpha * ((1 + depth) ** (-beta))))
        depth += 1
    prior_leaf_prob.append(1)

    return prior_leaf_prob


class SampleSplittingVariable:
    def __init__(self, alpha_vec: npt.NDArray):
        self.enu = list(enumerate(np.cumsum(alpha_vec / alpha_vec.sum())))

    def rvs(self) -> int | tuple[int, float]:
        rnd: float = np.random.random()
        for i, val in self.enu:
            if rnd <= val:
                return i
        return self.enu[-1]


def grow_tree(
    tree,
    index_leaf_node,
    ssv,
    available_predictors,
    X,
    missing_data,
    sum_trees,
    leaf_sd,
    m,
    response,
    normal,
    shape,
):
    current_node = tree.get_node(index_leaf_node)
    idx_data_points = current_node.idx_data_points

    index_selected_predictor = ssv.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    idx_data_points, available_splitting_values = filter_missing_values(
        X[idx_data_points, selected_predictor], idx_data_points, missing_data
    )

    split_rule = tree.split_rules[selected_predictor]

    # Добавлено: Логика для TargetMeanSplitRule
    if isinstance(split_rule, TargetMeanSplitRule):
        # Вычисляем y_for_split как среднее по residuals (sum_trees)
        y_for_split = np.mean(sum_trees[:, :, idx_data_points], axis=(0, 1)) if shape > 1 else sum_trees[0, 0, idx_data_points]

        unique_cats = np.unique(available_splitting_values)
        if len(unique_cats) <= 1:
            return None

        encoded = np.zeros_like(available_splitting_values, dtype=float)
        global_mean = np.mean(y_for_split)
        alpha = split_rule.smoothing_alpha  # Используем smoothing_alpha из правила

        for cat in unique_cats:
            mask = (available_splitting_values == cat)
            cat_mean = np.mean(y_for_split[mask]) if mask.sum() > 0 else global_mean
            n = mask.sum()
            smoothed_mean = (n * cat_mean + alpha * global_mean) / (n + alpha)
            encoded[mask] = smoothed_mean

        split_value = np.mean(encoded)
        to_left = encoded <= split_value
    else:
        # Стандартная логика для других правил
        split_value = split_rule.get_split_value(available_splitting_values)
        if split_value is None:
            return None
        to_left = split_rule.divide(available_splitting_values, split_value)

    new_idx_data_points = idx_data_points[to_left], idx_data_points[~to_left]

    current_node_children = (
        get_idx_left_child(index_leaf_node),
        get_idx_right_child(index_leaf_node),
    )

    if response == "mix":
        response = "linear" if np.random.random() >= 0.5 else "constant"

    for idx in range(2):
        idx_data_point = new_idx_data_points[idx]
        node_value, linear_params = draw_leaf_value(
            y_mu_pred=sum_trees[:, idx_data_point],
            x_mu=X[idx_data_point, selected_predictor],
            m=m,
            norm=normal.rvs() * leaf_sd,
            shape=shape,
            response=response,
        )

        new_node = Node.new_leaf_node(
            value=node_value,
            nvalue=len(idx_data_point),
            idx_data_points=idx_data_point,
            linear_params=linear_params,
        )
        tree.set_node(current_node_children[idx], new_node)

    tree.grow_leaf_node(current_node, selected_predictor, split_value, index_leaf_node)
    return current_node_children


def filter_missing_values(available_splitting_values, idx_data_points, missing_data):
    if missing_data:
        mask = ~np.isnan(available_splitting_values)
        idx_data_points = idx_data_points[mask]
        available_splitting_values = available_splitting_values[mask]
    return idx_data_points, available_splitting_values


def draw_leaf_value(
    y_mu_pred: npt.NDArray,
    x_mu: npt.NDArray,
    m: int,
    norm: npt.NDArray,
    shape: int,
    response: str,
) -> tuple[npt.NDArray, npt.NDArray | None]:
    """Draw Gaussian distributed leaf values."""
    linear_params = None
    mu_mean: npt.NDArray
    if y_mu_pred.size == 0:
        return np.zeros(shape), linear_params

    if y_mu_pred.size == 1:
        mu_mean = np.full(shape, y_mu_pred.item() / m) + norm
    elif y_mu_pred.size < 3 or response == "constant":
        mu_mean = fast_mean(y_mu_pred) / m + norm
    else:
        mu_mean, linear_params = fast_linear_fit(x=x_mu, y=y_mu_pred, m=m, norm=norm)

    return mu_mean, linear_params


@njit
def fast_mean(ari: npt.NDArray) -> float | npt.NDArray:
    """Use Numba to speed up the computation of the mean."""
    if ari.ndim == 1:
        count = ari.shape[0]
        suma = 0
        for i in range(count):
            suma += ari[i]
        return suma / count
    else:
        res = np.zeros(ari.shape[0])
        count = ari.shape[1]
        for j in range(ari.shape[0]):
            for i in range(count):
                res[j] += ari[j, i]
        return res / count


@njit
def fast_linear_fit(
    x: npt.NDArray,
    y: npt.NDArray,
    m: int,
    norm: npt.NDArray,
) -> tuple[npt.NDArray, list[npt.NDArray]]:
    n = len(x)
    y = y / m + np.expand_dims(norm, axis=1)

    xbar = np.sum(x) / n
    ybar = np.sum(y, axis=1) / n

    x_diff = x - xbar
    y_diff = y - np.expand_dims(ybar, axis=1)

    x_var = np.dot(x_diff, x_diff.T)

    if x_var == 0:
        b = np.zeros(y.shape[0])
    else:
        b = np.dot(x_diff, y_diff.T) / x_var

    a = ybar - b * xbar

    y_fit = np.expand_dims(a, axis=1) + np.expand_dims(b, axis=1) * x
    return y_fit.T, [a, b]


def discrete_uniform_sampler(upper_value):
    """Draw from the uniform distribution with bounds [0, upper_value).

    This is the same and np.random.randit(upper_value) but faster.
    """
    return int(np.random.random() * upper_value)


class NormalSampler:
    """Cache samples from a standard normal distribution."""

    def __init__(self, scale, shape):
        self.size = 1000
        self.scale = scale
        self.shape = shape
        self.update()

    def rvs(self):
        if self.idx == self.size:
            self.update()
        pop = self.cache[:, self.idx]
        self.idx += 1
        return pop

    def update(self):
        self.idx = 0
        self.cache = np.random.normal(loc=0.0, scale=self.scale, size=(self.shape, self.size))


class UniformSampler:
    """Cache samples from a uniform distribution."""

    def __init__(self, lower_bound, upper_bound, shape=None):
        self.size = 1000
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.shape = shape
        self.update()

    def rvs(self):
        if self.idx == self.size:
            self.update()
        if self.shape is None:
            pop = self.cache[self.idx]
        else:
            pop = self.cache[:, self.idx]
        self.idx += 1
        return pop

    def update(self):
        self.idx = 0
        if self.shape is None:
            self.cache = np.random.uniform(self.lower_bound, self.upper_bound, size=self.size)
        else:
            self.cache = np.random.uniform(
                self.lower_bound, self.upper_bound, size=(self.shape, self.size)
            )


@njit
def inverse_cdf(
    single_uniform: npt.NDArray, normalized_weights: npt.NDArray
) -> npt.NDArray[np.int_]:
    """
    Inverse CDF algorithm for a finite distribution.

    Parameters
    ----------
    single_uniform: npt.NDArray
        Ordered points in [0,1]

    normalized_weights: npt.NDArray)
        Normalized weights

    Returns
    -------
    new_indices: ndarray
        a vector of indices in range 0, ..., len(normalized_weights)

    Note: adapted from https://github.com/nchopin/particles
    """
    idx = 0
    a_weight = normalized_weights[0]
    sul = len(single_uniform)
    new_indices = np.empty(sul, dtype=np.int64)
    for i in range(sul):
        while single_uniform[i] > a_weight:
            idx += 1
            a_weight += normalized_weights[idx]
        new_indices[i] = idx
    return new_indices


@njit
def jitter_duplicated(array: npt.NDArray, std: float) -> npt.NDArray:
    """
    Jitter duplicated values.
    """
    if are_whole_number(array):
        seen = []
        for idx, num in enumerate(array):
            if num in seen and not np.isnan(num):
                array[idx] = num + np.random.normal(0, std / 12)
            else:
                seen.append(num)

    return array


@njit
def are_whole_number(array: npt.NDArray) -> np.bool_:
    """Check if all values in array are whole numbers"""
    return np.all(np.mod(array[~np.isnan(array)], 1) == 0)


def logp(
    point,
    out_vars: list[pm.Distribution],
    vars: list[pm.Distribution],
    shared: list[pt.TensorVariable],
):
    """Compile PyTensor function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc.Distribution` for the output variables
    vars: List
        containing :class:`pymc.Distribution` for the input variables
    shared: List
        containing :class:`pytensor.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    function = pytensor_function([inarray0], out_list[0])
    function.trust_input = True
    return function


class RunningSd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.var = np.zeros(shape)

    def update(self, x):
        self.n += 1
        old_mean = self.mean.copy()
        self.mean = old_mean + (x - old_mean) / self.n
        self.var = self.var + (x - old_mean) * (x - self.mean)
