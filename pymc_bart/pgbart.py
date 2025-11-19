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
from pymc_bart.split_rules import ContinuousSplitRule, TargetEncodingSplitRule, CounterEncodingSplitRule
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

        # Initialize target encoding rules with parameters
        self._initialize_target_encoding_rules()

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
            self.target_type = "binary"
        else:
            self.leaf_sd *= self.Y.std() / self.m**0.5
            self.target_type = "regression" if len(y_unique) > 10 else "multiclass"

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
            [ParticleTree(self.a_tree) for _ in range(self.m)] for _ in range(self.trees_shape)
        ]
        self.all_trees = np.array([[p.tree for p in pl] for pl in self.all_particles])
        self.lower = 0
        self.iter = 0
        super().__init__([value_bart], shared)

    def _initialize_target_encoding_rules(self):
        """Initialize target encoding rules with appropriate parameters."""
        for i, rule in enumerate(self.split_rules):
            if isinstance(rule, type) and issubclass(rule, (TargetEncodingSplitRule, CounterEncodingSplitRule)):
                # Replace class with instance and set parameters
                if rule == TargetEncodingSplitRule:
                    self.split_rules[i] = TargetEncodingSplitRule(
                        prior=1.0,
                        noise_level=0.01,
                        target_type=self._get_target_type(),
                        n_buckets=10
                    )
                elif rule == CounterEncodingSplitRule:
                    self.split_rules[i] = CounterEncodingSplitRule(
                        prior=1.0,
                        calculation_method="Full"
                    )

    def _get_target_type(self) -> str:
        """Determine target type for encoding."""
        y_unique = np.unique(self.Y)
        if len(y_unique) == 2 and np.all(y_unique == [0, 1]):
            return "binary"
        elif len(y_unique) > 10:
            return "regression"
        else:
            return "multiclass"

    def astep(self, _):
        variable_inclusion = np.zeros(self.num_variates, dtype="int")

        upper = min(self.lower + self.batch[not self.tune], self.m)
        tree_ids = range(self.lower, upper)
        self.lower = upper if upper < self.m else 0

        for odim in range(self.trees_shape):
            for tree_id in tree_ids:
                self.iter += 1
                # Compute the sum of trees without the old tree that we are attempting to replace
                self.sum_trees_noi[odim] = (
                    self.sum_trees[odim] - self.all_particles[odim][tree_id].tree._predict()
                )
                # Generate an initial set of particles
                # at the end we return one of these particles as the new tree
                particles = self.init_particles(tree_id, odim)

                while True:
                    # Sample each particle (try to grow each tree), except for the first one
                    stop_growing = True
                    for p in particles[1:]:
                        if p.sample_tree(
                            self.ssv,
                            self.available_predictors,
                            self.prior_prob_leaf_node,
                            self.X,
                            self.missing_data,
                            self.sum_trees[odim],
                            self.leaf_sd[odim],
                            self.m,
                            self.response,
                            self.normal,
                            self.leaves_shape,
                        ):
                            self.update_weight(p, odim)
                        if p.expansion_nodes:
                            stop_growing = False
                    if stop_growing:
                        break

                    # Normalize weights
                    normalized_weights = self.normalize(particles[1:])

                    # Resample
                    particles = self.resample(particles, normalized_weights)

                normalized_weights = self.normalize(particles)
                # Get the new particle and associated tree
                self.all_particles[odim][tree_id], new_tree = self.get_particle_tree(
                    particles, normalized_weights
                )
                # Update the sum of trees
                new = new_tree._predict()
                self.sum_trees[odim] = self.sum_trees_noi[odim] + new
                # To reduce memory usage, we trim the tree
                self.all_trees[odim][tree_id] = new_tree.trim()

                if self.tune:
                    # Update the splitting variable and the splitting variable sampler
                    if self.iter > self.m:
                        self.ssv = SampleSplittingVariable(self.alpha_vec)

                    for index in new_tree.get_split_variables():
                        self.alpha_vec[index] += 1

                    # update standard deviation at leaf nodes
                    if self.iter > 2:
                        self.leaf_sd[odim] = self.running_sd[odim].update(new)
                    else:
                        self.running_sd[odim].update(new)

                else:
                    # update the variable inclusion
                    for index in new_tree.get_split_variables():
                        variable_inclusion[index] += 1

        if not self.tune:
            self.bart.all_trees.append(self.all_trees)

        variable_inclusion = _encode_vi(variable_inclusion)

        stats = {"variable_inclusion": variable_inclusion, "tune": self.tune}
        return self.sum_trees, [stats]

    def normalize(self, particles: list[ParticleTree]) -> float:
        """
        Use softmax to get normalized_weights.
        """
        log_w = np.array([p.log_weight for p in particles])
        log_w_max = log_w.max()
        log_w_ = log_w - log_w_max
        wei = np.exp(log_w_) + 1e-12
        return wei / wei.sum()

    def resample(
        self, particles: list[ParticleTree], normalized_weights: npt.NDArray
    ) -> list[ParticleTree]:
        """
        Use systematic resample for all but the first particle

        Ensure particles are copied only if needed.
        """
        new_indices = self.systematic(normalized_weights) + 1
        seen: list[int] = []
        new_particles: list[ParticleTree] = []
        for idx in new_indices:
            if idx in seen:
                new_particles.append(particles[idx].copy())
            else:
                new_particles.append(particles[idx])
                seen.append(int(idx))

        particles[1:] = new_particles

        return particles

    def get_particle_tree(
        self, particles: list[ParticleTree], normalized_weights: npt.NDArray
    ) -> tuple[ParticleTree, Tree]:
        """
        Sample a new particle and associated tree
        """
        new_index = self.systematic(normalized_weights)[
            discrete_uniform_sampler(self.num_particles)
        ]
        new_particle = particles[new_index]

        return new_particle, new_particle.tree

    def systematic(self, normalized_weights: npt.NDArray) -> npt.NDArray[np.int_]:
        """
        Systematic resampling.

        Return indices in the range 0, ..., len(normalized_weights)

        Note: adapted from https://github.com/nchopin/particles
        """
        lnw = len(normalized_weights)
        single_uniform = (self.uniform.rvs() + np.arange(lnw)) / lnw
        return inverse_cdf(single_uniform, normalized_weights)

    def init_particles(self, tree_id: int, odim: int) -> list[ParticleTree]:
        """Initialize particles."""
        p0: ParticleTree = self.all_particles[odim][tree_id]
        # The old tree does not grow so we update the weight only once
        self.update_weight(p0, odim)
        particles: list[ParticleTree] = [p0]

        particles.extend(ParticleTree(self.a_tree) for _ in self.indices)
        return particles

    def update_weight(self, particle: ParticleTree, odim: int) -> None:
        """
        Update the weight of a particle.
        """

        delta = (
            np.identity(self.trees_shape)[odim][:, None, None]
            * particle.tree._predict()[None, :, :]
        )

        new_likelihood = self.likelihood_logp((self.sum_trees_noi + delta).flatten())
        particle.log_weight = new_likelihood

    @staticmethod
    def competence(var: pm.Distribution, has_grad: bool) -> Competence:
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    @staticmethod
    def _make_update_stats_functions():
        def update_stats(step_stats):
            return {key: step_stats[key] for key in ("variable_inclusion", "tune")}

        return (update_stats,)


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

    # For target encoding rules, pass additional parameters
    if isinstance(split_rule, (TargetEncodingSplitRule, CounterEncodingSplitRule)):
        # Get current residuals for target encoding
        current_residuals = sum_trees[:, idx_data_points]
        
        if isinstance(split_rule, TargetEncodingSplitRule):
            split_value = split_rule.get_split_value(
                available_splitting_values, 
                targets=tree.Y if hasattr(tree, 'Y') else None,
                residuals=current_residuals.flatten() if current_residuals.size > 0 else None
            )
        else:  # CounterEncodingSplitRule
            split_value = split_rule.get_split_value(
                available_splitting_values,
                training_categories=X[:, selected_predictor] if hasattr(tree, 'X') else None
            )
    else:
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


# Остальной код остается без изменений...
# [Keep all the existing helper classes and functions from the original file]
