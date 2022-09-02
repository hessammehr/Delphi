from __future__ import annotations

import itertools
from itertools import permutations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from numpyro import deterministic, sample

from oraclehub.model import Model as M


def indices(N: int, ndims: int, allow_repeat=False) -> np.ndarray:
    """
    Generates an array `A` of indices where the permutations of `A[n]` give the
    unique non-zero indices of a symmetric rank `ndims` tensor with dimensions
    `N`×`N`×...×`N` (`ndims` times).

    >>> A = indices(4, 3) # 4×4×4 tensor
    >>> A
    array([[0, 1, 2],
           [0, 1, 3],
           [0, 2, 3],
           [1, 2, 3]])

    Note that any permutation of the indices would correspond to the same linear
    index since the tensor is symmetric. `A[[0, 1, 2]] == A[[1, 2, 0]]` etc.
    """
    if allow_repeat:
        combs = itertools.combinations_with_replacement(range(N), ndims)
    else:
        combs = itertools.combinations(range(N), ndims)
    return np.array(list(combs))


def triangular_indices(N, ndims, shift=0):
    """
    N: is the number of properties len(v) =  N*(N-1)*...*(N-(ndim-1))/ndim!
    ndim: number of tensor dimensions
    """
    idx = indices(N, ndims)
    t = np.zeros(tuple(N for _ in range(ndims)), dtype="int")
    for i, ind in enumerate(idx):
        for perm in permutations(ind):
            t[perm] = i + shift + 1
    return t


def unique_reactions(df: pd.DataFrame, compound_cols, event_names):
    lookup = df.reset_index().set_index(compound_cols)
    for cmp in lookup.index:
        sub_combinations = []
        for size in range(2, len([i for i in cmp if i != -1])):
            for comb in itertools.combinations(cmp, size):
                if -1 not in comb:
                    comb = comb + (-1,) * (len(cmp) - len(comb))
                    sub_combinations.append(comb)
        sub_reactions = lookup.loc[sub_combinations, event_names].dropna()
        if sub_reactions.empty:
            continue
        lookup.loc[cmp, event_names] = (
            lookup.loc[cmp, event_names] - sub_reactions.max(axis=0)
        ).clip(lower=0.0)

    return lookup.reset_index().set_index("index")


class Model(M):
    metadata = {
        "N_props": "Number of properties",
        "mem_beta_a": (1.0, ""),
        "mem_beta_b": (2.0, ""),
        "react_beta_a": (1.0, ""),
        "react_beta_b": (2.0, ""),
        "likelihood_sd": (0.3, ""),
    }

    def __init__(
        self,
        metadata,
    ):
        super().__init__(metadata)

        # Number of _unique_ reactivities for each reaction arity
        self.N_bin = self.N_props * (self.N_props - 1) // 2
        self.N_tri = self.N_bin * (self.N_props - 2) // 3
        self.N_tet = self.N_tri * (self.N_props - 3) // 4
        self.N = [self.N_bin, self.N_tri, self.N_tet]

        # Indices for 2-, 3-, etc. component reactivities within
        # the sampled vector of _unique_ reactivities
        self.reactivity_indices = [
            triangular_indices(self.N_props, i + 2, shift=sum(self.N[:i]))
            for i in range(len(self.N))
        ]

    def _partition_by_arity(self, data: pd.DataFrame):
        return {
            2: data.query("compound3 == -1"),  # 2-component
            3: data.query("compound3 != -1 and compound4 == -1"),  # 3-component
            4: data.query("compound4 != -1"),  # 4-component
        }

    def post_process(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        observation_cols = [col for col in data.columns if col.startswith("event")]
        compound_cols = [col for col in data.columns if col.startswith("compound")]
        data = unique_reactions(data, compound_cols, observation_cols)

        fact_sets = self._partition_by_arity(data)
        return {
            f"unique_reactivities_{i}": fact_set[observation_cols].sum(axis=1, skipna=False).values
            for i, fact_set in fact_sets.items()
        }

    def predict(self, data: pd.DataFrame):
        fact_sets = self._partition_by_arity(data)

        reactants = [
            [fact_set[f"compound{j+1}"].values for j in range(n_components)]
            for n_components, fact_set in fact_sets.items()
        ]

        mem = self.mem()

        reactivities = jnp.concatenate(
            [
                sample(
                    f"reactivities{i}",
                    dist.Beta(self.react_beta_a, self.react_beta_b),
                    sample_shape=(n,),
                )
                for i, n in zip(fact_sets, self.N)
            ]
        )

        reactivities_with_zero = jnp.concatenate(
            [jnp.zeros((1,)), reactivities],
        )

        react_tensors = [
            deterministic(f"react_tensor{i+2}", reactivities_with_zero[idx])
            for i, idx in enumerate(self.reactivity_indices)
        ]

        reacts = [
            deterministic(
                "reacts2",
                jnp.sum(
                    mem[reactants[0][0]][:, :, np.newaxis]
                    * mem[reactants[0][1]][:, np.newaxis, :]
                    * react_tensors[0][np.newaxis, :, :],
                    axis=(1, 2),
                ),
            ),
            deterministic(
                "reacts3",
                jnp.sum(
                    mem[reactants[1][0]][:, np.newaxis, np.newaxis, :]
                    * mem[reactants[1][1]][:, np.newaxis, :, np.newaxis]
                    * mem[reactants[1][2]][:, :, np.newaxis, np.newaxis]
                    * react_tensors[1][np.newaxis, :, :, :],
                    axis=(1, 2, 3),
                ),
            ),
            deterministic(
                "reacts4",
                jnp.sum(
                    mem[reactants[2][0]][:, np.newaxis, np.newaxis, np.newaxis, :]
                    * mem[reactants[2][1]][:, np.newaxis, np.newaxis, :, np.newaxis]
                    * mem[reactants[2][2]][:, np.newaxis, :, np.newaxis, np.newaxis]
                    * mem[reactants[2][3]][:, :, np.newaxis, np.newaxis, np.newaxis]
                    * react_tensors[2][np.newaxis, :, :, :, :],
                    axis=(1, 2, 3, 4),
                ),
            ),
        ]

        return {
            f"unique_reactivities_{i+2}": dist.Normal(
                loc=reacts[i], scale=self.likelihood_sd
            )
            for i in range(len(reacts))
        }


class NonstructuralModel(Model):
    metadata = {
        "ncompounds": "Number of compounds",
        **Model.metadata,
    }

    def __init__(self, metadata):
        super().__init__(metadata)

    def mem(self):
        return sample(
            "mem",
            dist.Beta(self.mem_beta_a, self.mem_beta_b),
            sample_shape=(self.ncompounds, self.N_props),
        )


class StructuralModel(Model):
    metadata = {
        "fingerprint_matrix": """(n_compounds × fingerprint_length matrix) of fingerprint bits.""",
        **Model.metadata
    }

    def __init__(self, metadata):
        """Bayesian reactivity model informed by structural fingerprints.
        """
        super().__init__(metadata)
        self.ncompounds, self.fingerprint_length = self.fingerprint_matrix.shape

    def mem(self):
        fp_mem = sample(
            "fp_mem",
            dist.Beta(self.mem_beta_a, self.mem_beta_b),
            sample_shape=(self.fingerprint_length, self.N_props),
        )

        return deterministic(
            "mem", 1 - jnp.prod(1 - self.fingerprint_matrix[..., jnp.newaxis] * fp_mem, axis=1)
        )
