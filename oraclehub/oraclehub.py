from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path
import pickle

import pandas as pd
from jax import numpy as jnp
from jax.random import PRNGKey
from numpyro import distributions as dist
from numpyro import sample
from numpyro.handlers import scope
from numpyro.infer import MCMC, NUTS

from oraclehub.model import Model

# %%%


class OracleHub:
    def __init__(self, db_file=None):
        self.db = {}
        if db_file:
            self.db_file = Path(db_file)
        else:
            self.db_file = Path(__file__).with_name("oracle.db")

        if self.db_file.exists():
            self.load()

    def submit(self, model: type[Model]):
        assert issubclass(model, Model)
        model_name = model.__name__
        source_file = inspect.getsourcefile(model)
        if source_file is None:
            source = inspect.getsource(model)
        else:
            with open(source_file, "r") as f:
                source = f.read()
        model_hash = sha256(source.encode())
        model_hash.update(model_name.encode())
        model_hash = model_hash.hexdigest()[:8]

        self.db[model_hash] = {
            "model": model,
            "source": source,
            "name": model.__name__,
        }

        self.write()
        return model_hash

    def load(self):
        with self.db_file.open("rb") as f:
            db = pickle.load(f)
        for model_hash in db:
            obj = db[model_hash]
            glbs, locs = {}, {}
            exec(obj["source"], glbs, locs)
            obj["model"] = locs[obj["name"]]
        self.db = db

    def write(self):
        # remove model object as it's tricky to serialize
        db = {
            model_hash: {k: v for k, v in model_info.items() if k != "model"}
            for model_hash, model_info in self.db.items()
        }
        with self.db_file.open("wb") as f:
            pickle.dump(db, f)

    def dump(self, path: Path = None, model_hash=None):
        path = path or self.db_file.parent
        if model_hash:
            models = {model_hash: self[model_hash]}
        else:
            models = self.db
        for model_hash in models:
            py_file = (path / model_hash).with_suffix(".py")
            with open(py_file, "w") as f:
                f.write(models[model_hash]["source"])

    def run(self, model_hash: str, data: pd.DataFrame):
        nuts = NUTS(self[model_hash])

    def compare(
        self,
        model_hashes: list[str],
        data,
        alpha=1.0,
        nuts_params=None,
        mcmc_params=None,
    ):
        N_models = len(model_hashes)
        models = {model_hash: self[model_hash] for model_hash in model_hashes}

        def numpyro_model(data):
            weight_dist = dist.Dirichlet(jnp.ones(N_models) * alpha)
            mixing_dist = dist.Categorical(sample("probs", weight_dist))

            predictors = []
            for model_hash, model in models.items():
                with scope(prefix=model_hash, divider="."):
                    predictors.append(model.predict(data))

            observed_var_names = list(predictors[0])

            var_predictors = {
                var_name: [pred[var_name] for pred in predictors]
                for var_name in observed_var_names
            }

            mixtures = {
                var_name: dist.Mixture(mixing_dist, preds)
                for var_name, preds in var_predictors.items()
            }

            for var_name, mix_pred in mixtures.items():
                sample(var_name, mix_pred, obs=data[var_name])

        nuts = NUTS(numpyro_model)
        mcmc = MCMC(nuts, num_samples=1000, num_warmup=1000, num_chains=1)
        mcmc.run(PRNGKey(0), data=data)
        return mcmc

    def __getitem__(self, model_hash) -> Model:
        return self.db[model_hash]


# %%
