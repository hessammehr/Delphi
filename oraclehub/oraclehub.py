from __future__ import annotations

import inspect
import pickle
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax.random import PRNGKey
from numpyro import distributions as dist
from numpyro import sample
from numpyro.handlers import scope
from numpyro.infer import MCMC, NUTS

from oraclehub.model import Model


class OracleHub:
    def __init__(self, db_file=None):
        self.db = {}
        self._meta = None
        if db_file:
            self.db_file = Path(db_file)
        else:
            self.db_file = Path(__file__).with_name("oracle.db")

        if self.db_file.exists():
            self.load()

    @contextmanager
    def with_meta(self, meta: dict):
        try:
            self._meta = meta
            yield self
        finally:
            self._meta = None

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
        model_id = model_name + '-' + model_hash

        self.db[model_id] = {
            "model": model,
            "source": source,
            "name": model.__name__,
            "timestamp": datetime.now()
        }

        self.write()
        return model_id

    def load(self):
        with self.db_file.open("rb") as f:
            db = pickle.load(f)
        for model_id in db:
            obj = db[model_id]
            namespace = {}
            exec(obj["source"], namespace)
            obj["model"] = namespace[obj["name"]]
        self.db = db

    def write(self):
        # remove model object as it's tricky to serialize
        db = {
            model_id: {k: v for k, v in model_info.items() if k != "model"}
            for model_id, model_info in self.db.items()
        }
        with self.db_file.open("wb") as f:
            pickle.dump(db, f)

    def dump(self, path: Path = None, model_id=None):
        path = path or self.db_file.parent
        if model_id:
            models = {model_id: self.db[model_id]}
        else:
            models = self.db
        for model_id in models:
            py_file = (path / model_id).with_suffix(".py")
            with open(py_file, "w") as f:
                f.write(models[model_id]["source"])

    def run(self, predictor, data, observables=None, sample_params=None):
        sample_params = sample_params or {}

        def model_fn(data, observables=None):
            for var, dist in predictor(data).items():
                obs = observables and observables.get(var, None)
                sample(var, dist, obs=obs)

        return self.sample(model_fn, data, observables, **sample_params)

    def sample(
        self,
        model_fn,
        *args,
        nuts_params=None,
        mcmc_params=None,
    ):
        nuts_params = nuts_params or {}
        mcmc_params = mcmc_params or {}

        nuts = NUTS(model_fn, **nuts_params)
        mcmc = MCMC(nuts, num_samples=1000, num_warmup=1000, **mcmc_params)
        mcmc.run(PRNGKey(0), *args)
        return mcmc

    def compare(
        self,
        model_ids: list[str],
        data,
        alpha=1.0,
        sample_params=None,
    ):
        mix = self.mix(model_ids, alpha=alpha)
        models = {model_id: self[model_id] for model_id in model_ids}
        observables = [m.post_process(data) for m in models.values()]
        
        # TODO: make sure observables are the same for all models
        return self.run(mix, data, observables=observables[0], sample_params=sample_params)

    def mix(
        self,
        model_ids: list[str],
        alpha=1.0,
    ):
        N_models = len(model_ids)
        models = {model_id: self[model_id] for model_id in model_ids}

        def predictor_fn(data):
            weight_dist = dist.Dirichlet(jnp.ones(N_models) * alpha)
            mixing_dist = dist.Categorical(sample("probs", weight_dist))

            predictors = []
            for model_id, model in models.items():
                with scope(prefix=model_id, divider="."):
                    predictors.append(model.predict(data))

            observed_var_names = list(predictors[0])

            var_predictors = {
                var_name: [pred[var_name] for pred in predictors]
                for var_name in observed_var_names
            }

            return {
                var_name: dist.Mixture(mixing_dist, preds)
                for var_name, preds in var_predictors.items()
            }

        return predictor_fn

    def __getitem__(self, model_id) -> Model:
        if self._meta is None:
            raise Exception("Metadata not loaded")
        return self.db[model_id]["model"](self._meta)
