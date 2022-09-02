from __future__ import annotations
from importlib import reload

import inspect
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from jax import numpy as jnp
from jax.random import PRNGKey
from numpyro import distributions as dist
from numpyro import sample
from numpyro.handlers import scope
from numpyro.infer import MCMC, NUTS
from oraclehub import models

from oraclehub.model import Model


class OracleHub:
    def __init__(self):
        self.db = {}
        self._meta = None
        self.db_dir = Path(__file__).with_name("models")
        self._build_init()

    def _build_init(self):
        init_file = self.db_dir / "__init__.py"
        import_lines = ['db = {}\n']
        db_lines = []
        for path in self.db_dir.iterdir():
            if path.is_dir() and not path.name.startswith('_'):
                import_lines.append(f"from . import {path.name}\n")
                db_lines.append(f"db['{path.name}'] = {path.name}.model\n")

        with init_file.open("w") as f:
            f.writelines(import_lines)
            f.writelines(db_lines)
        
        reload(models)
        self.db = models.db

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

        model_id = model_name + "_" + model_hash

        if hasattr(models, model_id):
            return model_id

        model_dir = self.db_dir / f"{model_id}"
        model_dir.mkdir()

        with (model_dir / "code.py").open("w") as f:
            f.write(source)
        with (model_dir / "__init__.py").open("w") as f:
            f.writelines([
                f'from .code import {model_name} as model\n',
                f"timestamp = '{datetime.now()}'\n",
            ])

        self._build_init()
        return model_id
    
    def edit(self, model_id):
        scratch_dir = self.db_dir.with_name('scratch')
        _, scratch_file = tempfile.mkstemp(suffix='.py', prefix=model_id + '_', dir=scratch_dir)
        shutil.copy(self.db_dir / model_id / 'code.py', scratch_file)
        return scratch_file

    def run(self, predictor, data, observables=None, sample_params=None):
        sample_params = sample_params or {}

        def model_fn(data, observables=None):
            for var, dist in predictor(data).items():
                obs = observables and observables.get(var, None)
                if obs is not None:
                    # impute missing values
                    is_nan = np.isnan(obs)
                    nan_idx = np.nonzero(is_nan)[0]
                    impute_samples = sample(var, dist)
                    if nan_idx.size:
                        obs = jnp.asarray(obs).at[nan_idx].set(impute_samples[nan_idx])
                    sample(f'{var}_obs', dist.mask(~is_nan), obs=obs)
                else:
                    sample(var, dist)


        return self.sample(model_fn, data, observables, **sample_params)

    def sample(
        self,
        model_fn,
        *args,
        nuts_params=None,
        mcmc_params=None,
    ):
        nuts_params = nuts_params or {}
        mcmc_params = {
            'num_samples': 1000,
            'num_warmup': 1000,
            **(mcmc_params or {})
        }

        nuts = NUTS(model_fn, **nuts_params)
        mcmc = MCMC(nuts, **mcmc_params)
        
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
        return self.db[model_id](self._meta)
