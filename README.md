# Delphi
A version control system and standard runtime for Bayesian models in science.

## Installation
Optionally, modify `pyproject.toml` to specify whether to use the GPU (default) or CPU runtime for jax.

```toml
[tool.poetry.dependencies]
jaxlib = {source = "jax-gpu"} # or "jax-cpu"
```

Then, install the package with `poetry install`.

## Usage
### Depositing, retreiving, and editing models
The `Delphi` class acts as a repository for Bayesian models, which can be deposited into its store using the `Delphi.submit(ModelClass)` method. Delphi automatically finds the file containing the model class and stores it under a unique entry within its `models` directory. This unique hash (`model_id`) is returned and can be used to retrieve the model class from the store later using the `Delphi[model_id]` syntax.

For any modifications to the model, `Delphi.edit(model_id)` copies a temporary copy of the model file into the `scratch` directory, where it can be modified. Once the modifications are complete, `Delphi.submit(ModelClass, delete_file=True)` will store the new version of the model in the `models` directory and delete the temporary copy in the `scratch` directory.

### Operations on models
* `Delphi.run(model.predict, data)` runs MCMC sampling against the model and returns the final `MCMC` object containing sampled parameters and diagnostics.
* `Delphi.mix([model_id1, model_id2, ...], alpha=1.0)` returns a new predictor (dictionary of variable name => numpyro.dist.Distribution) using `Dirichlet(alpha)` as the mixing prior.
* `Delphi.compare([model_id1, model_id2, ...], data, alpha=1.0)` uses `Delphi.mix` to compute the posterior mixing distribution based on the observations in `data`, which can be used to calculate the relative marginal likelihood of each model (see [Kamari et al.][kamary2022]).

## Model API
*This API is currently in a state of flux and may be subject to breaking change.*

To be used with Delphi, your model simply needs to inherit from `delphi.model.Model` and implement the following methods:

* `predict`: Given a set of inputs, return a set of predictions as a dictionary of variable name => numpyro.dist.Distribution.
* `observables`: Given a (partially incomplete) dataset, return a dictionary of variable name => observations, with `np.nan` for missing values.

[kamary2022]: http://arxiv.org/abs/1412.2044