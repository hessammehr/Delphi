from __future__ import annotations
from pydantic import BaseModel, Extra

import pandas as pd
from numpyro import distributions as dist


class Model(BaseModel, extra=Extra.allow):
    def predict(self, experiments) -> dict[str, dist.Distribution]:
        pass

    def observables(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        pass
