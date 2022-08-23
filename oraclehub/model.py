from __future__ import annotations

import pandas as pd
from numpyro import distributions as dist


class Model:
    def __init__(self):
        pass

    def predict(self, experiments) -> dict[str, dist.Distribution]:
        pass

    def post_process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
