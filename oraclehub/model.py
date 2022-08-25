from __future__ import annotations
from curses import meta
from email.policy import default
from importlib.metadata import metadata

import pandas as pd
from numpyro import distributions as dist


class Model:
    metadata = {}
    def __init__(self, metadata: dict):
        for metaname, val in self.metadata.items():
            if isinstance(val, str):
                # no default
                metavalue = metadata[metaname]
            else:
                metavalue = metadata.get(metaname, val[0])
            setattr(self, metaname, metavalue)

    def predict(self, experiments) -> dict[str, dist.Distribution]:
        pass

    def post_process(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        pass
