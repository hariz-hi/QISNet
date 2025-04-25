from datetime import datetime
import os
import pandas as pd
from contextlib import contextmanager
import time


def date_fname():
    uniq_filename = (
            str(datetime.now().date()) + "_" + str(datetime.now().time()).replace(":", ".")
    )
    return uniq_filename


def safe_mkdirs(path: str, log) -> None:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            log.warning(e)
            raise IOError(f"Failed to create recursive directories: {path}")


def metrics_csv(data, output_dir, filename):
    df = pd.DataFrame(data)
    df.loc["mean"] = df.mean()
    df.to_csv(f"{output_dir}/{filename}.csv", index=False, header=False)


@contextmanager
def timer(name):
    t0 = time.perf_counter()
    yield
    print(f'[{name}] done in {time.perf_counter() - t0:0.4f} s')
