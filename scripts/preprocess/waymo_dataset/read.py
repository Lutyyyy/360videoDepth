# Reference: https://github.com/waymo-research/waymo-open-dataset

from typing import Optional
import sys
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2


# Path to the directory with all components
# dataset_dir = '/content'
# context_name = '10023947602400723454_1120_000_1140_000'

def read(dataset_dir: str, tag: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
   paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
   return dd.read_parquet(paths)

