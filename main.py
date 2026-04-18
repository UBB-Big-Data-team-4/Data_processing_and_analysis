import sys
from operator import add
from random import random

from pyspark.sql import SparkSession

import DataProcessing

if __name__ == "__main__":
    df = DataProcessing.getProcessedData()

    df.show(2000)