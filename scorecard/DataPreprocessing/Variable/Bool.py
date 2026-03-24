import pandas as pd
import numpy as np


# 1、将bool类型转化为数值类型
def bool2num(df, ft):
    # df[ft] = df[ft].astype(int)
    df[ft] = pd.to_numeric(df[ft])

# 2、