import pandas as pd
import numpy as np
from tqdm import tqdm

# 降低内存占用
def reduce_mem_usage(df):
    """ 
    降低内存占用
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('当前内存占用: {:.2f} MB'.format(start_mem))
    
    for col in tqdm(df.columns):
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = np.array(df[col]).astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = np.array(df[col]).astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = np.array(df[col]).astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = np.array(df[col]).astype(np.int64)

            elif str(col_type)[:5] == 'float':
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = np.array(df[col]).astype(np.float32)
                else:
                    df[col] = np.array(df[col]).astype(np.float64)
        else:
            # df[col] = df[col].astype('category')
            pass
    end_mem = df.memory_usage().sum() / 1024**2
    print('最终内存占用: {:.2f} MB'.format(end_mem))
    print('下降了 {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df