import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import gc

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
        col_type = str(df[col].dtype)
        if col_type not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            continue
        
        else:
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].values.astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].values.astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].values.astype(np.int32)
                else:
                    pass

            elif col_type[:5] == 'float':
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].values.astype(np.float32)
                else:
                    pass

    end_mem = df.memory_usage().sum() / 1024**2
    print('最终内存占用: {:.2f} MB'.format(end_mem))
    print('下降了 {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def reduce_mem_usage_list(df):
    """
    降低内存占用
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('当前内存占用: {:.2f} MB'.format(start_mem))
    
    transformer_dic = {num_type:[] for num_type in ['int8', 'int16', 'int32',
                                                    'float16', 'float32']}
    for col in df.columns:
        col_type = str(df[col].dtype)
        if col_type not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            continue
        
        else:
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    transformer_dic['int8'].append(col)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    transformer_dic['int16'].append(col)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    transformer_dic['int32'].append(col)
                else:
                    pass
            elif col_type[:5] == 'float':
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    transformer_dic['float32'].append(col)
                else:
                    pass
    print(f"""需转换：
                {len(transformer_dic["int8"])}列int8数据
                {len(transformer_dic["int16"])}列int16数据
                {len(transformer_dic["int32"])}列int32数据
                {len(transformer_dic["float16"])}列float16数据
                {len(transformer_dic["float32"])}列float32数据""")

    for num_type, cols in tqdm(transformer_dic.items(),desc='数据类型转换'):
        if len(cols) > 0:
            df[cols] = df[cols].astype(num_type)

    end_mem = df.memory_usage().sum() / 1024**2
    print('最终内存占用: {:.2f} MB'.format(end_mem))
    print('下降了 {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def reduce_mem_usage_dic(df):
    """
    降低内存占用
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('当前内存占用: {:.2f} MB'.format(start_mem))

    df_dic = {}
    for col in tqdm(df.columns):
        col_type = str(df[col].dtype)
        if col_type not in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            df_dic[col] = df[col].values
        
        else:
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_dic[col] = df[col].values.astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_dic[col] = df[col].values.astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_dic[col] = df[col].values.astype(np.int32)
                else:
                    df_dic[col] = df[col].values.astype(np.int64)

            elif col_type[:5] == 'float':
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df_dic[col] = df[col].values.astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_dic[col] = df[col].values.astype(np.float32)
                else:
                    df_dic[col] = df[col].values.astype(np.float64)

    df = pd.DataFrame(df_dic)
    end_mem = df.memory_usage().sum() / 1024**2
    print('最终内存占用: {:.2f} MB'.format(end_mem))
    print('下降了 {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df




def reduce_mem_usage_para(df, chunk_size=1000, n_jobs=-1, mode='columns',use_dic=True):
    """
    :param df: pandas DataFrame
    :param chunk_size: 分块大小
    :param n_jobs: 并行处理数量
    :param mode: 分块模式，可选值为 'columns' 或 'rows' ，按列分块或按行分块
    :return: pandas DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'当前内存占用: {start_mem:.2f} MB')

    if mode == 'columns':
        # 按列分块处理
        chunks = [df.iloc[:, i:i + chunk_size] for i in range(0, df.shape[1], chunk_size)]
    elif mode == 'rows':
        # 按行分块处理
        chunks = [df.iloc[i:i + chunk_size, :] for i in range(0, df.shape[0], chunk_size)]
    else:
        raise ValueError("mode 参数必须是 'columns' 或 'rows'")

    # 并行处理
    if use_dic:
        reduced_chunks = Parallel(n_jobs=n_jobs)(delayed(reduce_mem_usage_dic)(chunk) for chunk in chunks)
    else:
        reduced_chunks = Parallel(n_jobs=n_jobs)(delayed(reduce_mem_usage)(chunk) for chunk in chunks)

    # 合并分块
    df_reduced = pd.concat(reduced_chunks, axis=1 if mode == 'columns' else 0)

    end_mem = df_reduced.memory_usage().sum() / 1024**2
    print('最终内存占用: {:.2f} MB'.format(end_mem))
    print('下降了 {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df_reduced





def reduce_mem_usage_mp(df, chunk_size=1000, n_jobs=-1, mode='columns',use_dic=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print('初始内存占用: {:.2f} MB'.format(start_mem))

    if mode == 'columns':
        # 按列分块处理
        chunks = [df.iloc[:, i:i + chunk_size] for i in range(0, df.shape[1], chunk_size)]
    elif mode == 'rows':
        # 按行分块处理
        chunks = [df.iloc[i:i + chunk_size, :] for i in range(0, df.shape[0], chunk_size)]
    else:
        raise ValueError("mode 参数必须是 'columns' 或 'rows'")
    # 设置默认的进程数
    if n_jobs < 1:
        n_jobs = mp.cpu_count()  # 使用系统的 CPU 核心数
    if use_dic:
    # 使用 multiprocessing 并行处理，imp接受一个参数，starmap接受多个参数形成的元组
        with mp.Pool(processes=n_jobs) as pool:
            reduced_chunks = list(tqdm(pool.map(reduce_mem_usage_dic, chunks), total=len(chunks), desc="子任务处理进度"))
    else:
        with mp.Pool(processes=n_jobs) as pool:
            reduced_chunks = list(tqdm(pool.map(reduce_mem_usage, chunks), total=len(chunks), desc="子任务处理进度"))
    # 合并分块
    df_reduced = pd.concat(reduced_chunks, axis=1 if mode == 'columns' else 0)

    end_mem = df_reduced.memory_usage().sum() / 1024**2
    print('最终内存占用: {:.2f} MB'.format(end_mem))
    print('下降了 {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df_reduced







