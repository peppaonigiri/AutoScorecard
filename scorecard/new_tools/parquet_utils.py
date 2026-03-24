import os, gc, psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import polars as pl
from typing import Optional
import zipfile
import tempfile
import shutil
from contextlib import contextmanager

def concat_parquet_streaming(
    files, 
    output_path="merged_16_chunks.parquet",
    batch_size=100_000,
    mem_threshold_gb=6.0,  # 内存阈值，超过则立即清理
    columns_consistent=True  # 新增参数：是否所有文件的列都一致
):
    # 统一列顺序（若各文件列一致，可直接跳过这段以提速）
    if columns_consistent:
        # 列一致时，直接使用第一个文件的列顺序
        first_valid_file = None
        for f in files:
            if os.path.exists(f):
                first_valid_file = f
                break
        if not first_valid_file:
            raise ValueError("未找到有效的parquet文件")
        all_cols = pq.read_schema(first_valid_file).names
    else:
        # 列不一致时，需要收集所有列
        all_cols = []
        for f in files:
            if not os.path.exists(f):
                continue
            names = pq.read_schema(f).names
            for c in names:
                if c not in all_cols:
                    all_cols.append(c)
        if not all_cols:
            raise ValueError("未找到有效的parquet文件或schema为空")
    
    # 基于第一个文件的schema构建writer
    first_table = pq.read_table(files[0], use_threads=False)
    schema = first_table.schema
    del first_table; gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass

    writer = pq.ParquetWriter(output_path, schema=schema, use_dictionary=True)

    try:
        for idx, f in enumerate(files):
            print(f"正在处理文件 {idx+1}/{len(files)}: {f}")
            if not os.path.exists(f):
                continue
            pf = pq.ParquetFile(f)
            # 按批次流式处理
            for batch in pf.iter_batches(batch_size=batch_size, columns=all_cols, use_threads=False):
                # 可选：这里可以做列类型降级，但建议直接保持arrow类型，避免pandas中间态
                writer.write_batch(batch)

                # 内存监控 + 及时清理
                if psutil.Process(os.getpid()).memory_info().rss / 1024**3 > mem_threshold_gb:
                    gc.collect()
                    try:
                        pa.default_memory_pool().release_unused()
                    except Exception:
                        pass

            # 每个文件结束也做一次清理
            gc.collect()
            try:
                pa.default_memory_pool().release_unused()
            except Exception:
                pass
    finally:
        writer.close()
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

def fix_nan_parquet(ref_path, bad_path, fixed_path):
    """
    修正parquet文件中的NaN值
    参数:
        ref_path: 参考schema文件路径
        bad_path: 需要修正的文件路径
        fixed_path: 修正后的文件路径
    """

    # 1) 读取参考 schema（来自 chunk15）
    schema_ref = pq.read_schema(ref_path)

    # 2) 读取 chunk16 为 Arrow Table
    t16 = pq.read_table(bad_path)

    # 3) 对齐列：补缺失、丢多余、按顺序排列
    cols_ref = schema_ref.names
    cols_16 = set(t16.schema.names)

    # 3.1 补齐缺失列（用 null 填充，类型按参考列类型）
    missing = [c for c in cols_ref if c not in cols_16]
    for c in missing:
        # 用参考列类型创建全 null 列
        field = schema_ref.field(c)
        t16 = t16.append_column(c, pa.nulls(len(t16)).cast(field.type, safe=False))
    
    # # 3.2 丢弃 chunk16 中多出来的列（不在参考 schema 中）
    # keep_cols = [c for c in cols_ref]  # 只保留参考列
    # t16 = t16.select(keep_cols)

    # 4) 按参考 schema 强制转换类型（safe=False 可将全 NaN 的列从 null 转为目标类型）
    t16_fixed = t16.cast(schema_ref, safe=False)

    # 5) 保存修正后的 chunk16
    pq.write_table(t16_fixed, fixed_path)
    print("已生成：", fixed_path)

def split_parquet_by_field(
    src_path: str,
    out_path_template: str,
    split_field: str,
    split_labels,  # 可以是列表或字典
    *,
    filter_field: str = None,
    filter_value: str = None,
    batch_size: int = 100_000
):
    """
    根据指定字段值将Parquet文件分流到多个文件
    
    参数:
        src_path: 源Parquet文件路径
        out_path_template: 输出文件路径模板, 用{idx}作为索引占位符
        split_field: 用于分流的字段名 (如 "cust_type")
        split_labels: 分流标签, 可以是:
                      - 列表: ["客群1", "客群2"] → 自动索引1,2,3...
                      - 字典: {"客群1": 'cust1', "客群3": 'cust3'} → 自定义索引
        filter_field: 可选的基础过滤字段名 (如 "sample_type")
        filter_value: 可选的基础过滤字段值 (如 "建模样本")
        batch_size: 每批读取的行数
    """
    
    # 统一转换为 {标签: 索引} 映射格式
    if isinstance(split_labels, list):
        label_to_idx = {label: i+1 for i, label in enumerate(split_labels)}
    elif isinstance(split_labels, dict):
        label_to_idx = split_labels
    else:
        raise TypeError("split_labels 必须是 list 或 dict 类型")
    
    pf = pq.ParquetFile(src_path)
    src_schema = pf.schema_arrow
    
    # 准备 writer 字典
    writers = {}
    for label, idx in label_to_idx.items():
        writers[label] = pq.ParquetWriter(
            out_path_template.format(idx=idx),
            schema=src_schema,
            use_dictionary=True
        )
    
    # 需要检查的字段列表
    required_fields = {split_field}
    if filter_field:
        required_fields.add(filter_field)
    
    try:
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, use_threads=False)):
            print(f"\n处理批次，批次: {batch_idx+1}")
            print(f"处理批次，行数: {batch.num_rows}")
            
            tbl = pa.Table.from_batches([batch])
            
            # 字段存在性检查
            cols = set(tbl.schema.names)
            missing = required_fields - cols
            if missing:
                print(f"警告: 缺少字段 {missing}, 跳过该批次")
                del tbl, batch
                gc.collect()
                continue

            
            # 构建基础过滤条件 (如果提供了过滤参数)
            if filter_field and filter_value:
                print(f"处理过滤条件，字段: {filter_field}, 值: {filter_value}")
                filter_col = pc.cast(tbl[filter_field], pa.large_string(), safe=False)
                base_mask = pc.equal(filter_col, pa.scalar(filter_value, type=pa.large_string()))
            else:
                # 无过滤条件，全选
                print(f"无过滤条件，全选")
                filter_col = None
                base_mask = pa.array([True] * tbl.num_rows, type=pa.bool_())
            
            # 按标签分流
            split_col = pc.cast(tbl[split_field], pa.large_string(), safe=False)
            
            for label, idx in label_to_idx.items():
                print(f"处理标签: {label}, 索引: {idx}")
                # 组合条件: 基础过滤 AND 分类过滤
                label_value = label.split('|')
                for value in label_value:
                    label_mask = pc.equal(split_col, pa.scalar(value, type=pa.large_string()))
                    final_mask = pc.and_kleene(base_mask, label_mask)
                    part = tbl.filter(final_mask)
                    if part.num_rows > 0:
                        writers[label].write_table(part)
                    del part, label_mask, final_mask
                    gc.collect()
            
            del split_col, base_mask, filter_col
            del tbl, batch
            gc.collect()
            
            try:
                pa.default_memory_pool().release_unused()
            except Exception:
                pass
    
    finally:
        for w in writers.values():
            w.close()
        
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass


def _force_memory_cleanup():
    """强制进行内存清理"""
    # 多次调用gc.collect()确保彻底清理
    for _ in range(3):
        gc.collect()
    
    # 打印当前内存使用情况
    memory_info = psutil.virtual_memory()
    print(f"    内存清理后 - 可用内存: {memory_info.available / 1024**2:.0f} MB ({memory_info.percent:.1f}% 已使用)")


@contextmanager
def get_source_context(file_path: str):
    """
    上下文管理器：如果是ZIP文件，解压到临时目录并yield解压后的路径；
    如果是普通文件，直接yield原路径。
    """
    if file_path.lower().endswith('.zip'):
        temp_dir = tempfile.mkdtemp()
        print(f"  - 检测到ZIP文件，准备解压至临时目录: {temp_dir}")
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # 寻找ZIP中的文本文件 (.txt 或 .csv)
                target_file = None
                # 优先找同名文件，其次找第一个txt/csv
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                candidates = [n for n in zip_ref.namelist() if n.lower().endswith(('.txt', '.csv')) and not n.startswith('__MACOSX')]
                
                # 尝试匹配文件名包含zip文件名的文件
                for cand in candidates:
                    if base_name in cand:
                        target_file = cand
                        break
                
                # 如果没有匹配到，则取第一个候选文件
                if not target_file and candidates:
                    target_file = candidates[0]
                
                if not target_file:
                    raise ValueError(f"在ZIP文件 '{file_path}' 中未找到 .txt 或 .csv 文件")
                
                print(f"  - 正在解压: {target_file}")
                zip_ref.extract(target_file, temp_dir)
                extracted_path = os.path.join(temp_dir, target_file)
                yield extracted_path
                
        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
                print("  - 临时解压目录已清理")
            except Exception as e:
                print(f"  - 警告: 清理临时目录失败: {e}")
    else:
        yield file_path



def build_mixed_schema(csv_path: str, encoding: str, separator: str, feature_start_index: int) -> Optional[dict]:
    """
    根据“前5列为字符串，其余为数值”的规则，构建一个混合类型的Schema。

    Args:
        csv_path: CSV文件路径
        encoding: 文件编码
        separator: 文件分隔符

    Returns:
        一个为Polars准备的、包含混合类型的schema字典，或在失败时返回None。
    """
    try:
        # 使用pandas只读取表头，内存占用极小
        header = pd.read_csv(
            csv_path, 
            encoding=encoding, 
            sep=separator,
            nrows=0
        ).columns.tolist()
        
        schema = {}
        for i, col_name in enumerate(header):
            if i < feature_start_index:
                # 前feature_start_index列指定为字符串
                schema[col_name] = pl.Utf8
            else:
                # 其余列指定为Float32，以兼容整数、浮点数和缺失值
                schema[col_name] = pl.Float32
        
        print(f"✅ 成功构建混合类型Schema，共 {len(header)} 列。")
        return schema
    except Exception as e:
        print(f"❌ 读取文件表头以构建Schema时出错: {e}")
        return None


def prepare_parquet_from_csv(
    csv_path: str, 
    parquet_path: str, 
    force_regenerate: bool = False,
    encoding: str = 'utf8',
    separator: str = ',',
    enable_schema: bool = True,
    feature_start_index: int = 5,
    row_group_size = 10000
):
    """
    步骤1: 从CSV文件高效读取数据并以流式方式转换为Parquet格式。
    支持直接读取 .zip 文件（会先自动解压到临时文件夹）。
    
    采用最稳健的“全字符串”模式：
    1. 快速读取列名，并预设所有列都为字符串类型。
    2. Polars在读取时不做任何类型推断，直接按字符串读取，避免解析错误和前期内存开销。
    3. 流式地将数据从CSV写入Parquet，内存占用低且稳定。

    Args:
        csv_path: 输入CSV文件路径 (支持 .txt, .csv, .zip)
        parquet_path: 输出Parquet文件路径
        force_regenerate: 是否强制重新生成
        encoding: CSV文件的编码，默认为'utf8'。
    """
    if os.path.exists(parquet_path) and not force_regenerate:
        print(f"Parquet文件 '{parquet_path}' 已存在，跳过生成步骤。")
        return

    print(f"开始转换任务 '{os.path.basename(csv_path)}' -> '{os.path.basename(parquet_path)}'")
    
    try:
        with get_source_context(csv_path) as working_csv_path:
            # 使用解压后的路径（或原路径）进行处理，逻辑保持不变
            
            # --- 根据编码选择不同的执行路径 ---
            if encoding.lower() in ['utf8', 'utf-8']:
                print("  - 使用高效的 UTF-8 流式路径 (scan_csv)...")
                # --- 阶段1: 构建混合类型Schema ---
                print("  - 阶段1: 准备混合类型Schema...")
                if enable_schema:
                    schema = build_mixed_schema(working_csv_path, encoding, separator, feature_start_index)
                else:
                    schema = None

                # --- 阶段2: 准备 LazyFrame ---
                print("  - 阶段2: 准备LazyFrame (混合类型模式)...")
                lazy_df = pl.scan_csv(
                    working_csv_path,
                    try_parse_dates=True,
                    separator=separator,
                    encoding=encoding,
                    infer_schema_length=None,
                    dtypes=schema if enable_schema else None,  
                    null_values=["", "NULL", "NaN", "NA", "nan","-","null"]
                )

                # --- 阶段3: 流式写入 ---
                print("  - 阶段3: 开始流式写入Parquet文件 (CPU和I/O密集)...")
                lazy_df.sink_parquet(
                    parquet_path,
                    compression="zstd",
                    statistics=True,
                    row_group_size=row_group_size
                )
                # 清理LazyFrame引用
                del lazy_df
            
            else:
                # --- 为其他编码提供回退路径 (更耗内存) ---
                print(f"  - 警告: 编码 '{encoding}' 不支持流式处理，将使用内存密集型路径 (read_csv)...")
                
                # 直接使用 read_csv 读取
                eager_df = pl.read_csv(
                    working_csv_path,
                    try_parse_dates=True,
                    separator=separator,
                    encoding=encoding, # read_csv 支持 'gbk'
                    infer_schema_length=None,
                    null_values=["", "NULL", "NaN", "NA", "nan","-","null"]
                )
                
                print("  - 文件读取完成，开始写入Parquet...")
                eager_df.write_parquet(
                    parquet_path,
                    compression="zstd",
                    statistics=True
                )
                # 清理DataFrame引用
                del eager_df

        print(f"✅ 转换成功完成，文件已保存至 '{parquet_path}'")
        
        _force_memory_cleanup()

    except Exception as e:
        print(f"❌ 处理CSV文件时出错: {e}")
        # 确保在失败时删除可能不完整的输出文件
        if os.path.exists(parquet_path):
            try:
                os.remove(parquet_path)
                print(f"  - 已删除不完整的输出文件: {parquet_path}")
            except OSError as oe:
                print(f"  - 删除不完整的输出文件时失败: {oe}")
        raise



# 常量：统一的空值标识列表
_NULL_VALUES = ["", "NULL", "NaN", "NA", "nan", "-", "null"]


def prepare_parquet_from_csv_new(
    csv_path: str, 
    parquet_path: str, 
    force_regenerate: bool = False,
    encoding: str = 'utf8',
    separator: str = ',',
    enable_schema: bool = True,
    feature_start_index: int = 5,
    row_group_size: int = 10000,
    batch_size: int = 50000,
    mem_threshold_gb: float = 6.0,
    gc_interval: int = 3
):
    """
    从CSV文件高效读取数据并以流式方式转换为Parquet格式。
    支持直接读取 .zip 文件（会先自动解压到临时文件夹）。
    
    采用极致内存优化模式：
    1. 显式控制 batch_size，限制每批次读取的行数。
    2. 定期释放内存，避免累积。
    3. 内存监控阈值，超过时强制 GC。

    Args:
        csv_path: 输入CSV文件路径 (支持 .txt, .csv, .zip)
        parquet_path: 输出Parquet文件路径
        force_regenerate: 是否强制重新生成
        encoding: CSV文件的编码，默认为'utf8'
        separator: CSV分隔符，默认为','
        enable_schema: 是否启用混合类型Schema
        feature_start_index: 特征列起始索引（之前为字符串列）
        row_group_size: Parquet row group 大小
        batch_size: 每批次读取的行数（控制内存峰值的关键参数）
        mem_threshold_gb: 内存阈值（GB），超过时强制 GC
        gc_interval: 每隔多少个 batch 执行一次 GC（默认3）
    """
    if os.path.exists(parquet_path) and not force_regenerate:
        print(f"Parquet文件 '{parquet_path}' 已存在，跳过生成步骤。")
        return

    print(f"开始转换任务 '{os.path.basename(csv_path)}' -> '{os.path.basename(parquet_path)}'")
    print(f"  - 参数: batch_size={batch_size}, row_group_size={row_group_size}, mem_threshold={mem_threshold_gb}GB")
    
    def _gc_cleanup(force: bool = False):
        """
        内存清理函数
        Args:
            force: 是否强制多轮清理
        """
        rounds = 3 if force else 1
        for _ in range(rounds):
            gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except:
            pass
    
    def _check_memory_and_gc():
        """检查内存使用情况，超过阈值时强制 GC"""
        mem_used_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        if mem_used_gb > mem_threshold_gb:
            print(f"    ⚠️ 内存超阈值 ({mem_used_gb:.2f}GB > {mem_threshold_gb}GB)，强制清理...")
            _gc_cleanup(force=True)
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            print(f"    ✓ 清理后内存: {mem_after:.2f}GB")
        return mem_used_gb
    
    try:
        with get_source_context(csv_path) as working_csv_path:
            # --- 根据编码选择不同的执行路径 ---
            if encoding.lower() in ['utf8', 'utf-8']:
                print("  - 使用高效的 UTF-8 流式路径...")
                
                # --- 阶段1: 构建混合类型Schema ---
                print("  - 阶段1: 准备混合类型Schema...")
                schema = build_mixed_schema(working_csv_path, encoding, separator, feature_start_index) if enable_schema else None

                # --- 阶段2: 分块读取并流式写入 ---
                print("  - 阶段2: 开始流式读取并分块写入Parquet文件...")
                
                reader = pl.read_csv_batched(
                    working_csv_path,
                    has_header=True,
                    separator=separator,
                    encoding='utf8',
                    dtypes=schema,
                    infer_schema_length=None,
                    null_values=_NULL_VALUES,
                    batch_size=batch_size
                )

                writer = None
                batch_idx = 0
                total_rows = 0
                
                try:
                    while True:
                        batches = reader.next_batches(1)
                        if not batches:
                            break
                        
                        for df_batch in batches:
                            batch_idx += 1
                            total_rows += len(df_batch)
                            
                            # 转换为 Arrow Table 并立即释放 Polars DataFrame
                            arrow_batch = df_batch.to_arrow()
                            del df_batch
                            
                            if writer is None:
                                print(f"    - 初始化 ParquetWriter, row_group_size: {row_group_size}")
                                writer = pq.ParquetWriter(
                                    parquet_path, 
                                    schema=arrow_batch.schema, 
                                    compression="zstd",
                                    write_statistics=True,
                                    use_dictionary=True
                                )
                            
                            writer.write_table(arrow_batch, row_group_size=row_group_size)
                            del arrow_batch
                            
                            # 每隔 gc_interval 个 batch 执行一次 GC 和内存检查
                            if batch_idx % gc_interval == 0:
                                _gc_cleanup()
                            
                            # 定期输出进度
                            if batch_idx % 5 == 0:
                                mem_gb = _check_memory_and_gc()
                                print(f"    - 已处理 {batch_idx} 批次, 累计 {total_rows:,} 行, 内存: {mem_gb:.2f}GB")

                finally:
                    if writer:
                        writer.close()
                    del reader
                    _gc_cleanup()
                
                print(f"    ✓ 共处理 {batch_idx} 批次, {total_rows:,} 行")
            
            else:
                # --- 为其他编码提供回退路径 (更耗内存) ---
                print(f"  - 警告: 编码 '{encoding}' 不支持流式处理，将使用内存密集型路径...")
                
                eager_df = pl.read_csv(
                    working_csv_path,
                    try_parse_dates=True,
                    separator=separator,
                    encoding=encoding,
                    infer_schema_length=None,
                    null_values=_NULL_VALUES
                )
                
                print("  - 文件读取完成，开始写入Parquet...")
                eager_df.write_parquet(
                    parquet_path,
                    compression="zstd",
                    statistics=True
                )
                del eager_df

        print(f"✅ 转换成功完成，文件已保存至 '{parquet_path}'")
        _force_memory_cleanup()

    except Exception as e:
        print(f"❌ 处理CSV文件时出错: {e}")
        if os.path.exists(parquet_path):
            try:
                os.remove(parquet_path)
                print(f"  - 已删除不完整的输出文件: {parquet_path}")
            except OSError as oe:
                print(f"  - 删除不完整的输出文件时失败: {oe}")
        raise


if __name__ == "__main__":
    # # 合并的示例代码
    # files = [f"D:\work/06.std_product\wxjk/theta\data/chunk_{i}.parquet" for i in range(1,9)]
    # # print(files)
    # concat_parquet_streaming(
    #     files,
    #     output_path="D:\work/06.std_product\wxjk\data_or/theta_data_all1.parquet",
    #     batch_size=100_000,     # 可按内存调小
    #     mem_threshold_gb=14.0,  # 建议设为物理内存的30%~50%
    #     columns_consistent=True  # 如果所有文件的列都一致，设为True以提速
    # )

    # # 修复控制的实例代码
    # ref_path = "chunk15.parquet"
    # bad_path = "chunk16.parquet"
    # fixed_path = "chunk16_fixed.parquet"
    # fix_nan_parquet(ref_path, bad_path, fixed_path)


    # # 分类的示例使用方法
    # split_label_dict = {'新贷':'xd','新贷基测':'xdjc','复贷基测':'fdjc','复贷+if_app=1':'fdapp1','复贷+if_app=0':'fdapp0','复贷+if_app=1|复贷+if_app=0':'fd','新贷|复贷+if_app=1|复贷+if_app=0':'xdfd'}
    # split_parquet_by_field(
    #     src_path=r"D:\work\06.std_product\wxjk\data_or\zeta_data_all1.parquet",
    #     out_path_template=r"D:\work\06.std_product\wxjk\parquet_test\zeta_data_all1_{idx}.parquet",
    #     split_field="devflag_new",
    #     split_labels=split_label_dict,
    #     filter_field="train_or_test",
    #     filter_value="train",
    #     batch_size=80_000
    # )

    # 原始文件转parquet测试
    csv_path = r"D:\work\04.test\03-众安\data_or\first\众安保险20250908_LH_deltaV1_result.txt"
    zip_path = r"D:/work/06.std_product/wxjk/data_or/维信20251106_d1_t2_z2_deltaV1_result.zip"
    parquet_txt_path= r"D:\work\04.test\03-众安\data_or\first\众安保险20250908_LH_deltaV1_result.parquet"
    parquet_zip_path= r"D:/work/06.std_product/wxjk/data_or/维信20251106_d1_t2_z2_deltaV1_result_zip_new.parquet"
    force_regenerate = False
    encoding = 'utf8'
    separator = ','
    enable_schema = True
    feature_start_index = 5
    row_group_size = 1000
    batch_size = 30000         
    mem_threshold_gb = 4.0      

    # 测试 TXT
    print("\n--- 测试 1: TXT 文件 ---")
    prepare_parquet_from_csv_new(
        csv_path, parquet_txt_path, force_regenerate, encoding, separator, 
        enable_schema, feature_start_index, row_group_size,
        batch_size=batch_size, 
        mem_threshold_gb=mem_threshold_gb
    )
    
    # 测试 ZIP
    print("\n--- 测试 2: ZIP 文件 ---")
    # prepare_parquet_from_csv_new(
    #     zip_path, parquet_zip_path, force_regenerate, encoding, separator, 
    #     enable_schema, feature_start_index, row_group_size,
    #     batch_size=batch_size, 
    #     mem_threshold_gb=mem_threshold_gb
    # )