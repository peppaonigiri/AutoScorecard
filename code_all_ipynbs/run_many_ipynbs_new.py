import os
import sys
import logging
import time
from datetime import datetime

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter
from nbformat import read, write

import re
import warnings
warnings.filterwarnings("ignore")


# 自定义执行处理器，输出 cell 级别的进度日志
class LoggingExecutePreprocessor(ExecutePreprocessor):
    def __init__(self, *args, run_index=None, run_total=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_index = run_index
        self._run_total = run_total
        self._total_code_cells = 0
        self._current_code_cell_index = 0

    def preprocess(self, nb, resources, km=None):
        # 仅统计 code 类型的单元格
        self._total_code_cells = sum(1 for c in nb.cells if getattr(c, 'cell_type', None) == 'code')
        self._current_code_cell_index = 0
        if self._total_code_cells == 0:
            logging.info(f"[{self._run_index}/{self._run_total}] 本notebook无可执行的code单元格")
        return super().preprocess(nb, resources, km=km)

    def preprocess_cell(self, cell, resources, cell_index):
        if getattr(cell, 'cell_type', None) == 'code':
            # 预先计算显示索引，但在失败时仍然显示同一编号
            next_index = self._current_code_cell_index + 1
            progress_prefix = f"[{self._run_index}/{self._run_total}]"
            logging.info(f"{progress_prefix} 正在执行Cell [{next_index}/{self._total_code_cells}] …")
            cell_start_time = time.time()
            try:
                result = super().preprocess_cell(cell, resources, cell_index)
                elapsed = time.time() - cell_start_time
                self._current_code_cell_index = next_index
                logging.info(f"{progress_prefix} 完成Cell [{self._current_code_cell_index}/{self._total_code_cells}] (耗时: {elapsed:.2f}秒)")
                return result
            except Exception as e:
                elapsed = time.time() - cell_start_time
                logging.error(f"{progress_prefix} 失败Cell [{next_index}/{self._total_code_cells}] (耗时: {elapsed:.2f}秒): {e}")
                # 继续抛出以遵循 allow_errors=False 的整体行为
                raise
        # 非 code 单元格，按默认行为处理但不计入进度
        return super().preprocess_cell(cell, resources, cell_index)


class NoColorFormatter(logging.Formatter):
    """自定义格式化器，移除ANSI颜色代码"""
    def format(self, record):
        original = super().format(record)
        return self.strip_ansi_codes(original)
    
    def strip_ansi_codes(self, text):
        """移除ANSI颜色代码"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

def setup_logging(log_path):
    """配置日志格式，同时输出到控制台和文件"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    log_dir = log_path
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"notebook_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 清除现有的日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 文件处理器 - 使用NoColorFormatter
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = NoColorFormatter(log_format)
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器 - 使用NoColorFormatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = NoColorFormatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file

def run_notebook(notebook_path, index, total):
    """
    执行指定的Jupyter notebook文件
    
    Args:
        notebook_path: notebook文件路径
        index: 当前执行的序号
        total: 总数量
        
    Returns:
        bool: 执行是否成功
    """
    start_time = time.time()
    logging.info(f"[{index}/{total}] 开始执行: {notebook_path}")
    
    try:
        # 读取notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = read(f, as_version=4)
        

        # 设置执行处理器
        ep = LoggingExecutePreprocessor(
            timeout=None,
            kernel_name='python3',
            allow_errors=False,  # 遇到错误时停止执行
            run_index=index,
            run_total=total
        )
        
        # 执行notebook
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        # 保存执行后的notebook
        output_path = notebook_path.replace('.ipynb', '_executed_first.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            write(nb, f)
            
        execution_time = time.time() - start_time
        logging.info(f"[{index}/{total}] 成功执行: {notebook_path} (耗时: {execution_time:.2f}秒)")
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        logging.error(f"[{index}/{total}] 执行 {notebook_path} 时出错 (耗时: {execution_time:.2f}秒): {str(e)}")
        # 输出详细异常信息
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 日志输出配置
    log_path = '/home/zhuchen/poc/01-mashang/logs'
    log_file = setup_logging(log_path)

    logging.info("="*50)
    logging.info("开始执行notebook批量任务")
    logging.info("="*50)
    
    # 是否要调用统一的临时参数 例如epoch
    use_temp_config = False
    if use_temp_config:
        sys.path.append("/home/zhuchen/poc/01-mashang/config") 
        import temp_config

        # if use temp_config.py, uncomment the following
        logging.info("=="*5 + " config " + "=="*5)
        logging.info(f"start_seed: {temp_config.start_seed}")
        logging.info(f"end_seed: {temp_config.end_seed}")
        logging.info(f"epoch: {temp_config.epoch}")

    # 开始在运行列表中添加需要运行的ipynbs
    notebooks = []

    # cust_list = [
    #     'CA',
    #     'CR',
    #     'CACR',
    #     ]
    # # notebooks = [
    # #     '/home/zhuchen/poc/01-mashang/AlphaV3-ThetaV2_CACR_target2/ronghe.ipynb'
    # # ]

    # # zetaV2 明天会跑完的
    # data_list = ['ZetaV2']
    # target_list = [
    #     # 'target1',
    #     'target2'
    #     ]
    # ipynb_list = [
    #     '3_sample_selection.ipynb',
    #     '4_modeling.ipynb',
    #     ]

    # for ipynb_name in ipynb_list:
    #     for data_name in data_list:
    #         for cust in cust_list:
    #             for target in target_list:
    #                 notebooks.append(f'/home/zhuchen/poc/01-mashang/{data_name}_{cust}_{target}/{ipynb_name}')

    cust_list = [
        'CA',
        'CR',
        'CACR',
        ]
    target_list = [
        'target1',
        'target2'
        ]
    data_list = [
        'ronghe'
        # 'DeltaV1',
        # 'ZetaV2'
        ]
    ipynb_list = [
        # '1_data_preprocess.ipynb',
        # '2_feature_selection.ipynb',
        # '3_sample_selection.ipynb',
        # '4_modeling.ipynb',
        # '5_only_score.ipynb',
        'ronghe.ipynb'
        ]

    
    for data_name in data_list:
        for cust in cust_list:
            for target in target_list:
                for ipynb_name in ipynb_list:
                    notebooks.append(f'/home/zhuchen/poc/01-mashang/{data_name}_{cust}_{target}/{ipynb_name}')

    # notebooks.append('/home/zhuchen/poc/01-mashang/DeltaV1/1_data_preprocess.ipynb')
    


    logging.info("=="*5 + " notebooks " + "=="*5)
    for i, notebook in enumerate(notebooks):
        logging.info(f"{i+1}. {notebook}")
    
    valid_notebooks = []
    for nb_path in notebooks:
        if os.path.exists(nb_path):
            valid_notebooks.append(nb_path)
        else:
            logging.error(f"错误: 文件不存在 - {nb_path}")
    
    if not valid_notebooks:
        logging.error("没有有效的notebook可执行")
        exit(1)
    
    logging.info(f"找到 {len(valid_notebooks)} 个有效的notebook准备执行")
    
    # 执行所有notebook
    results = []
    for i, nb_path in enumerate(valid_notebooks, 1):
        result = run_notebook(nb_path, i, len(valid_notebooks))
        results.append(result)
    
    # 输出总结
    logging.info("="*50)
    logging.info("执行总结:")
    logging.info("="*50)
    
    success_count = 0
    for nb, result in zip(valid_notebooks, results):
        status = "成功" if result else "失败"
        if result:
            success_count += 1
        logging.info(f"{nb}: {status}")
    
    logging.info(f"成功: {success_count}/{len(valid_notebooks)}")
    logging.info(f"日志文件已保存至: {log_file}")
    
    if success_count == len(valid_notebooks):
        logging.info("所有notebook执行成功!")
        exit(0)
    else:
        logging.error("部分notebook执行失败!")
        exit(1)