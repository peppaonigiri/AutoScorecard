import os
import sys
import logging
import time
from datetime import datetime
# sys.path.append(r"D:\work\03.poc\01-马上消费\config") 
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter
from nbformat import read, write
# import temp_config
import warnings
warnings.filterwarnings("ignore")

# 设置日志
def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    log_dir = r"D:\work\03.poc\01-马上消费/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"notebook_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(log_file)      
        ]
    )
    
    return log_file

def run_notebook(notebook_path, index, total):
    start_time = time.time()
    logging.info(f"[{index}/{total}] 开始执行: {notebook_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = read(f, as_version=4)
        
        ep = ExecutePreprocessor(
            timeout=None,
            kernel_name='python3',
            allow_errors=False  # 遇到错误时停止执行
        )
        
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            write(nb, f)
            
        execution_time = time.time() - start_time
        logging.info(f"[{index}/{total}] 成功执行: {notebook_path} (耗时: {execution_time:.2f}秒)")
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        logging.error(f"[{index}/{total}] 执行 {notebook_path} 时出错 (耗时: {execution_time:.2f}秒): {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    log_file = setup_logging()
    logging.info("="*50)
    logging.info("开始执行notebook批量任务")
    logging.info("="*50)
    
    # logging.info("=="*5 + " config " + "=="*5)
    # logging.info(f"start_seed: {temp_config.start_seed}")
    # logging.info(f"end_seed: {temp_config.end_seed}")
    # logging.info(f"epoch: {temp_config.epoch}")

    notebooks = [
        # r'D:\work\03.poc\01-马上消费\ThetaV2_CR_target1\3_sample_selection.ipynb',
        # r'D:\work\03.poc\01-马上消费\ThetaV2_CR_target1\4_modeling.ipynb',
        # r'D:\work\03.poc\01-马上消费\ThetaV2_CR_target2\3_sample_selection.ipynb',
        # r'D:\work\03.poc\01-马上消费\ThetaV2_CR_target2\4_modeling.ipynb'
        r'D:\work\03.poc\01-马上消费\ThetaV2_CACR_target2\3_sample_selection.ipynb',
        r'D:\work\03.poc\01-马上消费\ThetaV2_CACR_target2\4_modeling.ipynb'
    ]
    
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