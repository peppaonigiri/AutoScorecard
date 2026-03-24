import os
import sys
import logging
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import re
import psutil
import platform

# nbconvert相关导入
try:
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat
    from nbformat import read, write
except ImportError as e:
    print(f"导入nbconvert相关模块失败: {e}")
    # 尝试备用导入方式
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        print("无法导入必要的nbconvert模块，请确保已安装nbconvert和nbformat")
        sys.exit(1)

import warnings
warnings.filterwarnings("ignore")

class LoggingExecutePreprocessor(ExecutePreprocessor):
    """自定义执行处理器，输出 cell 级别的进度日志"""
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

class NotebookManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("IPython Notebook 管理器 v2.0")
        self.root.geometry("1100x900")  # 增加高度以容纳内存监控窗口
        
        # 变量初始化
        self.selected_directory = tk.StringVar()
        self.python_path = tk.StringVar(value=sys.executable)  # 默认使用当前Python
        self.output_suffix = tk.StringVar(value="_executed")
        self.log_path = tk.StringVar(value=os.path.join(os.getcwd(), "notebook_execution.log"))
        self.log_text = None
        
        # 进度条相关变量
        self.progress_var = tk.DoubleVar()
        self.progress_text_var = tk.StringVar(value="就绪")
        
        # 内存监控相关变量
        self.memory_var = tk.DoubleVar()
        self.memory_text_var = tk.StringVar(value="内存: 0%")
        self.memory_monitoring = False
        
        # 主题模式
        self.dark_mode = tk.BooleanVar(value=False)  # 默认浅色模式
        
        # 进程隔离配置
        self.kernel_memory_mode = tk.StringVar(value="动态百分比")  # 内存限制模式：固定值 / 动态百分比
        self.max_kernel_memory_gb = tk.DoubleVar(value=8.0)  # 固定模式：单个kernel最大内存限制（GB）
        self.kernel_memory_percent = tk.DoubleVar(value=95.0)  # 动态模式：可用内存的百分比
        self.kernel_priority = tk.StringVar(value="低")  # kernel进程优先级
        
        # 运行时状态变量
        self.current_kernel_pid = None  # 当前运行的kernel PID
        self.current_job_handle = None  # Windows Job Object句柄
        
        # 存储文件树数据
        self.file_tree_data = {}
        self.checkbutton_vars = {}
        self.original_texts = {}  # 存储原始文本，用于正确显示
        
        # 存储运行顺序
        self.execution_order = []
        self.dragged_item = None
        
        self.setup_ui()
        self.setup_logging()
        
        # 启动时默认开始内存监控
        self.root.after(1000, self.start_memory_monitoring)  # 延迟1秒启动，确保界面完全加载
        
    

    def setup_logging(self):
        """配置日志"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # 创建日志目录
        log_dir = os.path.dirname(self.log_path.get())
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 清除现有的日志处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 创建无颜色格式化器
        formatter = NoColorFormatter(log_format)
        
        # 创建处理器并设置格式化器
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        
        text_handler = self.LogTextHandler(self)
        text_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(self.log_path.get(), encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 配置根日志记录器
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(stream_handler)
        logging.root.addHandler(text_handler)
        logging.root.addHandler(file_handler)
        
        logging.info(f"日志文件路径: {self.log_path.get()}")

    def update_logging(self):
        """更新日志配置"""
        # 移除现有的文件处理器
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logging.root.removeHandler(handler)
        
        # 添加新的文件处理器
        log_dir = os.path.dirname(self.log_path.get())
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(self.log_path.get(), encoding='utf-8')
        file_handler.setFormatter(NoColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.root.addHandler(file_handler)
        
        logging.info(f"日志路径已更新: {self.log_path.get()}")
    
    class LogTextHandler(logging.Handler):
        """自定义日志处理器，将日志输出到文本框"""
        def __init__(self, manager):
            super().__init__()
            self.manager = manager
            
        def emit(self, record):
            msg = self.format(record)
            if self.manager.log_text:
                self.manager.log_text.insert(tk.END, msg + '\n')
                self.manager.log_text.see(tk.END)
    
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)  # 文件树区域可扩展
        main_frame.rowconfigure(6, weight=1)  # 日志区域可扩展
        
        # 目录选择区域
        dir_frame = ttk.LabelFrame(main_frame, text="目录设置", padding="5")
        dir_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)
        
        ttk.Button(dir_frame, text="选择目录", command=self.select_directory).grid(row=0, column=0, padx=(0, 10))
        ttk.Entry(dir_frame, textvariable=self.selected_directory, state='readonly').grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(dir_frame, text="刷新文件树", command=self.refresh_file_tree).grid(row=0, column=2, padx=(10, 5))
        
        # 主题切换按钮
        self.theme_toggle_btn = ttk.Button(dir_frame, text="🌙 深色模式", command=self.toggle_theme, width=12)
        self.theme_toggle_btn.grid(row=0, column=3, padx=(5, 0))
        
        # 配置区域
        config_frame = ttk.LabelFrame(main_frame, text="运行配置", padding="5")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(3, weight=1)
        
        # 第一行配置
        ttk.Label(config_frame, text="Python路径:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.python_path, width=25).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 10), pady=2)
        ttk.Button(config_frame, text="浏览", command=self.browse_python_path).grid(row=0, column=2, pady=2)
        
        ttk.Label(config_frame, text="输出文件后缀:").grid(row=0, column=3, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.output_suffix).grid(row=0, column=3, sticky=tk.W, padx=(80, 10), pady=2)
        
        # 第二行配置
        ttk.Label(config_frame, text="日志路径:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.log_path, width=25).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 10), pady=2)
        ttk.Button(config_frame, text="浏览", command=self.browse_log_path).grid(row=1, column=2, pady=2, padx=(0, 5))
        ttk.Button(config_frame, text="更新日志", command=self.update_logging).grid(row=1, column=3, sticky=tk.W, pady=2, padx=(5, 0))
        
        # 第三行配置 - 进程隔离（所有参数在一行，紧凑布局）
        isolation_frame = ttk.Frame(config_frame)
        isolation_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=2)
        
        # 标题
        ttk.Label(isolation_frame, text="进程隔离  |").pack(side=tk.LEFT, padx=(0, 5))
        
        # 内存模式
        ttk.Label(isolation_frame, text="内存模式:").pack(side=tk.LEFT, padx=(0, 3))
        self.memory_mode_combo = ttk.Combobox(isolation_frame, textvariable=self.kernel_memory_mode,
                                       values=["固定值", "动态百分比"], state='readonly', width=10)
        self.memory_mode_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.memory_mode_combo.bind('<<ComboboxSelected>>', self.on_memory_mode_changed)
        
        # 内存参数（固定值或动态百分比）
        self.memory_param_label = ttk.Label(isolation_frame, text="")
        self.memory_param_label.pack(side=tk.LEFT, padx=(0, 3))
        
        # 固定值配置
        self.kernel_memory_spinbox = ttk.Spinbox(isolation_frame, from_=1, to=32, increment=1,
                                            textvariable=self.max_kernel_memory_gb, width=6)
        
        # 动态百分比配置
        self.kernel_memory_percent_spinbox = ttk.Spinbox(isolation_frame, from_=50, to=99, increment=1,
                                            textvariable=self.kernel_memory_percent, width=6)
        
        # 进程优先级
        self.priority_label = ttk.Label(isolation_frame, text="优先级:")
        self.priority_label.pack(side=tk.LEFT, padx=(8, 3))
        self.priority_combo = ttk.Combobox(isolation_frame, textvariable=self.kernel_priority,
                                     values=["低", "正常", "高"], state='readonly', width=6)
        self.priority_combo.pack(side=tk.LEFT, padx=(0, 0))
        
        # 初始化显示
        self.update_memory_mode_ui()
        
        # 文件树和排序区域 - 使用PanedWindow实现可调整宽度
        tree_sort_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        tree_sort_paned.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 文件树区域
        tree_frame = ttk.LabelFrame(tree_sort_paned, text="IPython Notebook 文件", padding="5")
        tree_sort_paned.add(tree_frame, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.rowconfigure(1, weight=0)  # 按钮区域不扩展
        
        # 创建树形视图容器框架
        tree_container = ttk.Frame(tree_frame)
        tree_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)
        
        # 创建树形视图 - 添加列并允许调整列宽
        self.tree = ttk.Treeview(tree_container, columns=('fullpath', 'type'), show='tree headings', selectmode='none')
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 设置列
        self.tree.column('#0', width=300, minwidth=150, stretch=True)
        self.tree.column('fullpath', width=0, stretch=False, minwidth=0)  # 隐藏列，仅用于存储数据
        self.tree.column('type', width=0, stretch=False, minwidth=0)  # 隐藏列，仅用于存储数据
        
        # 隐藏标题
        self.tree.heading('#0', text='文件结构')
        self.tree.heading('fullpath', text='')
        self.tree.heading('type', text='')
        
        # 垂直滚动条
        v_scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        
        # 水平滚动条
        h_scrollbar = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        
        # 文件树按钮区域
        tree_button_frame = ttk.Frame(tree_frame)
        tree_button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(tree_button_frame, text="全选", command=self.select_all).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(tree_button_frame, text="全不选", command=self.unselect_all).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(tree_button_frame, text="添加到运行列表", command=self.add_to_order_list).pack(side=tk.LEFT)
        
        # 运行顺序区域
        order_frame = ttk.LabelFrame(tree_sort_paned, text="运行顺序", padding="5")
        tree_sort_paned.add(order_frame, weight=1)
        order_frame.columnconfigure(0, weight=1)
        order_frame.rowconfigure(0, weight=1)
        order_frame.rowconfigure(1, weight=0)
        
        # 运行顺序列表容器框架
        order_container = ttk.Frame(order_frame)
        order_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        order_container.columnconfigure(0, weight=1)
        order_container.rowconfigure(0, weight=1)
        
        # 运行顺序列表 - 使用Treeview并允许调整列宽
        self.order_tree = ttk.Treeview(order_container, columns=('fullpath',), show='tree headings', selectmode='browse', height=15)
        self.order_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 设置列
        self.order_tree.column('#0', width=300, minwidth=150, stretch=True)
        self.order_tree.column('fullpath', width=0, stretch=False, minwidth=0)  # 隐藏列，仅用于存储数据
        
        # 设置标题
        self.order_tree.heading('#0', text='运行顺序')
        self.order_tree.heading('fullpath', text='')
        
        # 运行顺序列表的垂直滚动条
        order_v_scrollbar = ttk.Scrollbar(order_container, orient=tk.VERTICAL, command=self.order_tree.yview)
        order_v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.order_tree.configure(yscrollcommand=order_v_scrollbar.set)
        
        # 运行顺序列表的水平滚动条
        order_h_scrollbar = ttk.Scrollbar(order_container, orient=tk.HORIZONTAL, command=self.order_tree.xview)
        order_h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.order_tree.configure(xscrollcommand=order_h_scrollbar.set)
        
        # 绑定拖拽事件
        self.order_tree.bind('<Button-1>', self.on_order_tree_click)
        self.order_tree.bind('<B1-Motion>', self.on_order_tree_drag)
        self.order_tree.bind('<ButtonRelease-1>', self.on_order_tree_release)
        
        # 运行顺序操作按钮
        order_btn_frame = ttk.Frame(order_frame)
        order_btn_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(order_btn_frame, text="移除选中项", command=self.remove_selected_from_order_list).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(order_btn_frame, text="按文件名排序", command=self.sort_by_filename).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(order_btn_frame, text="按路径排序", command=self.sort_by_path).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(order_btn_frame, text="清空列表", command=self.clear_order_list).pack(side=tk.LEFT)
        
        # 操作按钮和进度条区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(button_frame, text="运行选中文件", command=self.run_selected_notebooks).pack(side=tk.LEFT, padx=(0, 10))
        
        # 进度条区域
        progress_frame = ttk.Frame(button_frame)
        progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 进度条文本标签
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_text_var, font=('Arial', 9))
        progress_label.pack(side=tk.LEFT, padx=(5, 5))
        
        # 进度条
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Kernel状态显示区域
        kernel_status_frame = ttk.LabelFrame(main_frame, text="Kernel状态", padding="3")
        kernel_status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        kernel_status_frame.columnconfigure(0, weight=1)
        
        # Kernel状态显示（只读，不可调整）
        self.kernel_status_var = tk.StringVar(value="无运行中的kernel")
        kernel_status_label = ttk.Label(kernel_status_frame, textvariable=self.kernel_status_var, 
                                        font=('Consolas', 9), foreground='#555555', wraplength=1000)
        kernel_status_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=3, padx=8)
        
        # 内存监控区域 - 紧凑布局
        memory_frame = ttk.LabelFrame(main_frame, text="系统内存监控", padding="3")
        memory_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        memory_frame.columnconfigure(0, weight=1)
        
        # 内存监控控制按钮
        memory_control_frame = ttk.Frame(memory_frame)
        memory_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        self.memory_start_btn = ttk.Button(memory_control_frame, text="开始监控", command=self.start_memory_monitoring)
        self.memory_start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.memory_stop_btn = ttk.Button(memory_control_frame, text="停止监控", command=self.stop_memory_monitoring, state='disabled')
        self.memory_stop_btn.pack(side=tk.LEFT)
        
        # 内存进度条
        self.memory_bar = ttk.Progressbar(memory_frame, variable=self.memory_var, 
                                        maximum=100, mode='determinate')
        self.memory_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 内存使用率和详细信息在同一行
        memory_info_details_frame = ttk.Frame(memory_frame)
        memory_info_details_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.memory_total_var = tk.StringVar(value="总计: 0 GB")
        self.memory_used_var = tk.StringVar(value="已用: 0 GB")
        self.memory_available_var = tk.StringVar(value="可用: 0 GB")
        
        # 内存使用率标签
        self.memory_percent_label = ttk.Label(memory_info_details_frame, textvariable=self.memory_text_var, font=('Arial', 9, 'bold'))
        self.memory_percent_label.pack(side=tk.LEFT, padx=(0, 15))
        
        # 内存详细信息 - 紧挨着显示
        ttk.Label(memory_info_details_frame, textvariable=self.memory_total_var, font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(memory_info_details_frame, textvariable=self.memory_used_var, font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(memory_info_details_frame, textvariable=self.memory_available_var, font=('Arial', 8)).pack(side=tk.LEFT)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="5")
        log_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 保存需要更新主题的组件引用
        self.themed_widgets = {
            'main_frame': main_frame,
            'log_text': self.log_text,
            'tree': self.tree,
            'order_tree': self.order_tree,
        }
        
        # 配置样式
        self.style = ttk.Style()
        
        # 尝试使用 'clam' 主题（更容易自定义）
        try:
            self.style.theme_use('clam')
        except:
            pass  # 如果clam不可用，使用默认主题
        
        self.style.configure('Accent.TButton', foreground='Black', background='#0078D4')
        
        # 配置内存监控进度条样式
        self.style.configure('Normal.Horizontal.TProgressbar', background='#4CAF50')  # 绿色
        self.style.configure('Warning.Horizontal.TProgressbar', background='#FF9800')  # 橙色
        self.style.configure('Danger.Horizontal.TProgressbar', background='#F44336')   # 红色
        
        # 应用初始主题
        self.apply_theme()
    
    def toggle_theme(self):
        """切换深色/浅色主题"""
        self.dark_mode.set(not self.dark_mode.get())
        self.apply_theme()
        
        # 更新按钮文本
        if self.dark_mode.get():
            self.theme_toggle_btn.config(text="☀️ 浅色模式")
        else:
            self.theme_toggle_btn.config(text="🌙 深色模式")
    
    def apply_theme(self):
        """应用当前主题 - Windows资源管理器风格"""
        is_dark = self.dark_mode.get()
        
        if is_dark:
            # 深色主题配色 - 参考Windows 11深色模式（使用浅灰色而非纯白）
            bg_color = '#202020'           # 主背景
            fg_color = '#e0e0e0'           # 主文字
            text_bg = '#2b2b2b'            # 文本框背景
            text_fg = '#e0e0e0'            # 文本框文字
            entry_bg = '#3c3c3c'           # 输入框背景（浅灰色）
            entry_fg = '#e0e0e0'           # 输入框文字
            button_bg = '#404040'          # 按钮背景（浅灰色，不是白色）
            button_fg = '#e0e0e0'          # 按钮文字
            button_hover = '#4a4a4a'       # 按钮悬停（稍亮）
            button_active = '#505050'      # 按钮按下（更亮）
            select_bg = '#0078d4'          # 选中背景
            select_fg = '#ffffff'          # 选中文字
            frame_bg = '#2b2b2b'           # 框架背景
            label_fg = '#e0e0e0'           # 标签文字
            border_color = '#555555'       # 边框颜色（更明显的灰色）
            disabled_bg = '#333333'        # 禁用背景
            disabled_fg = '#808080'        # 禁用文字
            # 只读状态使用更深的灰色
            readonly_bg = '#353535'        # 只读背景
        else:
            # 浅色主题配色
            bg_color = '#f3f3f3'
            fg_color = '#000000'
            text_bg = '#ffffff'
            text_fg = '#000000'
            entry_bg = '#ffffff'
            entry_fg = '#000000'
            button_bg = '#e1e1e1'
            button_fg = '#000000'
            button_hover = '#d0d0d0'
            button_active = '#c0c0c0'
            select_bg = '#0078d4'
            select_fg = '#ffffff'
            frame_bg = '#f3f3f3'
            label_fg = '#000000'
            border_color = '#cccccc'
            disabled_bg = '#f0f0f0'
            disabled_fg = '#a0a0a0'
            readonly_bg = '#f5f5f5'
        
        # 应用到根窗口
        self.root.configure(bg=bg_color)
        
        # 应用到主框架
        if 'main_frame' in self.themed_widgets:
            try:
                self.themed_widgets['main_frame'].configure(style='Custom.TFrame')
                self.style.configure('Custom.TFrame', background=frame_bg)
            except:
                pass
        
        # ===== 配置Frame样式 =====
        self.style.configure('TFrame', background=frame_bg)
        self.style.configure('TLabelframe', 
                           background=frame_bg, 
                           foreground=label_fg,
                           bordercolor=border_color,
                           relief='flat')
        self.style.configure('TLabelframe.Label', 
                           background=frame_bg, 
                           foreground=label_fg)
        
        # ===== 配置Label样式 =====
        self.style.configure('TLabel', 
                           background=frame_bg, 
                           foreground=label_fg)
        
        # ===== 配置Button样式 =====
        self.style.configure('TButton',
                           background=button_bg,
                           foreground=button_fg,
                           bordercolor=border_color,
                           lightcolor=border_color,
                           darkcolor=border_color,
                           relief='flat',
                           borderwidth=1,
                           focuscolor='none',
                           padding=(10, 5))
        self.style.map('TButton',
                      background=[('active', button_active), 
                                ('pressed', button_active),
                                ('!disabled', button_bg),
                                ('disabled', disabled_bg)],
                      foreground=[('!disabled', button_fg),
                                ('disabled', disabled_fg)],
                      bordercolor=[('focus', select_bg)],
                      relief=[('pressed', 'sunken'), ('!pressed', 'flat')])
        
        # ===== 配置Entry样式 =====
        self.style.configure('TEntry',
                           fieldbackground=entry_bg,
                           foreground=entry_fg,
                           bordercolor=border_color,
                           lightcolor=border_color,
                           darkcolor=border_color,
                           insertcolor=entry_fg,
                           selectbackground=select_bg,
                           selectforeground=select_fg,
                           relief='flat')
        self.style.map('TEntry',
                      fieldbackground=[('readonly', readonly_bg), ('disabled', disabled_bg)],
                      foreground=[('readonly', entry_fg), ('disabled', disabled_fg)],
                      bordercolor=[('focus', select_bg)])
        
        # ===== 配置Combobox样式 =====
        self.style.configure('TCombobox',
                           fieldbackground=entry_bg,
                           background=button_bg,
                           foreground=entry_fg,
                           arrowcolor=label_fg,
                           bordercolor=border_color,
                           lightcolor=border_color,
                           darkcolor=border_color,
                           selectbackground=select_bg,
                           selectforeground=select_fg,
                           relief='flat')
        self.style.map('TCombobox',
                      fieldbackground=[('readonly', entry_bg), ('disabled', disabled_bg)],
                      foreground=[('readonly', entry_fg), ('disabled', disabled_fg)],
                      background=[('readonly', button_bg), ('!disabled', button_bg), ('disabled', disabled_bg)],
                      bordercolor=[('focus', select_bg)],
                      arrowcolor=[('!disabled', label_fg), ('disabled', disabled_fg)])
        
        # ===== 配置Spinbox样式 =====
        self.style.configure('TSpinbox',
                           fieldbackground=entry_bg,
                           background=button_bg,
                           foreground=entry_fg,
                           arrowcolor=label_fg,
                           bordercolor=border_color,
                           lightcolor=border_color,
                           darkcolor=border_color,
                           selectbackground=select_bg,
                           selectforeground=select_fg,
                           relief='flat')
        self.style.map('TSpinbox',
                      fieldbackground=[('readonly', entry_bg), ('disabled', disabled_bg)],
                      foreground=[('readonly', entry_fg), ('disabled', disabled_fg)],
                      background=[('!disabled', button_bg), ('disabled', disabled_bg)],
                      bordercolor=[('focus', select_bg)],
                      arrowcolor=[('!disabled', label_fg), ('disabled', disabled_fg)])
        
        # ===== 配置Treeview样式 =====
        self.style.configure('Treeview',
                           background=text_bg,
                           foreground=text_fg,
                           fieldbackground=text_bg,
                           bordercolor=border_color,
                           lightcolor=frame_bg,
                           darkcolor=frame_bg)
        self.style.configure('Treeview.Heading',
                           background=button_bg,
                           foreground=label_fg,
                           relief='flat',
                           borderwidth=1)
        self.style.map('Treeview',
                      background=[('selected', select_bg)],
                      foreground=[('selected', select_fg)])
        self.style.map('Treeview.Heading',
                      background=[('active', button_hover)],
                      relief=[('pressed', 'sunken')])
        
        # ===== 配置Progressbar样式 =====
        if is_dark:
            self.style.configure('Normal.Horizontal.TProgressbar', 
                               background='#66BB6A', 
                               troughcolor='#333333',
                               bordercolor=border_color,
                               lightcolor='#66BB6A',
                               darkcolor='#66BB6A')
            self.style.configure('Warning.Horizontal.TProgressbar', 
                               background='#FFA726', 
                               troughcolor='#333333',
                               bordercolor=border_color)
            self.style.configure('Danger.Horizontal.TProgressbar', 
                               background='#EF5350', 
                               troughcolor='#333333',
                               bordercolor=border_color)
            self.style.configure('Horizontal.TProgressbar', 
                               background='#64B5F6', 
                               troughcolor='#333333',
                               bordercolor=border_color)
        else:
            self.style.configure('Normal.Horizontal.TProgressbar', 
                               background='#4CAF50', 
                               troughcolor='#e0e0e0',
                               bordercolor=border_color)
            self.style.configure('Warning.Horizontal.TProgressbar', 
                               background='#FF9800', 
                               troughcolor='#e0e0e0',
                               bordercolor=border_color)
            self.style.configure('Danger.Horizontal.TProgressbar', 
                               background='#F44336', 
                               troughcolor='#e0e0e0',
                               bordercolor=border_color)
            self.style.configure('Horizontal.TProgressbar', 
                               background='#2196F3', 
                               troughcolor='#e0e0e0',
                               bordercolor=border_color)
        
        # ===== 配置Scrollbar样式 =====
        self.style.configure('Vertical.TScrollbar',
                           background=button_bg,
                           troughcolor=frame_bg,
                           bordercolor=border_color,
                           arrowcolor=label_fg)
        self.style.map('Vertical.TScrollbar',
                      background=[('active', button_hover)])
        
        self.style.configure('Horizontal.TScrollbar',
                           background=button_bg,
                           troughcolor=frame_bg,
                           bordercolor=border_color,
                           arrowcolor=label_fg)
        self.style.map('Horizontal.TScrollbar',
                      background=[('active', button_hover)])
        
        # ===== 配置PanedWindow样式 =====
        # sash是分隔条的颜色
        if is_dark:
            sash_color = '#404040'  # 深色模式：浅灰色
        else:
            sash_color = '#d0d0d0'  # 浅色模式：中灰色
        
        self.style.configure('TPanedwindow',
                           background=frame_bg)
        # 注意：sashthickness设置分隔条的宽度
        self.style.configure('Sash',
                           sashthickness=5,
                           background=sash_color,
                           bordercolor=sash_color,
                           lightcolor=sash_color,
                           darkcolor=sash_color)
        
        # ===== 配置日志文本框 =====
        if 'log_text' in self.themed_widgets and self.themed_widgets['log_text']:
            self.themed_widgets['log_text'].configure(
                bg=text_bg,
                fg=text_fg,
                insertbackground=text_fg,
                selectbackground=select_bg,
                selectforeground=select_fg,
                borderwidth=1,
                relief='flat',
                highlightthickness=1,
                highlightbackground=border_color,
                highlightcolor=select_bg
            )
        
        # ===== 配置树形视图 =====
        if 'tree' in self.themed_widgets and self.themed_widgets['tree']:
            self.themed_widgets['tree'].configure(style='Treeview')
        
        if 'order_tree' in self.themed_widgets and self.themed_widgets['order_tree']:
            self.themed_widgets['order_tree'].configure(style='Treeview')
        
        logging.info(f"已切换到{'深色' if is_dark else '浅色'}主题")
    
    def select_directory(self):
        """选择目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.selected_directory.set(directory)
            self.refresh_file_tree()
    
    def browse_python_path(self):
        """浏览Python路径"""
        file_path = filedialog.askopenfilename(
            title="选择Python解释器",
            filetypes=[("Python Executable", "python.exe"), ("All files", "*.*")]
        )
        if file_path:
            self.python_path.set(file_path)
    
    def browse_log_path(self):
        """浏览日志路径"""
        file_path = filedialog.asksaveasfilename(
            title="选择日志文件",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if file_path:
            self.log_path.set(file_path)
    
    def on_memory_mode_changed(self, event=None):
        """内存模式改变时的回调"""
        self.update_memory_mode_ui()
    
    def update_memory_mode_ui(self):
        """根据内存模式更新UI显示"""
        mode = self.kernel_memory_mode.get()
        
        # 移除所有配置
        self.kernel_memory_spinbox.pack_forget()
        self.kernel_memory_percent_spinbox.pack_forget()
        
        if mode == "固定值":
            # 显示固定值配置
            self.memory_param_label.config(text="最大内存(GB):")
            self.kernel_memory_spinbox.pack(side=tk.LEFT, padx=(0, 8), before=self.priority_label)
        elif mode == "动态百分比":
            # 显示动态百分比配置
            self.memory_param_label.config(text="可用内存比例(%):")
            self.kernel_memory_percent_spinbox.pack(side=tk.LEFT, padx=(0, 8), before=self.priority_label)
    
    def refresh_file_tree(self):
        """刷新文件树"""
        directory = self.selected_directory.get()
        if not directory or not os.path.exists(directory):
            return
        
        # 清空现有树
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.file_tree_data = {}
        self.checkbutton_vars = {}
        self.original_texts = {}  # 清空原始文本存储
        
        # 扫描.ipynb文件
        ipynb_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.ipynb'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, directory)
                    ipynb_files.append((relative_path, full_path))
        
        # 按路径排序
        ipynb_files.sort()
        
        # 构建树形结构
        for relative_path, full_path in ipynb_files:
            parts = relative_path.split(os.sep)
            parent = ''
            
            # 构建目录层级
            for i, part in enumerate(parts[:-1]):
                current_path = os.sep.join(parts[:i+1])
                if current_path not in self.file_tree_data:
                    item_id = self.tree.insert(parent, 'end', text=part, values=('', 'directory'))
                    self.file_tree_data[current_path] = item_id
                    self.original_texts[item_id] = part  # 保存原始文本
                    parent = item_id
                else:
                    parent = self.file_tree_data[current_path]
            
            # 添加文件节点
            filename = parts[-1]
            item_id = self.tree.insert(parent, 'end', text=filename, values=(full_path, 'file'))
            self.original_texts[item_id] = filename  # 保存原始文本
            
            # 添加复选框
            var = tk.BooleanVar()
            self.checkbutton_vars[full_path] = var
            
            # 由于Treeview不支持直接嵌入Checkbutton，我们使用标签来模拟
            self.tree.item(item_id, tags=('checkable',))
        
        # 绑定点击事件来模拟复选框
        self.tree.tag_bind('checkable', '<Button-1>', self.on_item_click)
        
        # 展开所有节点
        for item in self.tree.get_children():
            self.tree.item(item, open=True)
    
    def on_item_click(self, event):
        """处理树形项目点击事件 - 修复版"""
        item = self.tree.identify_row(event.y)
        if item and 'checkable' in self.tree.item(item, 'tags'):
            full_path = self.tree.item(item, 'values')[0]
            if full_path in self.checkbutton_vars:
                # 获取当前状态并取反
                current_state = self.checkbutton_vars[full_path].get()
                new_state = not current_state
                self.checkbutton_vars[full_path].set(new_state)
                
                # 更新显示
                self.update_item_display(item)
    
    def update_item_display(self, item):
        """更新项目显示状态 - 修复版"""
        full_path = self.tree.item(item, 'values')[0]
        if full_path in self.checkbutton_vars:
            is_checked = self.checkbutton_vars[full_path].get()
            
            # 使用保存的原始文本，而不是当前显示的文本
            original_text = self.original_texts.get(item, self.tree.item(item, 'text'))
            
            
            # 根据状态添加正确的前缀
            if is_checked:
                new_text = f"✓ {original_text}"
            else:
                new_text = f"{original_text}"
                
            self.tree.item(item, text=new_text)
    
    def _update_file_tree_item_display(self, file_path):
        """根据文件路径更新文件树中对应项的显示"""
        # 遍历文件树找到对应的节点
        for item in self.tree.get_children():
            self._find_and_update_item(item, file_path)
    
    def _find_and_update_item(self, item, file_path):
        """递归查找并更新文件树节点"""
        values = self.tree.item(item, 'values')
        if values and len(values) > 0 and values[0] == file_path:
            # 找到了，更新显示
            self.update_item_display(item)
            return True
        
        # 递归查找子节点
        for child in self.tree.get_children(item):
            if self._find_and_update_item(child, file_path):
                return True
        
        return False
    
    def select_all(self):
        """全选"""
        for var in self.checkbutton_vars.values():
            var.set(True)
        self.update_all_display()
    
    def unselect_all(self):
        """全不选"""
        for var in self.checkbutton_vars.values():
            var.set(False)
        self.update_all_display()
    
    def update_all_display(self):
        """更新所有项目的显示"""
        for item in self.tree.get_children():
            self.update_children_display(item)
    
    def update_children_display(self, parent):
        """递归更新子项目显示"""
        for item in self.tree.get_children(parent):
            full_path = self.tree.item(item, 'values')[0]
            if full_path:  # 文件节点
                self.update_item_display(item)
            else:  # 目录节点
                self.update_children_display(item)
    
    def get_selected_notebooks(self):
        """获取选中的notebook文件"""
        selected = []
        for full_path, var in self.checkbutton_vars.items():
            if var.get():
                selected.append(full_path)
        return selected
    
    def add_to_order_list(self):
        """将选中的文件添加到运行顺序列表"""
        selected_files = self.get_selected_notebooks()
        
        for file_path in selected_files:
            # 检查是否已在列表中
            if file_path not in self.execution_order:
                self.execution_order.append(file_path)
                
                # 获取相对路径和父目录名
                if self.selected_directory.get():
                    relative_path = os.path.relpath(file_path, self.selected_directory.get())
                    parent_dir = os.path.dirname(relative_path)
                    
                    # 如果文件在根目录，显示文件名；否则显示父目录/文件名
                    if parent_dir == '.':
                        display_text = os.path.basename(file_path)
                    else:
                        display_text = f"{parent_dir}/{os.path.basename(file_path)}"
                else:
                    # 如果没有选择目录，显示完整路径
                    display_text = file_path
                
                # 添加到运行顺序树
                self.order_tree.insert('', 'end', text=display_text, values=(file_path,))
            
            # 取消文件树中的选中状态
            if file_path in self.checkbutton_vars:
                self.checkbutton_vars[file_path].set(False)
                # 更新显示 - 需要找到对应的树节点
                self._update_file_tree_item_display(file_path)
    
    def remove_selected_from_order_list(self):
        """从运行顺序列表中移除选中的项"""
        selected_items = self.order_tree.selection()
        
        if not selected_items:
            messagebox.showinfo("提示", "请先在运行顺序列表中选择要移除的项")
            return
        
        # 收集要移除的文件路径
        files_to_remove = []
        for item in selected_items:
            file_path = self.order_tree.item(item, 'values')[0]
            files_to_remove.append(file_path)
        
        # 从execution_order中移除
        for file_path in files_to_remove:
            if file_path in self.execution_order:
                self.execution_order.remove(file_path)
        
        # 从树中删除
        for item in selected_items:
            self.order_tree.delete(item)
    
    def clear_order_list(self):
        """清空运行顺序列表"""
        self.execution_order.clear()
        for item in self.order_tree.get_children():
            self.order_tree.delete(item)
    
    def sort_by_filename(self):
        """按文件名排序运行顺序列表"""
        if not self.execution_order:
            return
        
        # 按文件名排序
        sorted_files = sorted(self.execution_order, key=lambda x: os.path.basename(x).lower())
        
        # 更新列表
        self.execution_order = sorted_files
        self.update_order_tree()
    
    def sort_by_path(self):
        """按完整路径排序运行顺序列表"""
        if not self.execution_order:
            return
        
        # 按完整路径排序
        sorted_files = sorted(self.execution_order)
        
        # 更新列表
        self.execution_order = sorted_files
        self.update_order_tree()
    
    def update_order_tree(self):
        """更新运行顺序树的显示"""
        # 清空现有树
        for item in self.order_tree.get_children():
            self.order_tree.delete(item)
        
        # 重新添加所有项目
        for file_path in self.execution_order:
            # 获取相对路径和父目录名
            if self.selected_directory.get():
                relative_path = os.path.relpath(file_path, self.selected_directory.get())
                parent_dir = os.path.dirname(relative_path)
                
                # 如果文件在根目录，显示文件名；否则显示父目录/文件名
                if parent_dir == '.':
                    display_text = os.path.basename(file_path)
                else:
                    display_text = f"{parent_dir}/{os.path.basename(file_path)}"
            else:
                # 如果没有选择目录，显示完整路径
                display_text = file_path
            
            # 添加到运行顺序树
            self.order_tree.insert('', 'end', text=display_text, values=(file_path,))
    
    def on_order_tree_click(self, event):
        """处理运行顺序树点击事件"""
        item = self.order_tree.identify_row(event.y)
        if item:
            self.dragged_item = item
    
    def on_order_tree_drag(self, event):
        """处理运行顺序树拖拽事件"""
        # 这里不执行任何操作，但需要这个事件处理器来启用拖拽
        pass
    
    def on_order_tree_release(self, event):
        """处理运行顺序树释放事件"""
        if self.dragged_item is None:
            return
        
        # 获取释放位置的索引
        release_item = self.order_tree.identify_row(event.y)
        
        # 如果拖拽到了有效位置
        if release_item and release_item != self.dragged_item:
            # 获取被拖拽的项目
            dragged_index = self.execution_order.index(self.order_tree.item(self.dragged_item, 'values')[0])
            dragged_path = self.execution_order[dragged_index]
            
            # 从原位置删除
            self.execution_order.pop(dragged_index)
            self.order_tree.delete(self.dragged_item)
            
            # 获取新位置的索引
            all_items = list(self.order_tree.get_children())
            if release_item in all_items:
                release_index = all_items.index(release_item)
            else:
                release_index = len(all_items)
            
            # 插入到新位置
            self.execution_order.insert(release_index, dragged_path)
            
            # 重新构建运行顺序树
            self.update_order_tree()
            
            # 选中新位置的项目
            new_items = list(self.order_tree.get_children())
            if release_index < len(new_items):
                self.order_tree.selection_set(new_items[release_index])
        
        self.dragged_item = None
    
    def start_memory_monitoring(self):
        """开始内存监控"""
        if not self.memory_monitoring:
            self.memory_monitoring = True
            self.memory_start_btn.config(state='disabled')
            self.memory_stop_btn.config(state='normal')
            
            # 启动内存监控线程
            self.memory_thread = threading.Thread(target=self._memory_monitoring_thread, daemon=True)
            self.memory_thread.start()
            
            logging.info("内存监控已启动")
    
    def stop_memory_monitoring(self):
        """停止内存监控"""
        if self.memory_monitoring:
            self.memory_monitoring = False
            self.memory_start_btn.config(state='normal')
            self.memory_stop_btn.config(state='disabled')
            
            # 重置显示
            self.memory_var.set(0)
            self.memory_text_var.set("内存: 0%")
            self.memory_total_var.set("总计: 0 GB")
            self.memory_used_var.set("已用: 0 GB")
            self.memory_available_var.set("可用: 0 GB")
            
            logging.info("内存监控已停止")
    
    def _memory_monitoring_thread(self):
        """内存监控线程"""
        while self.memory_monitoring:
            try:
                # 获取内存信息
                memory = psutil.virtual_memory()
                
                # 计算内存使用率
                memory_percent = memory.percent
                
                # 转换字节到GB
                total_gb = memory.total / (1024**3)
                used_gb = memory.used / (1024**3)
                available_gb = memory.available / (1024**3)
                
                # 更新UI（在主线程中执行）
                self.root.after(0, lambda: self._update_memory_display(
                    memory_percent, total_gb, used_gb, available_gb
                ))
                
                # 每秒更新一次
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"内存监控出错: {e}")
                time.sleep(1)
    
    def _update_memory_display(self, percent, total_gb, used_gb, available_gb):
        """更新内存显示"""
        if self.memory_monitoring:
            # 更新进度条
            self.memory_var.set(percent)
            
            # 更新文本显示
            self.memory_text_var.set(f"内存: {percent:.1f}%")
            self.memory_total_var.set(f"总计: {total_gb:.1f} GB")
            self.memory_used_var.set(f"已用: {used_gb:.1f} GB")
            self.memory_available_var.set(f"可用: {available_gb:.1f} GB")
            
            # 根据内存使用率改变进度条颜色
            if percent > 90:
                self.memory_bar.config(style='Danger.Horizontal.TProgressbar')
            elif percent > 70:
                self.memory_bar.config(style='Warning.Horizontal.TProgressbar')
            else:
                self.memory_bar.config(style='Normal.Horizontal.TProgressbar')
    
    
    def apply_process_resource_limits(self, process):
        """
        对进程应用资源限制
        
        Args:
            process: psutil.Process对象
        """
        try:
            system = platform.system()
            
            # 1. 设置进程优先级
            priority_map = {
                "低": psutil.BELOW_NORMAL_PRIORITY_CLASS if system == "Windows" else 10,
                "正常": psutil.NORMAL_PRIORITY_CLASS if system == "Windows" else 0,
                "高": psutil.ABOVE_NORMAL_PRIORITY_CLASS if system == "Windows" else -10
            }
            
            priority = self.kernel_priority.get()
            if priority in priority_map:
                if system == "Windows":
                    process.nice(priority_map[priority])
                else:
                    process.nice(priority_map[priority])
                logging.info(f"✅ 已设置进程优先级: {priority}")
            
            # 2. 设置内存限制（根据平台和模式）
            # 使用动态计算的限制值
            memory_limit_gb = self.calculate_dynamic_memory_limit()
            max_memory_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
            
            if system == "Linux":
                # Linux: 使用resource模块
                try:
                    import resource
                    # 设置虚拟内存限制
                    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
                    logging.info(f"✅ 已设置内存限制: {memory_limit_gb:.1f}GB (Linux)")
                except Exception as e:
                    logging.warning(f"设置Linux内存限制失败: {e}")
            
            elif system == "Windows":
                # Windows: 使用Job Objects API
                try:
                    job_handle = self._apply_windows_job_limit(process.pid, max_memory_bytes)
                    if job_handle:
                        self.current_job_handle = job_handle
                        logging.info(f"✅ 已设置内存限制: {memory_limit_gb:.1f}GB (Windows)")
                except Exception as e:
                    logging.warning(f"设置Windows内存限制失败: {e}")
            
            else:
                logging.warning(f"不支持的操作系统: {system}，跳过内存限制设置")
                
        except Exception as e:
            logging.error(f"应用进程资源限制失败: {e}")
    
    def _get_windows_job_structures(self):
        """
        获取Windows Job Object的结构体定义（避免重复定义）
        
        Returns:
            tuple: (IO_COUNTERS, JOBOBJECT_BASIC_LIMIT_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
        """
        import ctypes
        
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]
        
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]
        
        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]
        
        return IO_COUNTERS, JOBOBJECT_BASIC_LIMIT_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    
    def _apply_windows_job_limit(self, pid, max_memory_bytes):
        """
        使用Windows Job Objects限制进程内存（Windows专用）
        
        Args:
            pid: 进程ID
            max_memory_bytes: 最大内存字节数
            
        Returns:
            job_handle: Job Object句柄（用于后续更新）
        """
        try:
            import ctypes
            from ctypes import wintypes
            
            # Windows API常量
            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
            JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
            
            # 创建Job Object
            kernel32 = ctypes.windll.kernel32
            job = kernel32.CreateJobObjectW(None, None)
            
            if not job:
                logging.warning("创建Job Object失败")
                return None
            
            # 获取结构体定义
            _, _, JOBOBJECT_EXTENDED_LIMIT_INFORMATION = self._get_windows_job_structures()
            
            limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            # 设置限制标志和内存限制
            ctypes.memset(ctypes.byref(limit_info), 0, ctypes.sizeof(limit_info))
            
            # 设置LimitFlags
            limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
            
            limit_info.ProcessMemoryLimit = max_memory_bytes
            limit_info.JobMemoryLimit = max_memory_bytes
            
            # 应用限制
            result = kernel32.SetInformationJobObject(
                job, 9,  # JobObjectExtendedLimitInformation
                ctypes.byref(limit_info),
                ctypes.sizeof(limit_info)
            )
            
            if not result:
                error_code = kernel32.GetLastError()
                logging.warning(f"设置Job Object限制失败，错误代码: {error_code}")
            
            # 将进程添加到Job
            process_handle = kernel32.OpenProcess(0x1F0FFF, False, pid)  # PROCESS_ALL_ACCESS
            if process_handle:
                kernel32.AssignProcessToJobObject(job, process_handle)
                kernel32.CloseHandle(process_handle)
                logging.info(f"🛡️ 已将进程{pid}添加到Job Object")
                
                # 保存job句柄供后续更新使用
                return job
            
            return None
            
        except Exception as e:
            logging.warning(f"Windows Job Object设置失败: {e}")
            return None
    
    def update_kernel_memory_limit_realtime(self, new_limit_gb):
        """
        实时更新kernel进程的内存限制
        
        Args:
            new_limit_gb: 新的内存限制（GB）
        """
        if not self.current_kernel_pid:
            logging.warning("没有运行中的kernel")
            return False
        
        try:
            system = platform.system()
            new_limit_bytes = int(new_limit_gb * 1024 * 1024 * 1024)
            
            if system == "Windows" and self.current_job_handle:
                # Windows: 更新Job Object限制
                import ctypes
                
                JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
                JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
                
                # 获取结构体定义（避免重复定义）
                _, _, JOBOBJECT_EXTENDED_LIMIT_INFORMATION = self._get_windows_job_structures()
                
                kernel32 = ctypes.windll.kernel32
                
                # 先读取当前设置
                limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                return_length = ctypes.c_ulong()
                
                query_result = kernel32.QueryInformationJobObject(
                    self.current_job_handle, 9,  # JobObjectExtendedLimitInformation
                    ctypes.byref(limit_info),
                    ctypes.sizeof(limit_info),
                    ctypes.byref(return_length)
                )
                
                if not query_result:
                    error_code = kernel32.GetLastError()
                    logging.debug(f"查询Job Object信息失败，错误代码: {error_code}，尝试直接设置")
                    # 如果查询失败，创建新的设置
                    ctypes.memset(ctypes.byref(limit_info), 0, ctypes.sizeof(limit_info))
                    limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
                
                # 更新内存限制值（保持现有的LimitFlags）
                limit_info.ProcessMemoryLimit = new_limit_bytes
                limit_info.JobMemoryLimit = new_limit_bytes
                
                # 应用新的限制
                result = kernel32.SetInformationJobObject(
                    self.current_job_handle, 9,
                    ctypes.byref(limit_info),
                    ctypes.sizeof(limit_info)
                )
                
                if result:
                    # logging.info(f"✅ 已更新内存限制: {new_limit_gb:.1f}GB (PID: {self.current_kernel_pid})")
                    return True
                else:
                    error_code = kernel32.GetLastError()
                    logging.error(f"更新Job Object限制失败，错误代码: {error_code}")
                    return False
                    
            elif system == "Linux":
                # Linux: 更新resource限制
                try:
                    import resource
                    resource.prlimit(self.current_kernel_pid, resource.RLIMIT_AS, (new_limit_bytes, new_limit_bytes))
                    logging.info(f"✅ 已更新内存限制: {new_limit_gb:.1f}GB (PID: {self.current_kernel_pid})")
                    return True
                except Exception as e:
                    logging.error(f"更新Linux资源限制失败: {e}")
                    return False
            else:
                logging.warning(f"不支持的操作系统: {system}")
                return False
                
        except Exception as e:
            logging.error(f"更新内存限制失败: {e}")
            return False
    
    def calculate_dynamic_memory_limit(self, kernel_process=None):
        """
        计算动态内存限制
        
        Args:
            kernel_process: kernel进程对象（用于获取当前使用量）
        
        Returns:
            内存限制（GB）
        """
        memory_mode = self.kernel_memory_mode.get()
        
        if memory_mode == "固定值":
            return self.max_kernel_memory_gb.get()
        elif memory_mode == "动态百分比":
            # 获取系统可用内存
            mem = psutil.virtual_memory()
            system_available_gb = mem.available / (1024**3)
            
            # 获取kernel当前使用的内存
            kernel_usage_gb = 0.0
            if kernel_process:
                try:
                    kernel_usage_gb = kernel_process.memory_info().rss / (1024**3)
                except:
                    pass
            
            # 计算真实可用内存 = 系统可用 + kernel当前占用
            # （因为kernel的内存可以被重新分配给kernel自己）
            real_available_gb = system_available_gb + kernel_usage_gb
            
            percent = self.kernel_memory_percent.get()
            
            # 计算限制：真实可用内存 × 百分比
            limit_gb = real_available_gb * (percent / 100.0)
            
            # 设置最小值（至少1GB）
            limit_gb = max(1.0, limit_gb)
            
            return limit_gb
        else:
            return self.max_kernel_memory_gb.get()
    
    def monitor_kernel_process(self, kernel_pid):
        """
        监控并限制kernel进程的资源使用
        
        Args:
            kernel_pid: kernel进程PID
        """
        try:
            process = psutil.Process(kernel_pid)
            memory_mode = self.kernel_memory_mode.get()
            
            # 更新UI状态
            mode_display = "固定" if memory_mode == "固定值" else "动态"
            self.root.after(0, lambda: self.kernel_status_var.set(f"Kernel运行中 (PID: {kernel_pid}) | 模式: {mode_display}"))
            
            last_limit_gb = 0  # 记录上次的限制，避免重复更新
            
            while process.is_running():
                try:
                    mem_info = process.memory_info()
                    current_memory = mem_info.rss
                    current_memory_gb = current_memory / (1024**3)
                    
                    # 计算当前应该的内存限制（传入kernel进程用于计算真实可用内存）
                    target_limit_gb = self.calculate_dynamic_memory_limit(kernel_process=process)
                    
                    # 如果是动态模式且限制有变化，自动更新
                    if memory_mode == "动态百分比" and abs(target_limit_gb - last_limit_gb) > 0.1:
                        # 限制变化超过0.1GB才更新，避免频繁调整
                        success = self.update_kernel_memory_limit_realtime(target_limit_gb)
                        if success:
                            last_limit_gb = target_limit_gb
                            mem = psutil.virtual_memory()
                            system_available_gb = mem.available / (1024**3)
                            real_available_gb = system_available_gb + current_memory_gb
                            # logging.info(f"🔄 自动调整内存限制: {target_limit_gb:.1f}GB (真实可用: {real_available_gb:.1f}GB = 系统可用{system_available_gb:.1f}GB + kernel占用{current_memory_gb:.1f}GB, 占比: {self.kernel_memory_percent.get():.0f}%)")
                    
                    # 使用当前的限制
                    max_memory_gb = target_limit_gb
                    max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
                    
                    # 更新UI显示当前使用量
                    usage_pct = (current_memory_gb / max_memory_gb) * 100 if max_memory_gb > 0 else 0
                    
                    if memory_mode == "动态百分比":
                        mem = psutil.virtual_memory()
                        system_available_gb = mem.available / (1024**3)
                        real_available_gb = system_available_gb + current_memory_gb
                        status_text = (f"Kernel运行中 (PID: {kernel_pid}) | 模式: 动态 | "
                                     f"使用: {current_memory_gb:.2f}GB / {max_memory_gb:.1f}GB ({usage_pct:.1f}%) | "
                                     f"真实可用: {real_available_gb:.1f}GB (系统{system_available_gb:.1f}GB + kernel{current_memory_gb:.1f}GB)")
                    else:
                        status_text = (f"Kernel运行中 (PID: {kernel_pid}) | 模式: 固定 | "
                                     f"使用: {current_memory_gb:.2f}GB / {max_memory_gb:.1f}GB ({usage_pct:.1f}%)")
                    
                    self.root.after(0, lambda t=status_text: self.kernel_status_var.set(t))
                    
                    # 检查是否超过限制（这是备用检查，OS应该已经限制了）
                    if current_memory > max_memory_bytes:
                        logging.error(f"⚠️ Kernel进程({kernel_pid})内存 {current_memory_gb:.2f}GB 超过限制 {max_memory_gb:.1f}GB")
                        logging.warning(f"🛑 终止超限kernel进程...")
                        process.kill()
                        break
                    
                    # 动态模式下更频繁检查（0.5秒），固定模式1秒即可
                    sleep_time = 0.5 if memory_mode == "动态百分比" else 1.0
                    time.sleep(sleep_time)
                    
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logging.error(f"监控kernel进程出错: {e}")
                    break
            
            # kernel已停止，重置UI
            self.root.after(0, lambda: self.kernel_status_var.set("无运行中的kernel"))
            self.current_kernel_pid = None
            self.current_job_handle = None
                    
        except Exception as e:
            logging.error(f"无法监控kernel进程: {e}")
    
    
    def update_progress(self, current, total, message=""):
        """更新进度条"""
        if total > 0:
            progress = (current / total) * 100
            self.progress_var.set(progress)
            if message:
                self.progress_text_var.set(f"{message} ({current}/{total})")
            else:
                self.progress_text_var.set(f"进度: {current}/{total}")
        else:
            self.progress_var.set(0)
            self.progress_text_var.set("就绪")
    
    def run_selected_notebooks(self):
        """运行选中的notebooks"""
        # 如果有运行顺序列表，使用它；否则使用选中的文件
        if self.execution_order:
            selected_files = self.execution_order
        else:
            selected_files = self.get_selected_notebooks()
            
        if not selected_files:
            messagebox.showwarning("警告", "请先选择要运行的IPython Notebook文件")
            return
        
        # 初始化进度条
        self.progress_var.set(0)
        self.progress_text_var.set(f"准备运行 {len(selected_files)} 个文件...")
        
        # 在新线程中运行，避免界面卡顿
        thread = threading.Thread(target=self._run_notebooks_thread, args=(selected_files,))
        thread.daemon = True
        thread.start()
    
    def _run_notebooks_thread(self, notebook_files):
        """在新线程中运行notebooks"""
        try:
            self.status_var.set(f"正在运行 {len(notebook_files)} 个文件...")
            
            successful_runs = 0
            failed_runs = 0
            total_files = len(notebook_files)
            
            for index, notebook_path in enumerate(notebook_files, 1):
                # 进程隔离模式，直接执行
                logging.info(f"[{index}/{total_files}] 进程隔离模式")
                
                # 更新进度条 - 开始执行当前文件
                self.root.after(0, lambda idx=index-1, total=total_files: self.update_progress(idx, total, "正在执行"))
                
                result = self.run_notebook(notebook_path, index, total_files)
                if result == "success":
                    successful_runs += 1
                elif result == "oom":
                    failed_runs += 1
                    logging.warning(f"[{index}/{total_files}] 🔄 检测到内存错误，进行垃圾回收...")
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    
                    # OOM后等待让系统恢复（进程隔离模式下，超限的kernel进程已被终止）
                    time.sleep(5)
                    logging.info("✅ 继续执行下一个文件")
                else:
                    failed_runs += 1
                
                # 更新进度条 - 完成当前文件
                self.root.after(0, lambda idx=index, total=total_files: self.update_progress(idx, total, "已完成"))
            
            # 运行完成
            self.root.after(0, lambda: self.update_progress(total_files, total_files, "完成"))
            self.status_var.set(f"运行完成: 成功 {successful_runs}, 失败 {failed_runs}")
            logging.info(f"批量执行完成: {successful_runs} 成功, {failed_runs} 失败")
            
        except Exception as e:
            self.root.after(0, lambda: self.update_progress(0, 0, "运行出错"))
            self.status_var.set("运行过程中出现错误")
            logging.error(f"批量执行出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    def run_notebook(self, notebook_path, index, total):
        """
        执行指定的Jupyter notebook文件
        
        Args:
            notebook_path: notebook文件路径
            index: 当前执行的序号
            total: 总数量
            
        Returns:
            str: 执行状态 ('success', 'oom', 'failed')
        """
        start_time = time.time()
        logging.info(f"[{index}/{total}] 开始执行: {notebook_path}")
        
        try:
            # 读取notebook - 使用更兼容的方式
            with open(notebook_path, 'r', encoding='utf-8') as f:
                # 使用nbformat.read替代直接导入的read函数
                nb = nbformat.read(f, as_version=4)
            
            # 检测可用的内核
            kernel_name = 'python3'
            logging.info(f"使用内核: {kernel_name}")
            
            # 设置执行处理器
            ep = LoggingExecutePreprocessor(
                timeout=None,
                kernel_name=kernel_name,
                allow_errors=False,  # 遇到错误时停止执行
                run_index=index,
                run_total=total
            )
            
            # 初始化资源隔离相关变量
            kernel_process = None
            kernel_monitor_thread = None
            kernel_isolated = False
            
            # 覆盖preprocess_cell以在首次执行前应用资源隔离
            original_preprocess_cell = ep.preprocess_cell
            first_cell = [True]  # 使用列表以便在闭包中修改
            
            def preprocess_cell_with_isolation(cell, resources, cell_index):
                """在首个cell执行前应用资源隔离"""
                nonlocal kernel_process, kernel_monitor_thread, kernel_isolated
                
                # 首次执行cell时，kernel已经启动，应用资源隔离
                if first_cell[0] and not kernel_isolated:
                    first_cell[0] = False
                    try:
                        if hasattr(ep, 'km') and ep.km:
                            km = ep.km
                            # 尝试获取kernel进程PID
                            kernel_pid = None
                            
                            # 方法1: 从kernel manager获取
                            if hasattr(km, 'kernel') and km.kernel:
                                if hasattr(km.kernel, 'pid'):
                                    kernel_pid = km.kernel.pid
                                elif hasattr(km.kernel, 'pgid'):
                                    kernel_pid = km.kernel.pgid
                            
                            # 方法2: 通过进程名查找
                            if not kernel_pid:
                                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                                    try:
                                        if 'python' in proc.info['name'].lower():
                                            cmdline = proc.info.get('cmdline', [])
                                            if any('kernel' in str(arg).lower() for arg in cmdline):
                                                kernel_pid = proc.info['pid']
                                                break
                                    except:
                                        continue
                            
                            if kernel_pid:
                                logging.info(f"🔍 检测到kernel进程PID: {kernel_pid}")
                                
                                # 保存当前kernel PID
                                self.current_kernel_pid = kernel_pid
                                
                                # 记录初始内存限制（根据模式计算）
                                initial_limit = self.calculate_dynamic_memory_limit()
                                logging.info(f"📊 初始内存限制: {initial_limit:.1f}GB (模式: {self.kernel_memory_mode.get()})")
                                
                                # 应用资源限制
                                kernel_process = psutil.Process(kernel_pid)
                                self.apply_process_resource_limits(kernel_process)
                                
                                # 启动kernel进程监控线程
                                kernel_monitor_thread = threading.Thread(
                                    target=self.monitor_kernel_process, 
                                    args=(kernel_pid,), 
                                    daemon=True
                                )
                                kernel_monitor_thread.start()
                                logging.info(f"🛡️ 已启动kernel进程监控线程")
                                kernel_isolated = True
                            else:
                                logging.warning("⚠️ 无法获取kernel进程PID，跳过资源隔离")
                                
                    except Exception as e:
                        logging.warning(f"应用资源隔离失败: {e}，继续执行")
                
                # 调用原始的preprocess_cell
                return original_preprocess_cell(cell, resources, cell_index)
            
            # 替换方法
            ep.preprocess_cell = preprocess_cell_with_isolation
            
            # 执行notebook
            try:
                ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            except Exception as e:
                raise
            
            # 构建输出路径
            output_suffix = self.output_suffix.get()
            base_name = os.path.splitext(notebook_path)[0]
            output_path = f"{base_name}{output_suffix}.ipynb"
            
            # 保存执行后的notebook - 使用更兼容的方式
            with open(output_path, 'w', encoding='utf-8') as f:
                # 使用nbformat.write替代直接导入的write函数
                nbformat.write(nb, f)
                
            execution_time = time.time() - start_time
            logging.info(f"[{index}/{total}] ✅ 成功执行: {notebook_path} -> {output_path} (耗时: {execution_time:.2f}秒)")
            return "success"
            
        except MemoryError as e:
            execution_time = time.time() - start_time
            logging.error(f"[{index}/{total}] 内存溢出(OOM): {notebook_path} (耗时: {execution_time:.2f}秒)")
            logging.error(f"MemoryError详情: {str(e)}")
            return "oom"
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e).lower()
            
            # 检测是否为内存保护中断或OOM相关错误
            oom_keywords = ['memory', 'oom', 'out of memory', 'memoryerror', 'cannot allocate', '内存保护', 'emergency']
            is_oom = any(keyword in error_msg for keyword in oom_keywords)
            
            if is_oom or '内存保护机制中断' in str(e):
                logging.error(f"[{index}/{total}] 💥 内存相关错误: {notebook_path} (耗时: {execution_time:.2f}秒)")
                logging.error(f"错误信息: {str(e)}")
                # 强制垃圾回收
                import gc
                gc.collect()
                return "oom"
            else:
                logging.error(f"[{index}/{total}] ❌ 执行出错: {notebook_path} (耗时: {execution_time:.2f}秒): {str(e)}")
            
            # 输出详细异常信息
            import traceback
            logging.error(traceback.format_exc())
            return "failed"

def main():
    """主函数"""
    app = NotebookManager()
    app.root.mainloop()

if __name__ == "__main__":
    main()