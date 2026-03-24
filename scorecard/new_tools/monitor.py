import psutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
import threading
import warnings
warnings.filterwarnings("ignore")

class SystemMonitor:
    def __init__(self, update_interval=1.0):
        self.monitoring = False
        self.start_time = None
        self.update_interval = update_interval  # 监控更新间隔（秒）
        self.monitor_thread = None
        self.data = {
            'timestamp': [],        # 存储实际时间戳
            'elapsed_sec': [],      # 存储从开始监控经过的秒数
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': [],
            'memory_available_gb': []
        }
    
    def start_monitoring(self):
        """开始监控系统资源"""
        if not self.monitoring:
            print(f"监控启动于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"监控间隔: {self.update_interval} 秒")
            self.monitoring = True
            self.start_time = time.time()
            self.data = {k: [] for k in self.data}  # 清空数据
            
            # 启动后台监控线程
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_loop(self):
        """后台监控循环"""
        while self.monitoring:
            try:
                self.update()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"监控过程中发生错误: {e}")
                break
    
    def update(self):
        """更新监控数据"""
        if self.monitoring:
            # 获取系统数据
            mem = psutil.virtual_memory()
            current_time = datetime.now()
            
            # 记录数据
            self.data['timestamp'].append(current_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
            self.data['elapsed_sec'].append(time.time() - self.start_time)
            self.data['cpu_percent'].append(psutil.cpu_percent(interval=0.1))
            self.data['memory_percent'].append(mem.percent)
            self.data['memory_used_gb'].append(mem.used / (1024**3))
            self.data['memory_available_gb'].append(mem.available / (1024**3))
    
    def stop_monitoring(self):
        """停止监控"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)  # 等待监控线程结束
            
            duration = time.time() - self.start_time
            print(f"监控结束于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"总监控时长: {duration:.2f} 秒")
            print(f"收集了 {len(self.data['timestamp'])} 个数据点")
    
    def get_current_stats(self):
        """获取当前系统状态（用于实时显示）"""
        if not self.monitoring:
            return None
        
        mem = psutil.virtual_memory()
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': mem.percent,
            'memory_used_gb': mem.used / (1024**3),
            'memory_available_gb': mem.available / (1024**3),
            'elapsed_sec': time.time() - self.start_time if self.start_time else 0
        }
    
    def save_results(self, prefix="monitor", save_path = None):
        """保存监控结果到CSV和图片
        
        Args:
            prefix: 文件名前缀
            csv_path: 自定义CSV文件路径，如果为None则自动生成
            png_path: 自定义PNG文件路径，如果为None则自动生成
        """
        if not self.data['timestamp']:
            print("没有可保存的数据")
            return
        
        if save_path is None:
            os.makedirs("monitor_results", exist_ok=True)
            save_path = "monitor_results/"
        else:
            os.makedirs(save_path, exist_ok=True)

        csv_path = save_path + f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        png_path = save_path + f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # 保存为CSV
        df = pd.DataFrame(self.data)
        df.to_csv(csv_path, index=False)
        print(f"数据已保存到 {csv_path}")
        
        # 生成并保存图表
        self.plot_results(save_path=png_path)
    
    def plot_results(self, save_path=None):
        """绘制监控结果图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 将时间字符串转换为datetime对象用于绘图
        time_stamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in self.data['timestamp']]
        
        # CPU图表
        ax1.plot(time_stamps, self.data['cpu_percent'], 'r-')
        ax1.set_title('CPU Usage (%)')
        ax1.set_ylabel('Percent')
        ax1.grid(True)
        
        # 格式化x轴时间显示
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 内存图表
        ax2.plot(time_stamps, self.data['memory_percent'], 'b-', label='Memory Usage %')
        ax2.set_title('Memory Usage')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Percent (%)')
        ax2.grid(True)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 添加内存使用量(GB)的第二个y轴
        ax2_mem = ax2.twinx()
        ax2_mem.plot(time_stamps, self.data['memory_used_gb'], 'g--', label='Used Memory')
        ax2_mem.plot(time_stamps, self.data['memory_available_gb'], 'c--', label='Available Memory')

        ax2_mem.set_ylabel('Memory (GB)')
        
        # 为两个y轴都添加图例
        ax2.legend(loc='upper left')
        ax2_mem.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"图表已保存到 {save_path}")
        else:
            plt.show()

# 使用示例保持不变
def your_core_operation():
    """这是你要监控的核心操作"""
    print("核心操作开始...")
    
    # 模拟一个耗时操作
    results = []
    for i in range(60):
        # 模拟CPU密集型操作
        start = time.time()
        while time.time() - start < 0.5:
            _ = 123456 * 654321
        
        # 模拟内存分配
        results.append(np.random.rand(1000000))  # 分配约8MB内存
        
        print(f"完成阶段 {i+1}/60")
        time.sleep(1)
    
    print("核心操作完成")
    return results

if __name__ == "__main__":
    # 创建监控器实例，设置监控间隔为0.5秒
    monitor = SystemMonitor(update_interval=0.5)
    
    try:
        # 阶段1: 开始监控
        monitor.start_monitoring()
        
        # 阶段2: 运行核心代码
        # 现在监控器会在后台持续收集数据
        core_result = your_core_operation()
        
        # 阶段3: 结束监控
        monitor.stop_monitoring()
        
        # 阶段4: 保存结果
        monitor_path = '/home/zhuchen/testing/monitor_results/'
        monitor.save_results("core_operation", save_path = monitor_path)
        
    except Exception as e:
        monitor.stop_monitoring()
        print(f"发生错误: {str(e)}")
        monitor.save_results("error_case")
        raise