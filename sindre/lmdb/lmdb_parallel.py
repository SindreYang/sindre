import os
import time
import traceback
from typing import List, Callable
import multiprocessing as mp
from tqdm import tqdm

from sindre.lmdb import Writer

__all__ = [ "LMDBParallel_MultiDB", "LMDBParallel_Batch"]
class LMDBParallel_MultiDB:
    """
    LMDB多进程写入器,写入到多个数据库

    特点：
    - 每个进程写入独立LMDB数据库
    - 支持多进程处理和批量写入
    Notes:
        def process(json_file):
           with open(json_file, "r") as f:
               data = json.loads(f.read())
               id = data["id_patient"]
               jaw = data["jaw"]
               labels = data["labels"]
               mesh = vedo.load(json_file.replace(".json", ".obj"))
               vertices = mesh.vertices
               faces = mesh.cells

               out = {
                   'mesh_faces': faces,
                   'mesh_vertices': vertices,
                   'vertex_labels': labels,
                   "jaw": jaw,
               }
               return out

       # 使用类的方式
       dirpath = "./output_lmdb"
       map_size_limit = 1024
       json_file_list = glob.glob("./*/*/*.json")[:16]
       writer = LMDBParallelWriter(
            output_dir=dirpath,
            map_size_limit=map_size_limit,
            num_processes=8,
            temp_root="./processing_temp",
        )
        writer.write(json_file_list, process)
    
    
    """

    def __init__(self,
                 output_dir: str,
                 map_size_limit: int,
                 num_processes: int = 4,
                 multiprocessing: bool = True,
                 ):
        """
        初始化LMDB并行写入器

        Args:
            output_dir: 最终输出LMDB路径
            map_size_limit: 总LMDB的map大小限制(MB)
            num_processes: 进程数量
            multiprocessing: 是否启用多进程模式
        """
        self.output_dir = output_dir
        self.map_size_limit = map_size_limit
        self.num_processes = num_processes
        self.multiprocessing = multiprocessing

        # 进程管理
        self.processes = []
        self.manager = None

        # 数据库列表
        os.makedirs(self.output_dir, exist_ok=True)
        self.temp_dirs = [
            os.path.join(self.output_dir, f"process_{i}.db")
            for i in range(self.num_processes)
        ]

    def write(self, args_list: list, process: Callable):
        """
        执行多进程写入

        Args:
            args_list: 文件路径列表
            process: 数据处理函数
        """

        try:
            # 初始化进程管理器
            self.manager = mp.Manager()
            progress_queue = self.manager.Queue()

            # 启动工作进程
            self._start_worker_processes(args_list, process, progress_queue)

            # 监控进度
            self._monitor_progress(args_list, progress_queue)

        except Exception as e:
            print(f"处理失败: {str(e)}")
            traceback.print_exc()

        finally:
            # 清理资源
            self._cleanup_resources()


    def _start_worker_processes(self, args_list: list, process: Callable, progress_queue):
        """启动工作进程"""
        for i in range(self.num_processes):
            p = mp.Process(
                target=self._worker_write,
                args=(
                    self.temp_dirs[i],
                    args_list,
                    process,
                    self.map_size_limit // self.num_processes,
                    self.multiprocessing,
                    i,
                    self.num_processes,
                    progress_queue
                )
            )
            self.processes.append(p)
            p.start()

    def _monitor_progress(self, args_list: list, progress_queue):
        """监控处理进度"""
        with tqdm(total=len(args_list), desc="多进程处理", unit="file") as main_pbar:
            while any(p.is_alive() for p in self.processes):
                while not progress_queue.empty():
                    main_pbar.update(progress_queue.get())
                time.sleep(0.1)

    def _cleanup_resources(self):
        """清理进程和临时文件"""
        # 等待进程结束
        for p in self.processes:
            p.join()

        # 重置状态
        self.processes.clear()
        if self.manager:
            self.manager.shutdown()
            self.manager = None

    @staticmethod
    def _worker_write(temp_dir: str,
                      args_list: list,
                      process: callable,
                      map_size_limit: int,
                      multiprocessing: bool,
                      process_id: int,
                      num_processes: int,
                      progress_queue):
        """
        子进程处理函数

        Args:
            temp_dir: 临时LMDB目录
            args_list: 文件列表/参数列表
            process: 数据处理函数
            map_size_limit: 单个LMDB的map大小限制(MB)
            multiprocessing: 是否启用多进程模式
            process_id: 进程ID
            num_processes: 总进程数
            progress_queue: 进度队列
        """
        # 创建Writer实例
        writer = Writer(temp_dir, map_size_limit=map_size_limit, multiprocessing=multiprocessing)

        # 带错误处理的处理流程
        processed_count = 0
        for idx, args in enumerate(args_list):
            # 分配任务给当前进程
            if idx % num_processes != process_id:
                continue

            try:
                # 执行数据处理
                out = process(args)

                if out:
                    # 写入数据库
                    writer.put_sample(out)
                else:
                    print(f"函数返回值异常: {out}")
                processed_count += 1

                # 每处理10个文件报告一次进度
                if processed_count % 10 == 0:
                    progress_queue.put(10)

            except Exception as e:
                print(f"\n处理失败: {args}")
                print(f"错误信息: {str(e)}")
                traceback.print_exc()
                continue

        # 报告剩余进度
        if processed_count % 10 != 0:
            progress_queue.put(processed_count % 10)

        writer.close()

    def __del__(self):
        """析构函数，确保资源清理"""
        if self.processes:
            self._cleanup_resources()





class LMDBParallel_Batch:
    """LMDB多进程并行处理器,批量提交"""
    def __init__(self,
                 output_dir: str,
                 map_size_limit: int,
                 num_processes: int = 8,
                 batch_size: int = 256,
                 multiprocessing: bool = True):
        """
        初始化并行处理器

        Args:
            output_dir: LMDB输出目录
            map_size_limit: LMDB map大小限制(MB)
            num_processes: 进程数量
            batch_size: 批量提交大小
            multiprocessing: 是否启用多进程
        """
        self.output_dir = output_dir
        self.map_size_limit = map_size_limit
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.multiprocessing = multiprocessing

        # 进程管理
        self.manager = None
        self.processes = []
        self.write_process = None

    def process_files(self, args_list: list, process_func: Callable):
        """
        处理文件列表并写入LMDB

        Args:
            args_list: 文件路径列表/参数列表
            process_func: 数据处理函数
        """

        # 初始化LMDB写入器
        self.writer = Writer(
            self.output_dir,
            self.map_size_limit,
            self.multiprocessing
        )

        try:
            # 初始化进程管理器
            self.manager = mp.Manager()
            data_queue = self.manager.Queue(maxsize=self.num_processes * 2)
            progress_queue = self.manager.Queue()
            error_queue = self.manager.Queue()

            # 启动写入进程
            self.write_process = mp.Process(
                target=self._batch_writer_worker,
                args=(data_queue, progress_queue)
            )
            self.write_process.start()

            # 启动处理进程
            for i in range(self.num_processes):
                p = mp.Process(
                    target=self._processor_worker,
                    args=(args_list, process_func, i, data_queue, progress_queue, error_queue)
                )
                self.processes.append(p)
                p.start()

            # 显示进度
            self._monitor_progress(args_list, progress_queue, error_queue)

            # 发送结束信号
            data_queue.put(None)

        except Exception as e:
            print(f"\n处理失败: {str(e)}")
            traceback.print_exc()
        finally:
            self._cleanup()

    def _processor_worker(self,
                          args_list: list,
                          process_func: Callable,
                          process_id: int,
                          data_queue: mp.Queue,
                          progress_queue: mp.Queue,
                          error_queue: mp.Queue):
        """数据处理工作进程"""
        batch_data = []

        for idx, args in enumerate(args_list):
            # 任务分配
            if idx % self.num_processes != process_id:
                continue

            try:
                # 处理单个文件
                result = process_func(args)

                if result:
                    batch_data.append(result)

                    # 批量满了就发送
                    if len(batch_data) >= self.batch_size:
                        data_queue.put(batch_data)
                        progress_queue.put(len(batch_data))
                        batch_data = []

            except Exception as e:
                error_msg = f"参数{args_list} 处理失败: {str(e)}"
                error_queue.put(error_msg)
                traceback.print_exc()

        # 发送剩余数据
        if batch_data:
            data_queue.put(batch_data)
            progress_queue.put(len(batch_data))

    def _batch_writer_worker(self, data_queue: mp.Queue, progress_queue: mp.Queue):
        """批量写入工作进程"""
        while True:
            try:
                batch = data_queue.get()

                # 结束信号
                if batch is None:
                    break

                if batch:
                    self.writer.put_batch_sample(batch)

            except Exception as e:
                print(f"\n批量写入失败: {str(e)}")
                traceback.print_exc()

    def _monitor_progress(self,
                          file_list: List[str],
                          progress_queue: mp.Queue,
                          error_queue: mp.Queue):
        """监控处理进度"""
        with tqdm(total=len(file_list), desc="多进程处理", unit="file") as pbar:
            processed = 0

            while processed < len(file_list):
                # 检查错误
                while not error_queue.empty():
                    error_info = error_queue.get()
                    print(f"\n错误: {error_info}")

                # 更新进度
                while not progress_queue.empty():
                    count = progress_queue.get()
                    pbar.update(count)
                    processed += count

                # 检查进程状态
                all_processes_alive = any(p.is_alive() for p in self.processes)
                write_process_alive = self.write_process.is_alive() if self.write_process else False

                if not all_processes_alive and not write_process_alive and progress_queue.empty():
                    break

                time.sleep(0.1)

    def _cleanup(self):
        """清理资源"""
        # 等待进程结束
        if self.write_process:
            self.write_process.join()

        for p in self.processes:
            p.join()

        # 关闭LMDB写入器
        if hasattr(self, 'writer'):
            self.writer.close()

        # 清理进程列表
        self.processes.clear()
        self.write_process = None

        if self.manager:
            self.manager.shutdown()

    def __del__(self):
        """析构函数，确保资源被释放"""
        self._cleanup()
