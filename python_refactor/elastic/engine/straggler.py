import time
import os
import numpy as np
import hetu as ht
from .utils import Args

WORKLOAD_ITERS = 20

class WorkloadInfo(Args):
    def __init__(
        self,
        mbs: int,
        seq_len: int,
        hidden_size: int,
    ):
        self.mbs = mbs
        self.seq_len = seq_len
        self.hidden_size = hidden_size

class Straggler(Args):
    def __init__(
        self,
        local_device: ht.device, 
        log_file: str,
        workload_info: WorkloadInfo
    ):
        self.local_device = local_device
        self.log_file = log_file
        self.workload_info = workload_info
     
    @staticmethod
    def read_profile(log_file: str, length: int, ignore_first: bool = True):
        # polling
        while True:
            with open(log_file, "r") as file:
                content = file.readlines()
                if len(content) == 0 or content[-1] != "EOF":
                    continue
                # get rid of the first entry
                # because of cold start
                start_idx = 0
                if ignore_first:
                    start_idx = 1
                straggler_info = [float(line.strip()) for line in content[start_idx:-1]]
                assert len(straggler_info) == length, "file length wrong"
                return straggler_info
                    
    def begin_profile(self):
        assert self.log_file != None and self.log_file != "", "should set log file path in advance"
        os.makedirs(os.path.basename(os.path.dirname(self.log_file)), exist_ok=True)
        with open(self.log_file, "w") as file:
            file.truncate(0)
            file.flush()
            os.fsync(file.fileno())
        os.environ['HETU_STRAGGLER_LOG_FILE'] = self.log_file
        
    def end_profile(self):
        assert os.path.exists(self.log_file) and "HETU_STRAGGLER_LOG_FILE" in os.environ, \
            "the log file and corresponding env is not existed" 
        with open(self.log_file, "a") as file:
            file.write("EOF")
            file.flush()
            os.fsync(file.fileno())
        del os.environ["HETU_STRAGGLER_LOG_FILE"]
        
    def run_profile(self):
        with ht.graph("eager", create_new=True, prefix="tmp", tmp=True):
            with ht.context(eager_device=self.local_device):
                time_elapse = self.run_workload()
        with open(self.log_file, "a") as file:
            file.write(f"{time_elapse}\n")
            file.flush()
            os.fsync(file.fileno())
        
    def run_workload(self):
        shape_x = [self.workload_info.mbs * self.workload_info.seq_len, self.workload_info.hidden_size]
        shape_y = [self.workload_info.hidden_size, self.workload_info.hidden_size * 4]
        shape_z = [self.workload_info.hidden_size * 4, self.workload_info.hidden_size]
        x_np = np.random.randn(*shape_x).astype(np.float32)
        y_np = np.random.randn(*shape_y).astype(np.float32)
        z_np = np.random.randn(*shape_z).astype(np.float32)
        x = ht.from_numpy(x_np)
        y = ht.from_numpy(y_np)
        z = ht.from_numpy(z_np)
        x = ht.matmul(ht.matmul(x, y), z)
        x_np = x.numpy(force=True)
        start_time = time.time()
        for i in range(WORKLOAD_ITERS):
            x = ht.matmul(ht.matmul(x, y), z)
        x_np = x.numpy(force=True)
        end_time = time.time()
        return end_time - start_time
        
            