import torch
import time

class ResourceMonitor:

    def __init__(self):
        return None

    def print_gpu_info(self):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('__CUDNN VERSION:', torch.backends.cudnn.version())
            print('__Number CUDA Devices:', torch.cuda.device_count())
            print('__CUDA Device Name:', torch.cuda.get_device_name(0))
            print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
        else:
            print('__CUDA is not available')

    def cuda_start_timer(self):
        self.start = time.time()

    def cuda_stop_timer(self):
        torch.cuda.synchronize()  # <---------------- extra line
        end = time.time()
        print("Run time [s]: ", end - self.start)






