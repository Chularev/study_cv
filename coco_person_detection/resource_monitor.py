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

    def memory_convert(self,memory, flag):
        if flag == 'MB':
            print("{:.2f} MB".format(memory / 1024 ** 2))
            return
        if flag == 'GB':
            print("{:.2f} GB".format(memory / 1024 ** 3))
            return
        print("{:.2f} B".format(torch.cuda.memory_allocated()))

    def get_cuda_memory_allocated(self, flag):
        memory = torch.cuda.memory_allocated()
        self.memory_convert(memory,flag)


    def get_cuda_cached_memory(self, flag):
        memory = torch.cuda.memory_reserved()
        self.memory_convert(memory, flag)