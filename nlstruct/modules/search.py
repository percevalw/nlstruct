import numpy as np
import torch

try:
    import faiss
    from faiss import cast_integer_to_float_ptr as cast_float
    from faiss import cast_integer_to_long_ptr as cast_long
except ImportError:
    cast_float = cast_long = faiss = None

from nlstruct.utils import torch_global as tg


def ptr(tensor):
    if torch.is_tensor(tensor):
        return tensor.storage().data_ptr()
    elif hasattr(tensor, 'data'):
        return tensor.data.storage().data_ptr()
    else:
        return tensor


def ensure_device(tensor, device):
    device = torch.device(device)
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif type(tensor) is np.ndarray:
        return torch.as_tensor(tensor, device=device)
    else:
        raise Exception()


class FAISSIndex(object):
    SGR = None

    def __init__(self, dim=100, use_bias=False, factory_str="Flat", metric=faiss.METRIC_INNER_PRODUCT,
                 nprobe=32, device=None):
        super(FAISSIndex, self).__init__()
        assert faiss is not None, "Could not import faiss, make sure it is installed"

        if device is None:
            device = tg.device
        self.dim = dim
        self.use_bias = use_bias
        if self.use_bias:
            self.dim += 1
        self.device = torch.device("cpu")
        self.index = faiss.index_factory(self.dim, factory_str, metric)
        self.positions = None
        self.to(device)
        self.index.nprobe = nprobe

    def __del__(self):
        self.reset()

    def to(self, device):
        if self.device == device:
            return self
        elif device.type == "cpu" and self.device.type == "cuda":
            self.index = faiss.index_gpu_to_cpu(self.index)
        elif device.type == "cuda" and self.device.type == "cpu":
            FAISSIndex.SGR = FAISSIndex.SGR if FAISSIndex.SGR is not None else faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(FAISSIndex.SGR, device.index or 0, self.index)
        elif device.type == "cuda" and self.device.type == "cuda":
            index = faiss.index_gpu_to_cpu(self.index)
            FAISSIndex.SGR = FAISSIndex.SGR if FAISSIndex.SGR is not None else faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(FAISSIndex.SGR, device.index or 0, index)
        self.device = device

    def train(self, weights, bias=None, and_add=True, positions=None):
        if bias is None:
            weights = ensure_device(weights, "cpu")
        else:
            bias = ensure_device(bias, "cpu")
            weights = ensure_device(weights, "cpu")
            weights = torch.cat((weights, bias.view(-1, 1)), 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.index.train(weights.numpy())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if and_add:
            self.add(weights, positions=positions)

    def reset(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.index.reset()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def add(self, other, positions=None, last=None):
        other = ensure_device(other, self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        other = other[:last, :] if last is not None else other
        self.index.add_c(other.size(0), cast_float(ptr(other)))
        if positions is not None:
            if self.positions is None:
                self.positions = ensure_device(positions, self.device)
            else:
                self.positions = torch.cat([self.positions, ensure_device(positions, self.device)])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def search(self, query, k=1, copy=False):
        if copy:
            query = query.clone()

        query = ensure_device(query, self.device)
        if self.use_bias:
            query = torch.cat((query, torch.ones(len(query), 1, device=self.device)), 1)

        b, n = query.shape

        distances = torch.zeros(b, k, device=self.device, dtype=torch.float)
        labels = torch.zeros(b, k, device=self.device, dtype=torch.long)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.index.search_c(
            b,
            cast_float(ptr(query)),
            k,
            cast_float(ptr(distances)),
            cast_long(ptr(labels))
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return distances, labels if self.positions is None else self.positions[labels]
