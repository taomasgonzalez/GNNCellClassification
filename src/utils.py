import numpy as np
import os
import random
import torch


def set_seed(seed, cuda_deterministic=True) -> None:
    """
    seed: The seed value for Python, NumPy, and PyTorch to make experiments reproducible.
    cuda_deterministic: set cuDNN / CUDA to run in deterministic mode. This has a performance impact.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ``warn_only`` avoids a hard crash if deterministic ops are not
        # implemented for some layers.
        # torch.use_deterministic_algorithms(True, warn_only=True)
        torch.use_deterministic_algorithms(True)

        # For deterministic cuBLAS (needed for some GPUs / CUDA versions)
        # must be set before the first CUDA kernel is launched.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
