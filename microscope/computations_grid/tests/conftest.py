import pytest
import torch


@pytest.fixture(params=["cuda:0", "cpu"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param