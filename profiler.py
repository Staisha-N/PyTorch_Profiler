import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

class SlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 62 * 62, 10)

    def forward(self, x):
        for _ in range(100):
            x = x + 1
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def run_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SlowModel().to(device)
    input = torch.randn(8, 3, 64, 64).to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("log"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(6):  # must cover wait + warmup + active steps
            with record_function("model_inference"):
                model(input)
            prof.step()

if __name__ == "__main__":
    run_model()
