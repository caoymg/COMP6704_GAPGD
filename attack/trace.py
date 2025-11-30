import torch

def hutchinson_trace(delta, grad, num_samples=1):
    trace_estimates = []
    for _ in range(num_samples):
        v = torch.randint_like(delta, 0, 2).float()
        v = 2 * v - 1
        Hv = torch.autograd.grad(
            outputs=(grad * v).sum(),
            inputs=delta,
            retain_graph=True
        )[0]
        trace_estimates.append((Hv * v).sum().item())
    return sum(trace_estimates) / len(trace_estimates)
