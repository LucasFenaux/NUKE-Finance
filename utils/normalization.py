import numpy as np
import torch
import torch.multiprocessing as mp
import queue
import time


def normalization_worker(x, q: mp.Queue):
    normalized_x = normalization(x, use_multiprocessing=False)
    q.put(normalized_x)
    return


def multiprocess_normalization(x, num_workers: int = 16):
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_workers)

    p = int(x.size(0)/num_workers)
    c = 0
    jobs = []

    for i in range(num_workers - 1):
        job = pool.apply_async(normalization_worker, (x[c:c+p], q))
        jobs.append(job)
        c += p

    job = pool.apply_async(normalization_worker, (x[c:], q))
    jobs.append(job)

    for job in jobs:
        job.get()

    data = []
    for i in range(num_workers):
        try:
            d = q.get(timeout=10)

            data.append(d)
        except queue.Empty:
            print("Getting from queue timed-out")
            return None

    pool.close()
    pool.join()

    return torch.cat(data, dim=0)


def normalization_trader(x):
    if isinstance(x, torch.Tensor):
        size = x.size()
    elif isinstance(x, np.ndarray):
        size = x.shape
    else:
        size = x.size()
    min_values = torch.zeros((size[0], size[2]))
    max_values = torch.zeros((size[0], size[2]))

    normalized_x = []
    for i in range(size[0]):
        row = []
        for j in range(size[2]):
            min_val = x[i, :, j].min()
            min_values[i, j] = min_val

            shifted_x = x[i, :, j] - min_val + 1e-6
            max_val = shifted_x.max()

            max_values[i, j] = max_val

            shifted_x = shifted_x / max_val
            row.append(shifted_x.unsqueeze(1))
        normalized_x.append(torch.cat(row, dim=1).unsqueeze(0))
    return torch.cat(normalized_x, dim=0), min_values, max_values


def parallelized_normalization_trader(x):
    positive = (x >= 0).to(torch.int)
    negative = (positive == 0).to(torch.int)
    min_values = torch.min(x*positive, dim=1, keepdim=True)[0].expand(-1, x.size()[1], -1)
    shifted_x = x*positive - min_values + 1e-6
    max_values = torch.max(shifted_x*positive, dim=1, keepdim=True)[0].expand(-1, x.size()[1], -1)
    normalized_x = torch.div(shifted_x, max_values)
    negative = negative.to(torch.bool)
    normalized_x[negative] = -1.
    # normalized_x = torch.ones_like(normalized_x)*negative*-1.

    return normalized_x, min_values, max_values


def parallelized_undo_normalization_trader(normalized_x, min_values, max_values):
    positive = (normalized_x >= 0).to(torch.int)
    negative = (positive == 0).to(torch.bool)
    x = torch.mul(normalized_x*positive, max_values)
    x = x - min_values*positive - 1e-6
    x[negative] = -1.
    return x


def undo_normalization_trader(normalized_x, min_values, max_values):
    if isinstance(normalized_x, torch.Tensor):
        size = normalized_x.size()
    elif isinstance(normalized_x, np.ndarray):
        size = normalized_x.shape
    else:
        size = normalized_x.size()

    x = []
    for i in range(size[0]):
        row = []
        for j in range(size[2]):
            min_val = min_values[i, j]
            max_val = max_values[i, j]

            mul_x = normalized_x[i, :, j] * max_val
            unshifted_x = mul_x + min_val - 1e-6

            row.append(unshifted_x.unsqueeze(1))
        x.append(torch.cat(row, dim=1).unsqueeze(0))
    return torch.cat(x, dim=0)


def normalization(x, use_multiprocessing: bool = False, num_workers: int = 16):
    if use_multiprocessing:
        # using multiprocessing does not preserve order
        return multiprocess_normalization(x, num_workers)
    else:
        if isinstance(x, torch.Tensor):
            size = x.size()
        elif isinstance(x, np.ndarray):
            size = x.shape
        else:
            size = x.size()
        normalized_x = []
        for i in range(size[0]):
            row = []
            for j in range(size[2]):
                # we ignore the future value when normalizing to avoid leaking future information to the model
                min_val = x[i, :-1, j].min()
                # min_val = x[i, :, j].min()

                shifted_x = x[i, :, j] - min_val + 1e-6
                max_val = shifted_x[:-1].max()
                # max_val = shifted_x.max()
                # we clamp to prevent a too big a blow up in the values while keeping the idea that the last value is
                # bigger
                shifted_x = shifted_x/max_val
                shifted_x = torch.clamp(shifted_x, min=-1.0, max=2.0)
                row.append(shifted_x.unsqueeze(1))
            normalized_x.append(torch.cat(row, dim=1).unsqueeze(0))
        return torch.cat(normalized_x, dim=0)


if __name__ == '__main__':
    # benchmarking
    device = torch.device("cuda")
    x = torch.rand(500, 100, 5).to(device)
    start_time = time.time()
    n_x, n_min, n_max = normalization_trader(x)
    end_time = time.time()
    print(f"Normalization time {end_time - start_time}")
    start_time = time.time()
    pn_x, pn_min, pn_max = parallelized_normalization_trader(x)
    end_time = time.time()
    print(f"Normalization time {end_time - start_time}")
    n_min, n_max = n_min.to(device), n_max.to(device)
    print(torch.mean(torch.abs(n_x - pn_x)))

    undo_x = parallelized_undo_normalization_trader(pn_x, pn_min, pn_max)
    print(torch.mean(torch.abs(undo_x - x)))
    undo_x = undo_normalization_trader(n_x, n_min, n_max)
    print(torch.mean(torch.abs(undo_x - x)))
    undo_x = undo_normalization_trader(pn_x, pn_min[:, 0, :].squeeze(), pn_max[:, 0, :].squeeze())
    print(torch.mean(torch.abs(undo_x - x)))
