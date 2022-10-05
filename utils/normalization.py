import torch
import torch.multiprocessing as mp
import queue


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


def normalization(x, use_multiprocessing: bool = False, num_workers: int = 16):
    if use_multiprocessing:
        # using multiprocessing does not preserve order
        return multiprocess_normalization(x, num_workers)
    else:
        normalized_x = []
        for i in range(x.size()[0]):
            row = []
            for j in range(x.size()[2]):
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
