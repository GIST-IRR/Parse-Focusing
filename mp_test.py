from multiprocessing import Pool, Manager
import multiprocessing as mp
from time import sleep

def manager(n, e):
    while e.is_set():
        e.wait(1)
    e.set()
    c_proc = mp.current_process()
    print(f'Running on Process {c_proc.name} PID {c_proc.pid} Q {n}')
    sleep(5)
    print(f'Ended Process {c_proc.name}. Return Q {n}')
    e.clear()

if __name__ == '__main__':
    n_proc = 4
    n_models = 10
    events = [Manager().Event() for _ in range(n_proc)]

    pool = Pool(processes=n_proc)
    for i in range(n_models):
        n = i%n_proc
        e = events[n]
        print(f'Process {i} start')
        res = pool.apply_async(manager, args=(n, e,))
        sleep(1)
        # while all([e.is_set() for e in events]):
        #     pass
    pool.close()
    pool.join()

    # event = mp.Event()
    # p1 = mp.Process(target=manager, args=(0, event))
    # p2 = mp.Process(target=manager, args=(1, event))
    # p1.start()
    # p2.start()

    # p1.join()
    # p2.join()

    # p1.close()
    # p2.close()