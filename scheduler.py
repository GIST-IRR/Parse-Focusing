import argparse
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
import time
from queue import Queue
import tempfile
import traceback

from easydict import EasyDict

from torch_support.train_support import (
    setup_log_dir,
    get_config_from,
    save_config_to
)
from train import train


def device_wrapper(i, device, func, *args, **kwargs):
    kwargs.update({"device": device})
    res = func(*args, **kwargs)
    return i, device, res

def run_multi_gpu(
        n_device, n_models, func, prog,
        args_list=None, kwargs_list=None,
        process_gap=5,
    ):
    if args_list is not None:
        assert n_models == len(args_list), \
            'Number of models must be equal to number of arguments.'
    if kwargs_list is not None:
        assert n_models == len(kwargs_list), \
            'Number of models must be equal to number of keyword arguments.'

    # GPU job queue
    q = Queue()
    for i in range(n_device):
        q.put(i)

    results = []

    pool = Pool(processes=n_device)
    for i in range(n_models):
        # If the job is already done, skip it.
        if prog[i]:
            continue

        try:
            device = q.get()
            args = args_list[i] if args_list is not None else []
            kwargs = kwargs_list[i] if kwargs_list is not None else {}

            res = pool.apply_async(
                device_wrapper,
                args=(i, device, func, *args),
                kwds=kwargs
            )

            results.append(res)
        except:
            traceback.print_exc()
            print(f'Error occurs in model {i}.')
            q.put(device)

        time.sleep(process_gap)

        while q.empty():
            for res in results:
                if res.ready():
                    i, device, _ = res.get()
                    prog[i] = True
                    q.put(device)
                    results.remove(res)
        
    pool.close()
    pool.join()

def read_schedule(schedule_path):
    configuration = get_config_from(schedule_path, easydict=False)
    schedules = configuration['schedules']

    global_option = configuration.get('option', 'standard')

    config_set = []
    for schedule in schedules:
        base_cfg_name = schedule.get('base', None)
        if base_cfg_name is not None:
            base_conf = get_config_from(schedule['base'], easydict=False)
        else:
            base_conf = {}

        local_option = schedule.get('option', global_option)

        # Check whether number of configurations is specified or not.
        n_conf = schedule.get('number', None)
        if n_conf is None:
            variables = schedule.get('variables', None)
            if variables is not None:
                n_conf = len(list(variables.values())[0])
            else:
                raise ValueError('Number of configurations must be specified.')

        # Update base configuration with constants.
        if schedule.get('constants', None) is not None:
            for k, v in schedule['constants'].items():
                base_conf[k] = v
            
        if local_option == 'standard':
            # Check whether number of variables is equal to number of configurations or not.
            for k, v in schedule['variables'].items():
                if len(v) != n_conf:
                    raise ValueError('Number of variables must be equal to number of configurations in STANDARD mode.')
                
            for t in range(n_conf):
                nc = deepcopy(base_conf)
                for k, v in schedule['variables'].items():
                    if '.' in k:
                        k = k.split('.')
                        key = nc
                        for i in range(len(k)-1):
                            key = key[k[i]]
                        key[k[-1]] = v[t]
                    else:
                        nc[k] = v[t]
                config_set.append(nc)
        
        elif local_option == 'grid':
            raise NotImplementedError('GRID mode is not implemented yet.')
        
        elif local_option == 'random':
            raise NotImplementedError('RANDOM mode is not implemented yet.')

    return config_set

def save_temp_config(config, prefix=None):
    try:
        file_path = tempfile.mkstemp(prefix=prefix, suffix='.yaml')[1]
        save_config_to(config, file_path)
        return file_path
    except:
        return False
    
def delete_temp_config(config):
    try:
        Path(config).remove()
    except:
        return False

def main(args):
    cfgs = read_schedule(args.schedule)
    
    # Need to check for duplicated configurations. 
    cfgs_prog_path = Path(Path(args.schedule).stem + '.prog')
    if cfgs_prog_path.exists():
        with cfgs_prog_path.open('r') as f:
            cfgs_prog = f.read().split('\n')
        cfgs_prog = [True if p == 'True' else False for p in cfgs_prog]

        gap = len(cfgs) - len(cfgs_prog)
        if gap > 0:
            cfgs_prog = cfgs_prog + [False for _ in range(gap)]
    else:
        cfgs_prog = [False for _ in range(len(cfgs))]

    cfgs_path = []
    for cfg in cfgs:
        prefix = cfg['conf'].split('/')[-1].split('.')[0] + '_'
        tmp_cfg_path = save_temp_config(cfg, prefix=prefix)
        if tmp_cfg_path:
            cfgs_path.append(tmp_cfg_path)

    # Argument formatting
    args_list = [
        [EasyDict({'conf': n})]
        for n in cfgs_path
    ]
    kwargs_list = None
    # execute training code
    try:
        run_multi_gpu(
            args.n_device, len(cfgs_path), train, cfgs_prog,
            args_list=args_list,
            kwargs_list=kwargs_list
        )
    except:
        traceback.print_exc()
    finally:
        # Delete temporary configuration files.
        for path in cfgs_path:
            delete_temp_config(path)

        # Save training progress.
        with cfgs_prog_path.open('w') as f:
            for p in cfgs_prog:
                f.write(str(p) + '\n')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model.'
    )
    parser.add_argument('exec_file', metavar='exec_file', type=str, help='A path for execution file that you want to run.')
    parser.add_argument('--schedule', '-s', default='schedule.json')
    parser.add_argument('--n_device', '-nd', default=4)
    args = parser.parse_args()

    main(args)