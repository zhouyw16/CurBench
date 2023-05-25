import multiprocessing
import subprocess
import itertools
from tqdm import tqdm


datasets = ['cola', 'sst2', 'mrpc', 'qqp',
            'stsb', 'mnli', 'qnli', 'rte', 'wnli']
# models = ['lstm','bert','gpt']
models = ['bert']
policies = ['online', 'naive', 'sampling', 'window']
seeds = [42, 666, 777, 888, 999]
tasks = [it for it in itertools.product(datasets, models, policies,seeds)]


def run(data, net, policy, max_iter=2000, seed=42, gpu=0):
    cmd = f'python examples/rl_teacher.py --data {data} --net {net} --policy {policy}  --epoch {max_iter} --seed {seed} --gpu {gpu}'
    print(f"Processing {data} {net} {policy}\nseed:{seed} epoch:{max_iter} on gpu{gpu}\n")
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    r = p.wait()
    if r == 0:
        print(f"Finished {data} {net} {policy}")
    else:
        print(f"\nError occurred when processing {data} {net} {policy}\n")
    return


if __name__ == '__main__':
    pbar = tqdm(total=len(tasks))

    def update(*a):
        pbar.update()

    pool = multiprocessing.Pool(5)  # todo
    for task in tasks:
        r = pool.apply_async(
            run,
            args=(task[0], task[1], task[2],task[3]),
            callback=update
        )
    pool.close()
    pool.join()


# bert 10659M
# gpt 17099M
# lstm
