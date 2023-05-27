import multiprocessing
import subprocess
import itertools
from tqdm import tqdm


datasets = ['cifar10', 'cifar100','tinyimagenet']
# datasets = [ _ + '-noise-0.4' for _ in datasets]
datasets = [ _ + '-imbalance-50' for _ in datasets]
# datasets = datasets + dataset_imbalance + datasets_noise

models = ['vit']
policies = ['online', 'naive', 'sampling', 'window']
seeds = [42, 666, 777, 888, 999]
tasks = [it for it in itertools.product(datasets, models, policies,seeds)]

error_list = []

def run(data, net, policy, seed, max_iter=200, gpu=5):
    cmd = f'python examples/rl_teacher.py --data {data} --net {net} --policy {policy} --epoch {max_iter} --seed {seed} --gpu {gpu}'
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

    pool = multiprocessing.Pool(3)  # todo
    for task in tasks:
        r = pool.apply_async(
            run,
            args=(task[0], task[1], task[2],task[3]),
            callback=update
        )
    pool.close()
    pool.join()
