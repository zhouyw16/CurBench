import multiprocessing
import subprocess
import itertools
from tqdm import tqdm

seeds = [42, 666, 777, 888, 999]

datasets = ['nci1', 'mutag', 'proteins', 'dd',
            'ptcmr', 'cora', 'citeseer', 'pubmed']

datasets_noise = [ _ + '-noise-0.4' for _ in datasets]

models = ['gcn', 'gat', 'sage']

policies = ['online', 'naive', 'sampling', 'window']

tasks = [it for it in itertools.product(datasets_noise, models, policies, seeds)]


def run(data, net, policy, seed, max_iter=200, gpu=2):
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
        print(f"Finished {data} {net} {policy} {seed}")
    else:
        print(
            f"\nError occurred when processing {data} {net} {policy} {seed}\n")
    return


if __name__ == '__main__':
    pbar = tqdm(total=len(tasks))

    def update(*a):
        pbar.update()

    pool = multiprocessing.Pool(5)  # todo
    for task in tasks:
        print(task)
        r = pool.apply_async(
            run,
            args=(task[0], task[1], task[2], task[3]),
            callback=update
        )
    pool.close()
    pool.join()


# gcn 1059M
# gat 1135M
# sage 1061M
