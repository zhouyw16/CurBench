import argparse

from curbench.algorithms import RLTeacherTrainer

import neptune
from neptune.utils import stringify_unsupported

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--policy', type=str, default='online',
                    help='online, naive, window, sampling')
args = parser.parse_args()

run = neptune.init_run(
        project="2568602045/rl-teacher",
        name=f"trial-{args.data}-{args.net}-{args.policy}",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZGU2NGE3ZS05NWU0LTQ4ZDItOWFmYS0zNzAyNjkyZWQ1ZjAifQ==",
    )

parameters = {
        "dataset":args.data,
        "policy":args.policy,
        "net":args.net,
        "epochs":args.epochs,
        "gpu": args.gpu,
        "seed":args.seed,
    }

# log run parameters
run["parms"] = stringify_unsupported(parameters)

trainer = RLTeacherTrainer(
    data_name=parameters['dataset'],
    net_name=parameters['net'],
    gpu_index=parameters["gpu"],
    num_epochs= parameters['epochs'],
    random_seed=parameters['seed'],
    policy=parameters['policy'],
    tracker=run,
)

# trainer = RLTeacherTrainer(
#     data_name=args.data,
#     net_name=args.net,
#     gpu_index=args.gpu,
#     num_epochs=args.epochs,
#     random_seed=args.seed,
#     policy=args.policy
# )

trainer.fit()
trainer.evaluate()

run.stop()