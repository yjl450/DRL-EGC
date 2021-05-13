import gym
import argparse
import os
import sys

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"

import gym.wrappers  # NOQA:E402
import numpy as np  # NOQA:E402
from torch import nn  # NOQA:E402
import torch

import pfrl  # NOQA:E402
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import acer  # NOQA:E402
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt  # NOQA:E402
from pfrl.policies import SoftmaxCategoricalHead  # NOQA:E402
from pfrl.q_functions import DiscreteActionValueHead  # NOQA:E402
from pfrl import replay_buffers  # NOQA:E402
EPISODE = 1000


def main():
    # torch.set_default_tensor_type(torch.FloatTensor)
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--env", type=str, default="gym_elevator:Elevator-v0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--replay-start-size", type=int, default=100000)
    parser.add_argument("--n-times-replay", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--setting", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10**9)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=10**4)
    parser.add_argument("--eval-n-runs", type=int, default=20)
    parser.add_argument("--use-lstm", default=False, action="store_true")
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load-pretrained",
                        action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--reward", type=int, default=3)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    args = parser.parse_args()
    if args.setting:
        args.outdir = args.outdir+"/"+str(args.setting)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31


    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = gym.make(args.env, elevator_num=4, elevator_limit=10, floor_num=10,
                       floor_limit=40, step_size=args.max_episode_steps, poisson_lambda=3, 
                       seed=env_seed, reward_func=args.reward, unload_reward=100, 
                       load_reward=100, discount=1)
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    sample_env = make_env(0, False)
    obs_size = sample_env.observation_space.shape[0]
    n_actions = sample_env.action_space.n

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    utils.set_random_seed(args.seed)
    args.__dict__["envarg"] = str(sample_env.args)

    if args.setting:
        args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
        print("Output files are saved in {}".format(args.outdir))
        log_file = args.outdir + "/train_eval.log"
        logging.basicConfig(filename=log_file, filemode='a',level=logging.INFO)
    else:
        args.outdir += "/test"
        args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
        print("Output files are saved in {}".format(args.outdir))
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    input_to_hidden = nn.Sequential(
        nn.Linear(obs_size, 512),
        nn.ReLU(),
        nn.Linear(512, 2592),
        nn.ReLU(),
        nn.Linear(2592, 256),
        nn.ReLU(),
    )

    head = acer.ACERDiscreteActionHead(
        pi=nn.Sequential(
            nn.Linear(256, n_actions),
            SoftmaxCategoricalHead(),
        ),
        q=nn.Sequential(
            nn.Linear(256, n_actions),
            DiscreteActionValueHead(),
        ),
    )

    if args.use_lstm:
        model = pfrl.nn.RecurrentSequential(
            input_to_hidden,
            nn.LSTM(num_layers=1, input_size=256, hidden_size=256),
            head,
        )
    else:
        model = nn.Sequential(input_to_hidden, head)

    model.apply(pfrl.initializers.init_chainer_default)

    # SharedRMSprop is same as torch.optim.RMSprop except that it initializes
    # its state in __init__, allowing it to be moved to shared memory.
    opt = SharedRMSpropEpsInsideSqrt(
        model.parameters(), lr=args.lr, eps=1e-1, alpha=0.99
    )

    rbuf_capacity = 10**6
    betasteps = (args.steps - args.replay_start_size)
    replay_buffer = replay_buffers.PrioritizedReplayBuffer(
        rbuf_capacity, 
        betasteps=betasteps,
        
        num_steps=3,
        normalize_by_max="memory",
    )
    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = acer.ACER(
        model,
        opt,
        t_max=args.t_max,
        gamma=args.gamma,
        replay_buffer=replay_buffer,
        n_times_replay=args.n_times_replay,
        replay_start_size=args.replay_start_size,
        beta=args.beta,
        phi=phi,
        max_grad_norm=40,
        recurrent=args.use_lstm,
    )

    if args.load:
        agent.load(args.load)
    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=args.steps,
            logger=logger
        )
        print(
            "n_steps: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for pg in agent.optimizer.param_groups:
                assert "lr" in pg
                pg["lr"] = value

        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.lr, 0, lr_setter
        )

        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            global_step_hooks=[lr_decay_hook],
            save_best_so_far_agent=True,
            logger=logger
        )


if __name__ == "__main__":
    main()