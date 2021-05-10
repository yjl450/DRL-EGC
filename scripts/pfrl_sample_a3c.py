import gym
import argparse
import os
import functools

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  # NOQA:E402
from torch import nn  # NOQA:E402

import pfrl  # NOQA:E402
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import a3c  # NOQA:E402
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt  # NOQA:E402
from pfrl.policies import SoftmaxCategoricalHead  # NOQA:E402


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--env", type=str, default="gym_elevator:Elevator-v0")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed [0, 2 ** 31)")
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
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=25000)
    parser.add_argument("--eval-n-runs", type=int, default=100)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load-pretrained",
                        action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--reward", type=int, default=1)
    parser.add_argument(
        "--num-demo-envs", type=int, default=1, help="Number of demo envs run in parallel."
    )
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

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = gym.make(args.env, elevator_num=2, elevator_limit=10, floor_num=3,
                       floor_limit=10, step_size=args.steps, poisson_lambda=1, 
                       seed=env_seed, reward_func=args.reward, unload_reward=100, 
                       load_reward=100, discount=None)
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

    model = nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 512),
        nn.Tanh(),
        nn.Linear(512, 2592),
        nn.Tanh(),
        # nn.Conv2d(obs_size, 16, 8, stride=4),
        # nn.ReLU(),
        # nn.Conv2d(16, 32, 4, stride=2),
        # nn.ReLU(),
        # nn.Flatten(),
        nn.Linear(2592, 256),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(256, n_actions),
                SoftmaxCategoricalHead(),
            ),
            nn.Linear(256, 1),
        ),
    )

    # SharedRMSprop is same as torch.optim.RMSprop except that it initializes
    # its state in __init__, allowing it to be moved to shared memory.
    opt = SharedRMSpropEpsInsideSqrt(
        model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99)
    assert opt.state_dict()["state"], (
        "To share optimizer state across processes, the state must be"
        " initialized before training."
    )

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = a3c.A3C(
        model,
        opt,
        t_max=args.t_max,
        gamma=1,
        beta=args.beta,
        phi=phi,
        max_grad_norm=40.0,
    )

    if args.load or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model("A3C", args.env, model_type=args.pretrained_type)[
                    0
                ]
            )
    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_demo_envs))
            ]
        )
    if args.demo:
        env = make_batch_env(True)
        eval_stats = experiments.eval_performance(
            env=args.env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=args.steps,
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
        )


if __name__ == "__main__":
    main()
