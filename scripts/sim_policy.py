import argparse
import joblib
import uuid

from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    policy = None
    env = None

    data = joblib.load(args.file)
    if "algo" in data:
        policy = data["algo"].policy
        env = data["algo"].env
    else:
        policy = data['policy']
        env = data['env']
    while True:
        path = rollout(env, policy, max_path_length=args.max_path_length,
                       animated=True, speedup=args.speedup)
