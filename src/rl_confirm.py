"""One confirmatory run of the RL routing study: train (prereg settings), evaluate on the binned sets, mechanism
measures. Usage: python src/rl_confirm.py <tag> <arm> <lr> [rl_seed]"""
import sys
sys.path[:0] = ['/Users/alityb/projects/cubemoe/src']
from rl_train import train
from rl_eval import evaluate
import rl_mech
tag, arm, lr = sys.argv[1], sys.argv[2], float(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
train(tag, arm, lr, steps=300, seed=seed)
run = f'{tag}_{arm}_lr{lr:g}_s{seed}'
evaluate(tag, run); rl_mech.main(tag, [run])
