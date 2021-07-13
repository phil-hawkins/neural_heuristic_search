"""
run A* path planning to provide a clairvoyant oracle for training data from the set of enviroment scenarios 
determined by target_dist d as Manhattan distance from the base platform to the target position
"""
from absl import app, flags, logging
from pickle import Pickler

from scenario import plan_path
from environment.truss_env import SpanTruss_GymEnv
from truss_state import TrussState

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'show debug logging messages')
flags.DEFINE_integer('target_dist', 2, 'triangular lattice manhattan distance to target')
flags.DEFINE_integer('eps', 0, 'number of expansions to do per stage. Unlimmited if 0')
flags.DEFINE_string('train_example_path', "./data/h_net_train.pkl", 'training examples output file')

def main(_argv):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    train_examples = []
    start_configs = TrussState.get_start_configs(FLAGS.target_dist)
    for i, start_config in enumerate(start_configs):
        logging.debug("Running scenario {}".format(i))
        env = SpanTruss_GymEnv(supports=start_config)
        ex = plan_path(
            env=env,
            eps=FLAGS.eps,
            greedy=False,
            heuristic='Manhattan',
            render=False,
            epsilon=0.,
            return_examples=True
        )
        train_examples.extend(ex)
        env.close()

    logging.info("Saving {} training examples to {}".format(len(train_examples), FLAGS.train_example_path))
    with open(FLAGS.train_example_path, "wb") as f:
        Pickler(f).dump(train_examples)

if __name__ == '__main__':
    app.run(main)