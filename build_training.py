"""
run A* path planning to provide a clairvoyant oracle for training data from the set of enviroment scenarios 
determined by target_dist d as Manhattan distance from the base platform to the target position
"""
from absl import app, flags, logging
from pickle import Pickler

from scenario import plan_path
from truss_state import BreakableTrussState

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'show debug logging messages')
flags.DEFINE_integer('target_dist', 1, 'triangular lattice manhattan distance to target')
flags.DEFINE_integer('eps', 0, 'number of expansions to do per stage. Unlimmited if 0')
flags.DEFINE_string('train_example_path', "./data/h_net_train.pkl", 'training examples output file')

def main(_argv):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    train_examples = []
    start_configs = BreakableTrussState.get_start_configs(FLAGS.target_dist)
    for i, start_config in enumerate(start_configs):
        logging.debug("Running scenario {}".format(i))
        ex, end_state = plan_path(
            start_state=BreakableTrussState.from_config(start_config),
            greedy=False,
            heuristic='Manhattan',
            eps=FLAGS.eps,
            render=False,
            return_examples=True
        )
        train_examples.extend(ex)
        
        # logging.debug("Damaging a node")
        # damaged_state = end_state.clone()
        # if damaged_state.damage_random_node():
        #     logging.debug("Running damaged node scenario {}".format(i))
        #     ex, end_state = plan_path(
        #         start_state=damaged_state,
        #         greedy=False,
        #         heuristic='ManhattanBraced',
        #         eps=FLAGS.eps,
        #         render=False,
        #         return_examples=True
        #     )
        #     train_examples.extend(ex)

    logging.info("Saving {} training examples to {}".format(len(train_examples), FLAGS.train_example_path))
    with open(FLAGS.train_example_path, "wb") as f:
        Pickler(f).dump(train_examples)

if __name__ == '__main__':
    app.run(main)