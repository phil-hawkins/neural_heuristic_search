"""
Run evaluation on to compare agents

hnet
astar (manhattan)
astar (mean)
greedy (manhattan)
greedy (mean)

evaluation statistics with error margin at 0.05 confidence:

time
open set size
explored set size
structure size (struts)

over d in [1,6]
"""
import os,sys; sys.path.insert(0, os.path.abspath('.'))
from absl import app, flags, logging
import json
from random import shuffle
import psutil
#from memory_profiler import profile

from scenario import plan_path
from truss_state import BreakableTrussState

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'show debug logging messages')
flags.DEFINE_integer('target_dist', 1, 'triangular lattice manhattan distance to target')
flags.DEFINE_float('epsilon', 0., 'probability of random node vs greedy selection in search')
flags.DEFINE_integer('timeout', 0, 'end path planning after this many seconds or no limmit if 0')
flags.DEFINE_integer('eps', 0, 'upper bound on the number of expansions to do per stage. Unlimmited if 0')
flags.DEFINE_integer('max_scenarios', 25, 'upper bound on the number of scenarios to run. Unlimmited if 0')
flags.DEFINE_string('result_file', './logs/results.json', 'output file for results')
flags.DEFINE_string('planner_file', None, 'planner list file path')
flags.DEFINE_integer('batch_size', 128, 'network input batch size')
flags.DEFINE_boolean('add_obstacles', False, 'add obstacles to the space')

#@profile(precision=4)
def main(_argv):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    
    if FLAGS.planner_file is not None:
        with open(FLAGS.planner_file, "r") as f:
            config = json.load(f)
            planners = config['planners']
    else:
        planners = [
            [
                "Greedy",
                "HNet_batch",
                "GIN",
                "GIN"
            ],
            [
                "Greedy", 
                "Manhattan",
                None,
                None
            ],
            [
                "Greedy", 
                "Mean",
                None,
                None
            ]
        ]

    results = []

    # generate the scenario configurations
    logging.info("Target distance {}".format(FLAGS.target_dist))
    start_configs = BreakableTrussState.get_start_configs(FLAGS.target_dist)
    shuffle(start_configs)
    if FLAGS.max_scenarios > 0:
        start_configs = start_configs[:FLAGS.max_scenarios]

    for i, start_config in enumerate(start_configs):
        logging.info("  Scenario {}".format(i))
        logging.info("  {} Mb used".format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
        
        for planner, heuristic, model_config, checkpoint in planners:
            model_checkpoint = None if model_config is None else "models/{}.pt".format(checkpoint)
            stats = plan_path(
                start_state=BreakableTrussState.from_config(start_config, add_obstacles=FLAGS.add_obstacles),
                greedy=(planner == 'Greedy'),
                heuristic=heuristic,
                render=False,
                checkpoint=model_checkpoint,
                model_config=model_config,
                batch_size=FLAGS.batch_size,
                eps=FLAGS.eps,
                timeout=FLAGS.timeout
            )
            if stats['timed_out']:
                logging.info("    Timed out after {} seconds".format(stats['time']))
            else:
                logging.info("    {} ({}): time-{:.2f}s, struts-{}, expansions-{}, {} Mb used".format(
                    planner, 
                    heuristic if model_config is None else model_config, 
                    stats['time'], 
                    len(stats['path']),
                    stats['explored_nodes'],
                    psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                ))
            stats['planner'] = planner
            stats['heuristic'] = heuristic
            stats['model_config'] = model_config
            stats['checkpoint'] = checkpoint
            stats['distance'] = FLAGS.target_dist
            stats['scenario'] = i
            results.append(stats)

    with open(FLAGS.result_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    app.run(main)
