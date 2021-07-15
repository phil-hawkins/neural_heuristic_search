import os,sys; sys.path.insert(0, os.path.abspath('.'))
from absl import app, flags, logging
import json
from time import sleep
import torch
import random
import pyglet
import numpy as np

from truss_state import TrussState, BreakableTrussState
from models.config import args
from astar import AStarNode, GreedyNode, search
from view import View

FLAGS = flags.FLAGS

def render_build(view, path, state, save_images=False, stay_open=True):
    """
    run the build path actions in the simulation environment and render the results step by step
    """

    for i, action in enumerate(path):
        logging.debug("Doing action {}".format(action))
        sleep(0.5)
        state.action_update(action)
        view.show(state)
        
        if save_images:
            filename = "logs/images/step_{:03d}.png".format(i)
            pyglet.image.get_buffer_manager().get_color_buffer().save(filename)

        if not view.window_still_open: 
            break

    while stay_open and view.window_still_open:
        view.show(state)
        sleep(0.1)


def save_log_file(file_path, stats):
    """
    Save the environment definition and action path for action replay visualisation
    """
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = []

    all_stats.append(stats)
    with open(file_path, 'w') as f:
        json.dump(all_stats, f)


def plan_path(start_state, greedy, heuristic, render,  
            checkpoint=None, model_config=None, batch_size=32,
            eps=1000, save_images=False,
            timeout=0, return_examples=False, pretrained_net=None):
    """
    Runs the search to plan an action path for the truss build

    Args:
        start_state
        greedy: use greedy search instead of A*
        heuristic: type of heuristic to use, see astar.py
        render: if True, render the build process graphically
        checkpoint
        model_config
        batch_size
        eps
        save_images
        timeout
        return_examples: returns training examples if True
        pretrained_net
    """
    view = View() if render else None
    if view:
        view.show(start_state)
    Node = GreedyNode if greedy else AStarNode 
    Node.heuristic = heuristic
    Node.batch_size = batch_size

    if pretrained_net is not None:
        Node.nnet = pretrained_net
        Node.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif (heuristic == 'HNet') or (heuristic == 'HNet_batch'):
        Node.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = args[model_config] 
        Node.nnet = config['nnet'](config).to(Node.device)      
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location=Node.device)
            Node.nnet.load_state_dict(checkpoint['state_dict'])

    root = Node(state=start_state.clone())
    end_state, stats = search(root, eps=eps)
    
    if timeout and stats['time'] > timeout:
        logging.debug("Timed out after {} seconds".format(stats['time']))
    else:
        logging.debug("Search took {} seconds".format(stats['time']))
        logging.debug("Action path {}".format(stats['path']))   
        if view and view.window_still_open:
            render_build(view, stats['path'], start_state.clone(), save_images=save_images, stay_open=False)


    logging.debug("Damaging a node")
    end_state._state.damage_random_node()
    damaged_state = end_state._state.clone()
    view.show(damaged_state)
    root = Node(state=damaged_state.clone())
    _, stats = search(root, eps=eps)
    render_build(view, stats['path'], damaged_state, save_images=save_images)

def main(_argv):
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    BreakableTrussState.max_unbraced_struts = FLAGS.max_unsupported_struts
    config = random.choice(TrussState.get_start_configs(FLAGS.target_dist))
    checkpoint = "models/{}.pt".format(FLAGS.model_config) if FLAGS.checkpoint is None else FLAGS.checkpoint
    plan_path(
        start_state=BreakableTrussState.from_config(config),
        greedy=FLAGS.greedy,
        heuristic=FLAGS.heuristic,
        render=FLAGS.render,
        checkpoint=checkpoint,
        model_config=FLAGS.model_config,
        batch_size=FLAGS.batch_size,
        eps=FLAGS.eps,
        save_images=FLAGS.save_images
    )

if __name__ == '__main__':
    flags.DEFINE_boolean('debug', False, 'show debug logging messages')
    flags.DEFINE_integer('max_unsupported_struts', 1, 'maximum number of connected unsupported struts before collapse')
    flags.DEFINE_integer('eps', 0, 'number of expansions to do per stage. Unlimmited if 0')
    flags.DEFINE_integer('target_dist', 2, 'triangular lattice manhattan distance to target')
    flags.DEFINE_string('log_file_path', "./logs/astar_log.json", 'result statistics log file')
    flags.DEFINE_string('train_example_path', "./logs/h_net_train.pkl", 'training examples output file')
    flags.DEFINE_string('model_config', "LGC", 'nueral net configutation arguments')
    flags.DEFINE_integer('batch_size', 32, 'network input batch size')
    flags.DEFINE_boolean('render', True, 'display the build steps')
    flags.DEFINE_string('checkpoint', None, 'nueral net parameter checkpoint')
    flags.DEFINE_boolean('greedy', False, 'use greedy search')
    flags.DEFINE_boolean('save_images', False, 'snaphot an image of each build step in the render')
    flags.DEFINE_enum('heuristic', 'HNet_batch', ['Manhattan', 'Mean', 'HNet', 'HNet_batch', 'MeanTopK'],
                      'type of heuristic function to use in search')
    app.run(main)