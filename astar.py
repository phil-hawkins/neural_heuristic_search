import os,sys; sys.path.insert(0, os.path.abspath('.'))
import torch
import random
import heapq
from absl import logging
import psutil

from models.utils import Timer
from graph_data import GraphDataBatch

class AStarNode():
    """
    A node in the A* search space
    """
    device = None
    nnet = None
    heuristic = 'Manhattan'
    batch_size = 32

    def __init__(self, state, parent=None, action=None):
        self._state = state
        self._parent = parent
        self._action = action
        self._device = self.__class__.device

        if self._action is not None:
            self._state.action_update(action)

        if self._parent is None:
            self._g = 0
        else:
            self._g = self._parent._g + 1

        if self.__class__.heuristic == 'Manhattan':
            self._h = self._state.min_manhattan_distance
        elif self.__class__.heuristic == 'ManhattanBraced':
            self._h = self._state.min_manhattan_braced_distance
        elif self.__class__.heuristic == 'Euclidean':
            self._h = self._state.min_euclidean_distance
        elif self.__class__.heuristic == 'Mean':
            self._h = self._state.mean_euclidean_distance
        elif self.__class__.heuristic == 'MeanTopK':
            self._h = self._state.mean_topk(k=16)
        elif self.__class__.heuristic == 'HNet':
            with torch.no_grad():
                graph = self._state.get_graph(device=self._device)
                self._h = self.__class__.nnet(graph)
        elif self.__class__.heuristic == 'HNet_batch':
            # h value prediction is deferred until a batch is formed
            self._h = None
        else:
            raise NotImplementedError()

    @property
    def f(self):
        """
        Estimated node search value
        """
        return self._g + self._h

    @property
    def is_goal(self):
        return self._state.is_complete

    @property
    def scene_config(self):
        return self._state.export_env()

    def get_cannonical_str(self):
        return self._state.cannon_str
 
    def is_valid(self, template):
        """
        Check any constraints on the the child as a valid selection
        """
        if template is not None:
            if not template.action_included(other_ts=self._state, action=self._action):
                return False

        return self._state.is_valid()

    def get_children(self, template=None):
        """
        Generator function that returns valid child nodes. To be 
        valid, the transition action must be possible for the state.

        Args:
            template: if supplied this is used to restrict valid actions to ones that are 
                included in the template state

        Returns:

        """
        actions = self._state.get_valid_actions(device='cpu')
        # randomise child order
        action_list = list(actions.nonzero(as_tuple=False))
        random.shuffle(action_list)
        
        for a in action_list:
            child = self.__class__(
                state=self._state.clone(),
                parent=self,
                action=tuple(a.tolist())
            )
            if child.is_valid(template=template):
                yield child

    def get_action_path(self):
        """
        Returns:
            the list of actions that traces the path from the root to this node
        """
        path = []
        n = self
        while n._parent:
            path.append(n._action)
            n = n._parent
        path.reverse()

        return path

    def action_to_env(self, obs, action):
        node_map = self._state.generate_node_map(obs)
        return node_map.action_to_env(action)

    def __lt__(self, other):
        """
        Nodes are ordered by their f score
        """
        return self.f < other.f
    
    def get_train_examples(self, action_path):
        """
        build a set of training examples from the next action path

        Returns:
            a list of training example tuples (graph, steps_to_go, distance)
        """
        def add_examples(st, stg, d, augmentation=[]):
            graph, sgraph = st.get_graph(device='cpu'), state.get_graph(device='cpu', reflect=True)
            examples.append((graph, stg, d, augmentation))
            examples.append((sgraph, stg, d, augmentation + ['symetry']))

        action_path = action_path.copy()
        state = self._state
        done = False
        examples = []
        ua_states = []

        # add examples of states and their realised steps to go
        while not done:
            distance = state.min_manhattan_distance
            steps_to_go = len(action_path)
            add_examples(state, steps_to_go, distance)
            if steps_to_go > 0:
                action = action_path.pop(0)
                # track unused actions for possible negative example states
                uactions = state.get_actions_list(valid=True)
                uactions.discard(action)
                for ua in uactions:
                    ua_states.append((state.clone(), ua, steps_to_go))

                state.action_update(action)
            else:
                done = True

        # add negative examples i.e. states from unused actions that  
        # would probably not reduce steps to the goal
        invalidated_actions = state.get_actions_list(valid=False)
        for state, ua, steps_to_go in ua_states:
            # TODO: add broken states and labels
            # only add states that could not have arisen from a different 
            # sequence of the same actions
            if ua not in invalidated_actions:
                state.action_update(ua)
                if state.is_valid():
                    distance = state.min_manhattan_distance
                    add_examples(state, steps_to_go, distance, augmentation=['path_adjacent'])

        return examples

    @classmethod
    def set_hnet_batch(cls, node_list):
        """
        Predict and set h values for a set of nodes as a batch

        Args:
            node_list: list of nodes to update
        """
        def batches():
            for i in range(0, len(node_list), cls.batch_size):
                yield node_list[i:i + cls.batch_size]

        for bi, batch in enumerate(batches()):
            logging.debug("   batch: {} size: {}".format(bi, len(batch)))
            graphs = [n._state.get_graph(device=n._device) for n in batch]
            data_batch = GraphDataBatch(graphs)
            logging.debug("   truss state graph - nodes: {} edges: {}".format(
                data_batch.node_attr.size(0), 
                data_batch.edge_index.size(1))
            )

            with torch.no_grad():
                h = cls.nnet(data_batch)

            for n, h in zip(batch, h):
                n._h = h.item()
                logging.debug("     h: {}".format(n._h))


class GreedyNode(AStarNode):
    @property
    def f(self):
        return self._h


def memory_used():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def search(root, epsilon=0., eps=10000, max_memory=16, template=None, view=None):
    """
    The A* path search

    Returns:
        the leaf node that reaches the goal
    """
    def is_out_of_resources():
        return ((eps > 0) and (i >= eps)) or ((max_memory > 0) and (max_memory > memory_used()))

    open_set = []
    visited = set()
    heapq.heappush(open_set, root)
    i = 0
    timer = Timer()

    while open_set:
        current = heapq.heappop(open_set)
        if view:
            view.show(current._state)

        if current.is_goal or is_out_of_resources() or timer.is_timed_out:
            stats = {
                'explored_nodes': i, 
                'open_nodes': len(open_set), 
                'time': timer.timing, 
                'timed_out': timer.is_timed_out,
                'out_of_resources' : is_out_of_resources(),
                'path': current.get_action_path(),
                'goal_complete': current.is_goal
            }
                
            return current, stats

        logging.debug("Pre get_children {} Mb used".format(memory_used()))
        children = []
        for child in current.get_children(template=template):
            cstr = child.get_cannonical_str()
            if cstr not in visited:
                visited.add(cstr)
                children.append(child)

        logging.debug("Post get_children {} Mb used".format(memory_used()))
        if children:
            # predict h values for all children if batching is used
            if root.__class__.heuristic == 'HNet_batch':
                root.__class__.set_hnet_batch(children)

            # add children to the open set
            for child in children:
                heapq.heappush(open_set, child)

        i += 1
        logging.debug("{}: {} nodes added {} open nodes".format(i, len(children), len(open_set)))

    assert False, "All reachable states were explored without hitting the goal"