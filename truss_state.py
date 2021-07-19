from collections import namedtuple
from math import pi, sin
import numpy as np
import torch
from copy import deepcopy
from operator import itemgetter
import random

from graph_data import GraphData


Polar = namedtuple('Polar', 'rho phi')


class Node():
    """
    An abstract truss node set in the triangular lattice
    rows run horzontally while columns run at 60 degrees from the positive x-axis
    """
    def __init__(self, row, col, pinned=False):
        self.row = row
        self.col = col
        self.pinned = pinned
        self.target_dist = None
        self.polar = None
        self.damaged = False

    @classmethod
    def create_truss_node(cls, row, col, pinned, target, polar):
        """
        Create a node with distance to target calculated
        """
        n = Node(row, col, pinned)
        n.target_dist = n.tl_dist(target)
        # polar cordinates relative to the target
        n.polar = polar

        return n

    @property
    def loc(self):
        return self.row, self.col

    def tl_dist(self, other):
        """
        The manhattan distance along edges in the tringular lattice between two nodes
        """
        vdist = other.row - self.row
        hdist = other.col - self.col
        if vdist > 0 and hdist < 0:
            hdist = min(hdist + vdist, 0)
        elif vdist < 0 and hdist > 0:
            hdist = max(hdist + vdist, 0)

        return abs(vdist) + abs(hdist)


class NodeMap():
    """
    Maps between truss state node index and the environment observation index 
    which changes over time
    """
    def __init__(self, state_to_env, env_to_state):
        self._state_to_env = state_to_env
        self._env_to_state = env_to_state

    def action_to_env(self, action):
        """ Convert a state action to environment node indexing
        """
        node_ndx, slot = action
        return self._state_to_env[node_ndx], slot

    def action_to_state(self, action):
        """ Convert an environment action to state node indexing
        """
        node_ndx, slot = action
        return self._env_to_state[node_ndx], slot


class Obstacle():
    def __init__(self, row, col, radius=0.5):
        self.row = row
        self.col = col
        self.radius = radius

    def is_obstructing(self, row, col):
        # TODO: add radius checking
        return (self.row == row) and (self.col == col)

class TrussState():
    """
    Holds an abstract truss state for use in the tree search
    """
    node_spacing = 1.0
    row_spacing = node_spacing * sin(pi/3)
    slot_offset = [
        (0, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1)
    ]
    reflection_map = {0:3, 1:2, 2:1, 3:0, 4:5, 5:4}


    def __init__(self, nodes, edge_index, edge_slots, target, origin, obstacles=[]):
        """
        Args:
            nodes: list of Node objects
            edge_index: ndarray [2, edges] the indicies of the graph adjacency matrix
            edge_slots: edge directions as slot_offset indicies in range [0,5], list corresponds to edge_index order
            target: target Node object
            origin: ndarray shape [2] as (x, y) in environment world coordinates
            obstacle: list of obstacles (Obstacle) that the truss may not colocate
        """
        self._target = target
        self._nodes = nodes
        self._node_locs = {(n.row, n.col):i for i, n in enumerate(nodes)}
        self._span_remaining = min([n.target_dist for n in nodes])
        self._edge_index = edge_index
        self._strut_locs = {(edge_index[0,i], edge_index[1,i]):i for i in range(edge_index.shape[1])}
        self._edge_slots = edge_slots
        self._origin = origin
        self._obstacles = obstacles
        self._free_node_slots = [self._get_unobstructed_slots(n.loc) for n in self._nodes]
        for i in range(len(self._edge_slots)):
            node_idx = edge_index[0,i]
            slot_idx = self._edge_slots[i]
            self._free_node_slots[node_idx][slot_idx] = False

    @classmethod
    def get_start_configs(cls, target_distance):
        """
        gets the set of start states (support nodes and targets) for the given 
        target distance in triangular manhattan distance

        Args:
            target_distance - 

        Returns:
            a list of environment start configurations, each a list [target, support1, support2] 
            of world coordinate tuples (x,y)
        """
        def config_pos(origin, pos):
            return tuple(cls.state_to_env(origin, pos[0], pos[1]).tolist())

        states = []
        zero_origin = np.zeros(2)
        pinned_nodes = np.array([[0, 0], [0, -1]])
        grid_delta = np.array(cls.slot_offset)
        for d1 in range(6):
            o1 = np.array(grid_delta[d1]) * target_distance
            d2 = (d1 + 2) % 6
            s = target_distance
            if d1 in [1, 4]:
                s += 1
            for j in range(s):
                o2 = np.array(grid_delta[d2]) * j
                base_node = pinned_nodes[0] if ((d1-2) % 6) > 2 else pinned_nodes[1]
                target = base_node + o1 + o2
                t_pos = cls.state_to_env(zero_origin, target[0], target[1])
                world_origin = -t_pos/2
                t = config_pos(world_origin, target)
                p1 = config_pos(world_origin, pinned_nodes[0])
                p2 = config_pos(world_origin, pinned_nodes[1])
                states.append([t, p1, p2])
        assert len(states) == (target_distance * 6) + 2

        return states

    @classmethod
    def env_to_state(cls, origin, pos):
        """
        Convert a position given in environment world coordinates to a truss
        location

        Args:
            origin: truss origin in environment world coordinates
            pos: position given in environment world coordinates

        Returns:
            a truss location (row, col) that represents a node position in the 
            triangular truss lattice. See Node for details of the coordinate system.
        """
        spos = pos - origin
        row = round(spos[1] / cls.row_spacing)
        col = round((spos[0] / cls.node_spacing) - (row / 2))

        return row, col

    @classmethod
    def state_to_env(cls, origin, row, col):
        """
        Convert a position given as a truss location to environment world coordinates

        Args:
            origin: truss origin in environment world coordinates
            row: row number in the triangular truss lattice
            col: col number in the triangular truss lattice

        Returns:
            environment world coordinates for a node position in the 
            triangular truss lattice. See Node for details of the coordinate system.
        """
        pos = np.array([
            (col * cls.node_spacing) + (row / 2), 
            row * cls.row_spacing
        ], dtype=np.float)
        pos += origin

        return pos

    def node_geometry(self):
        """
        gets the centre positions of nodes and pinned status

        Returns:
            list of centres in world coordinates: (x, y), list of pinned node flags
        """
        def node_to_env(n):
            geo = self.__class__.state_to_env(self._origin, n.row, n.col)
            return tuple(geo)

        return [(node_to_env(n), n.pinned, n.damaged) for n in self._nodes]

    def edge_geometry(self):
        """
        gets the start and end positions of edges

        Returns:
            list of start and end positions in world coordinates: ((x1, y1), (x2, y2))
        """
        def edge_to_env(start_ndx, end_ndx):
            return (node_geometry[start_ndx][0], node_geometry[end_ndx][0])

        node_geometry = self.node_geometry()

        return [edge_to_env(*e) for e in self._edge_index.T]

    def target_geometry(self):
        """
        gets the position of the target

        Returns:
            target position in world coordinates: (x, y)
        """
        return self.__class__.state_to_env(self._origin, self._target.row, self._target.col)

    def obstacles_geometry(self):
        """
        gets the position and shape of the obstacles

        Returns:
            list of obstacle positions and radii in world coordinates: [(x, y, radius), ...]
        """
        return [(self.__class__.state_to_env(self._origin, o.row, o.col), o.radius) for o in self._obstacles]

    @classmethod
    def _target_polar(self, node_loc, target):
        """
        gets the polar world coordinates of a node WRT the target

        Returns:
            polar coordinates: Polar
        """
        row, col = node_loc
        trow, tcol = target.loc
        fake_origin = np.array([0,0])
        x, y = self.state_to_env(fake_origin, row-trow, col-tcol)
        polar = Polar(
            rho=np.sqrt(x**2+y**2),
            phi=np.arctan2(y,x)
        )

        return polar

    @classmethod
    def from_env(cls, obs):
        """
        Create a TrussState object from environment observations
        """
        edge_index = obs['edge_index']
        edge_strut_slots = obs['edge_data']['direction'].tolist()
        node_positions = obs['node_data']['position']
        target_pos = obs['target_position']

        #get polar coordinates
        x, y = (node_positions - target_pos).T
        rho = np.sqrt(x**2+y**2)
        phi = np.arctan2(y,x)

        pinned_nodes = obs['node_data']['pinned']
        origin = np.copy(node_positions[0])
        row, col = cls.env_to_state(origin, target_pos)
        target = Node(row=row, col=col)
        nodes = []
        for i in range(pinned_nodes.size):
            row, col = cls.env_to_state(origin, node_positions[i])
            nodes.append(Node.create_truss_node(
                row=row, 
                col=col, 
                polar=Polar(rho=rho[i], phi=phi[i]),
                pinned=pinned_nodes[i], 
                target=target
            ))

        return TrussState(nodes, edge_index, edge_strut_slots, target, origin)

    def action_included(self, other_ts, action):
        """
        Checks whether any strut on this truss correlates with a strut action on another truss

        Args:
            action: [node_id, slot] of a strut placement action on the other truss
            other_ts: the other truss state

        Returns:
            True if the is a strut correlates
        """
        ots_node_id, slot = action
        loc = other_ts._nodes[ots_node_id].loc
        if loc in self._node_locs:
            node_id = self._node_locs[loc]
            if not self._free_node_slots[node_id][slot]:
                return True
        return False

    def generate_node_map(self, obs):
        """
        Returns:
            a dictionary that maps the ids of state nodes to observed nodes
        """
        o_node_pos = obs['node_data']['position']
        state_to_env, env_to_state = {}, {}
        for i, pos in enumerate(o_node_pos):
            loc = self.env_to_state(self._origin, pos)
            truss_node_ndx = self._node_locs[loc]
            state_to_env[truss_node_ndx] = i
            env_to_state[i] = truss_node_ndx

        return NodeMap(state_to_env, env_to_state)

    @property
    def is_complete(self):
        return self._span_remaining == 0

    @property
    def span_remaining(self):
        return self._span_remaining

    @property
    def min_manhattan_distance(self):
        """
        Gets the minimum Manhattan distance of nodes to the goal
        """
        return self.span_remaining

    @property
    def mean_euclidean_distance(self):
        """
        Gets the mean Euclidean distance from each node to the goal
        """
        rval = sum([n.polar.rho for n in self._nodes]) / len(self._nodes)
        return rval

    def mean_topk(self, k=4):
        """
        Gets the mean Euclidean distance from the closest 4 nodes to the goal
        """
        topk = sorted([n.polar.rho for n in self._nodes])[:k]
        rval = sum(topk) / len(topk)
        return rval

    @property
    def min_euclidean_distance(self):
        """
        Gets the minimum Euclidean distance of nodes to the goal
        """
        rval = min([n.polar.rho for n in self._nodes])
        return rval

    @property
    def num_nodes(self):
        return len(self._nodes)

    def get_graph(self, device, reflect=False):
        """
        get the graph representation of the truss that is used by the neural network

        Args:
            device: torch device to use for tensors
            reflect: if True, the graph shoud be reflected in the y axis to 
                give a symetrical graph that can augment training data
        """
        edge_slots_np = np.array(self._edge_slots)
        node_attr_np = np.array([(
            n.target_dist, 
            n.pinned, 
            n.polar.rho, 
            n.polar.phi
            #n.damaged
        ) for n in self._nodes])

        if reflect:           
            # reflect the edge directions
            edge_slots_np = np.vectorize(self.__class__.reflection_map.get)(edge_slots_np)
            # reflect node phi values
            node_attr_np[:, 3] = pi - node_attr_np[:, 3]

        g =  GraphData(
            edge_index=torch.tensor(self._edge_index, dtype=torch.long, device=device), 
            node_attr=torch.tensor(node_attr_np, dtype=torch.float, device=device), 
            edge_strut_slots=torch.tensor(edge_slots_np, device=device),
            device=device
        )

        return g

    def clone(self):
        return deepcopy(self)

    def get_valid_actions(self, device):
        """
        get the valid actions for this state
        
        Returns:
            a bool tensor shape (nodes, 6) 
        """
        return torch.tensor(self._free_node_slots, device=device)

    def get_actions_list(self, valid):
        """
        get the valid actions for this state
        
        Returns:
            a set of actions
        """
        rval = set()
        for i, n in enumerate(self._free_node_slots):
            for j, a in enumerate(n):
                if a == valid:
                    rval.add((i, j))
        return rval

    def _add_strut(self, node_idx_a, node_idx_b, slot_idx_a):
        """
        Adds a new strut to the truss. This creates 2 edges in the edge index,
        adds 2 corresponding entries to the _strut_locs and _edge_slots tables
        and updates the _free_node_slots table.

        Args:
            node_idx_a: index of node in nodes table
            node_idx_b: index of node in nodes table
            slot_idx_a: index of slot in node a that strut is placed
        """
        assert self._free_node_slots[node_idx_a][slot_idx_a] == True, "a strut already exists between these nodes"

        self._edge_index = np.concatenate([
            self._edge_index,
            np.array([[node_idx_a, node_idx_b], [node_idx_b, node_idx_a]])
        ], axis=1)

        slot_idx_b = (slot_idx_a + 3)%6
        self._edge_slots.extend([slot_idx_a, slot_idx_b])

        self._strut_locs[(node_idx_a, node_idx_b)] = self._edge_index.shape[1] - 2
        self._strut_locs[(node_idx_b, node_idx_a)] = self._edge_index.shape[1] - 1

        self._free_node_slots[node_idx_a][slot_idx_a] = False
        self._free_node_slots[node_idx_b][slot_idx_b] = False


    def _get_unobstructed_slots(self, node_loc):
        """
        finds node slots that can accept a strut not obstructed by an obstacle

        Args:
            node_loc: (row, col) location of the node in the truss
        """
        def is_unobstructed(row, col):
            return not any([o.is_obstructing(row, col) for o in self._obstacles])

        return [is_unobstructed(node_loc[0]+so[0], node_loc[1]+so[1]) for so in self.slot_offset]


    def _add_node(self, node_loc):
        """
        Adds a new node to the truss. This adds the node to _nodes table, updates the
         _span_remaining value if the truss is now closer to the target and updates 
        the _free_node_slots and _node_locs tables

        Args:
            node_loc: (row, col) location of the node in the truss
        """
        row, col = node_loc
        polar = self._target_polar(node_loc, self._target)
        node = Node.create_truss_node(row, col, False, self._target, polar)
        if node.target_dist < self._span_remaining:
            self._span_remaining = node.target_dist
        self._nodes.append(node)
        self._node_locs[node_loc] = len(self._nodes) - 1
        # mark as free any slots that can accept a strut not obstructed by an obstacle
        self._free_node_slots.append(self._get_unobstructed_slots(node_loc))

        return len(self._nodes) - 1

    def _recipricol_edge(self, edge_ndx):
        e = tuple(np.flip(self._edge_index[:, edge_ndx]))
        return self._strut_locs[e]

    def action_update(self, action):
        """
        adds a new strut to the node given in the action arg

        Args:
            action: (node index, slot index)
        """
        node_idx, slot = action
        node = self._nodes[node_idx]
        orow, ocol =  self.slot_offset[slot]
        end_node_loc = node.row+orow, node.col+ocol

        # check whether the node exists yet, if not add it
        if end_node_loc in self._node_locs:
            end_node_idx = self._node_locs[end_node_loc]
            assert (node_idx, end_node_idx) not in self._strut_locs, "strut already exixts"
        else:
            end_node_idx = self._add_node(end_node_loc)
        self._add_strut(node_idx, end_node_idx, slot)


class BreakableTrussState(TrussState):
    """
    A minimal environment model with simple breakability rules
    """
    max_unbraced_struts = 1

    def __init__(self, nodes, edge_index, edge_slots, target, origin, obstacles=[]):
        super().__init__(nodes, edge_index, edge_slots, target, origin, obstacles)

    @classmethod
    def from_truss_state(cls, ts):
        return cls(ts._nodes, ts._edge_index, ts._edge_slots, ts._target, ts._origin)

    @classmethod
    def from_config(cls, config, add_obstacles=False):
        """
        Create a TrussState object from config params

        Returns:
            TrussState
        """
        target_pos, p1, p2 = config
        assert p1[0] > p2[0]
        edge_index = np.array([[0,1], [1,0]])
        edge_strut_slots = [3, 0]
        node_positions = np.stack([p1, p2])

        #get polar coordinates
        x, y = (node_positions - target_pos).T
        rho = np.sqrt(x**2+y**2)
        phi = np.arctan2(y,x)

        origin = np.copy(node_positions[0])
        row, col = cls.env_to_state(origin, target_pos)
        target = Node(row=row, col=col)
        nodes = []
        for i in range(2):
            row, col = cls.env_to_state(origin, node_positions[i])
            nodes.append(Node.create_truss_node(
                row=row, 
                col=col, 
                polar=Polar(rho=rho[i], phi=phi[i]),
                pinned=True, 
                target=target
            ))

        obstacles = []
        if add_obstacles:
            row = nodes[0].row + target.row // 2
            col = nodes[0].col + target.col // 2
            #obstacles.append(Obstacle(row=row, col=col))
            obstacles.append(Obstacle(row=row, col=col-1))
            obstacles.append(Obstacle(row=row, col=col+1))

        return BreakableTrussState(nodes, edge_index, edge_strut_slots, target, origin, obstacles)

    @classmethod
    def create_env(cls, target_dist):
        origin = np.array([0, 0])
        target = Node(row=target_dist, col=0)
        nodes = [
            Node.create_truss_node(0, -1, True, target, TrussState._target_polar((0, -1), target)),
            Node.create_truss_node(0, 0, True, target, TrussState._target_polar((0, 0), target))
        ]
        edge_index = np.array([
            [0, 1],
            [1, 0]
        ])
        edge_slots = [0, 3]

        return cls(nodes, edge_index, edge_slots, target, origin)

    def get_braced_edges(self, check_damaged=False):
        """
        Returns:
            a bool mask of edges, True where they participate in bracing 
            triangles although these are not necessarily stabily connected to 
            the pinned nodes and therefore not stabily braced
        """
        if check_damaged:
            damaged_nodes = [i for i, n in enumerate(self._nodes) if n.damaged]
            test_edge_mask = ~np.isin(self._edge_index, damaged_nodes).any(axis=0)
            test_edge_index = self._edge_index[:, test_edge_mask]
        else:
            test_edge_index = self._edge_index

        adj = torch.sparse_coo_tensor(
            indices=test_edge_index, 
            values=torch.ones(test_edge_index.shape[1]), 
            size=(self.num_nodes, self.num_nodes)
        ).to_dense()
        bracing = adj.matmul(adj) * adj
        braced_edges = bracing[self._edge_index[0], self._edge_index[1]] > 0

        return braced_edges.numpy()

    @property
    def pinned_nodes(self):
        nodes = [ni for ni in range(self.num_nodes) if self._nodes[ni].pinned]
        return nodes

    def get_braced_nodes(self, check_damaged=False):
        """
        Returns:
            set of indicies of nodes that are stabily braced against the 
            statically pinned nodes
        """
        braced_edges = self.get_braced_edges(check_damaged)
        queue = self.pinned_nodes
        braced_nodes = set(queue)
        node_bracing = {}
        edges = np.arange(self._edge_index.shape[1])

        while len(queue) > 0:
            n = queue.pop()
            edge_mask = (self._edge_index[0] == n) & braced_edges
            for n_ndx, e_ndx in zip(self._edge_index[1, edge_mask], edges[edge_mask]):
                if n_ndx in node_bracing:
                    e_ndx1, e_ndx2 = node_bracing[n_ndx]
                    # node is braced if connected by two braced struts(edges)
                    if e_ndx2 is None and e_ndx1 != e_ndx:
                        node_bracing[n_ndx] = (e_ndx1, e_ndx)
                        queue.append(n_ndx)
                        braced_nodes.add(n_ndx)
                else:
                    node_bracing[n_ndx] = (e_ndx, None)

        return braced_nodes

    def get_unbraced_dist(self, check_damaged=False):
        """
        Returns:
            maximum distance of an unbraced node from the braced structure
        """
        bnl = list(self.get_braced_nodes(check_damaged))
        queue = bnl
        braced_nodes = np.zeros(self.num_nodes, dtype=np.bool)
        braced_nodes[bnl] = True
        visited = np.zeros(self.num_nodes, dtype=np.bool)
        unbraced_dist = np.zeros(self.num_nodes, dtype=np.int)

        while len(queue) > 0:
            n = queue.pop(0)
            dist = unbraced_dist[n]
            edge_mask = self._edge_index[0] == n
            for en in self._edge_index[1, edge_mask]:
                d = 0 if braced_nodes[en] else dist + 1
                if not visited[en] or (d < unbraced_dist[en]):
                    visited[en] = True
                    unbraced_dist[en] = d
                    queue.append(en)

        return unbraced_dist.max()

    def is_broken(self):
        """
        True if the truss fails the integrity rules
        """
        rval = self.get_unbraced_dist() > self.max_unbraced_struts

        return rval

    def is_valid(self):
        return not self.is_broken()

    @property
    def is_complete(self):
        complete = False
        if self._span_remaining == 0:
            braced_nodes = self.get_braced_nodes(check_damaged=True)
            target_idx = self._node_locs[self._target.loc]
            complete = target_idx in braced_nodes

        return complete

    @property
    def cannon_str(self):
        """
        Creates a cannonical string representation that is identical for all
        topologicaly equivalent states based on the same root environment 
        state

        Returns:
            a string representation for the truss state
        """
        locs = list(self._node_locs.keys())
        sort_ndx, node_locs = zip(*sorted([(i,e) for i,e in enumerate(locs)], key=itemgetter(1)))
        cmap = {self._node_locs[locs[ni]]:i for i, ni in enumerate(sort_ndx)}
        edge_index = np.vectorize(cmap.__getitem__)(self._edge_index)
        edge_index = sorted([ei.tolist() for ei in edge_index.T])
        srep = str(node_locs) + str(edge_index)

        return srep

    def export_env(self):
        """
        Get the information required to display the base environment
        """
        env = []
        tpos = self.state_to_env(self._origin, self._target.row, self._target.col)
        env.append(tuple(tpos.tolist()))
        
        for n in self.pinned_nodes:
            npos = self.state_to_env(self._origin, self._nodes[n].row, self._nodes[n].col)
            env.append(tuple(npos.tolist()))

        return env

    def damage_random_node(self):
        """
        Marks a random node as damaged

        Returns:
            True if a node was found to damage otherwise false
        """
        def is_damageable(n):
            return not n.damaged and not n.pinned and n.target_dist is not None and n.target_dist > 0

        nodes_idx = [i for i, n in enumerate(self._nodes) if is_damageable(n)]
        if nodes_idx:
            dni = random.choice(nodes_idx)
            dn = self._nodes[dni]
            dn.damaged = True
            self._obstacles.append(Obstacle(row=dn.row, col=dn.col))
            for ni in range(self.num_nodes):
                us = self._get_unobstructed_slots(self._nodes[ni].loc)
                fs = self._free_node_slots[ni] 
                self._free_node_slots[ni] = [u and f for u, f in zip(us,fs)]
            return True
        else:
            return False

    @property
    def min_manhattan_braced_distance(self):
        """
        Gets the minimum Manhattan distance from braced nodes to the goal
        """
        bn_ndx = list(self.get_braced_nodes(check_damaged=True))
        span_remaining = min([self._nodes[n_i].target_dist for n_i in bn_ndx])

        return span_remaining