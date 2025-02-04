import jax
from jax import vmap, pmap, jit
import jax.numpy as jnp
import chex
from chex import dataclass
from bloom_filter import BloomFilter, BloomFilterState
from core.memory.replay_memory import BaseExperience

@dataclass(frozen=True)
class Node:
    """Data structure for node in BFS frontier.
    Args:
        - `state`: current state of node
        - `path`: path of actions to current node
        - `next_idx`: index of next action in path
    """
    state: chex.Array
    path: chex.Array
    next_idx: int


@dataclass(frozen=True)
class BFSState:
    """Data structure for mutable Frontier states.
    Args:
        - `frontier`: array of NodeStates comprising current frontier
        - `slice_idx`: index of next node to be added to frontier
        - `contact_mask`: mask for frontier nodes that have been in contact with goal frontier
        - `filter_state`: state of BloomFilter for frontier nodes
    """
    frontier: chex.Array 
    slice_idx: int
    contact_mask: chex.Array
    filter_state: BloomFilterState
    
class BFS:
    """
    Args:
        - `env`: environment
        - `filter`: BloomFilter for frontier nodes
    """
    def __init__(self, env):
        self.env = env
        self.filter = BloomFilter()
    
    def node_transition(self, node: Node, action: int) -> Node:
        next_state = self.env.transition(node.state, action)
        next_path = node.path.at[node.next_idx].set(action)
        return Node(next_state, next_path, node.next_idx+1)

    def transition(self, bfs_state: BFSState) -> BFSState:
        actions = jnp.arange(self.env.n_actions)
        frontier = bfs_state.frontier[:bfs_state.slice_idx]
        next_frontier = vmap(
            lambda node: vmap(self.node_transition, in_axes=(None, 0))(node, actions)
        )(frontier).flatten()
        next_filter_state = self.filter.insert(next_frontier.state, bfs_state.filter_state)
        next_slice_idx = next_frontier.shape[0]
        next_frontier = bfs_state.frontier.at[:next_slice_idx].set(next_frontier)
        return BFSState(next_frontier, next_slice_idx, next_filter_state)

    def init(self, capacity: int, root: Node, contact: bool) -> BFSState:
        """Initialize BFSState with root node padded to size `capacity`.
        
        Args:
        - `capacity`: maximum number of nodes in frontier
        - `root`: root node of BFS tree
        
        Returns:
        - `BFSState`: initialized frontier state. `padded_frontier` is standardized to an array/pytree of size `capacity`.
        """
        zero_padded_frontier = jax.tree_util.tree_map(
            lambda x: jnp.zeros((capacity, *x.shape)),
            root
        )
        zero_mask = jnp.zeros(capacity, dtype=jnp.bool_)
        return BFSState(
            frontier = jax.tree_util.tree_map(
                lambda x, y: x.at[0].set(y),
                zero_padded_frontier,
                root
            ),
            slice_idx=1,
            filter_state=self.filter.init(),
            contact_mask=zero_mask.at[0].set(contact)
        )

    def search(self, start_state: chex.Array, goal_state: chex.Array, depth: int) -> chex.Array:
        """Bidirectional breadth-first search between start and goal states.
        Args:
            - `start_state`: start state of BFS tree
            - `goal_state`: goal state of BFS tree
            - `depth`: depth of BFS search
        Returns:
            - `start_bfs_state`: BFSState of start frontier
            - `goal_bfs_state`: BFSState of goal frontier
        """

        def bfs_loop(i, carry):
            start_bfs_state, goal_bfs_state = carry
            new_start_bfs_state = self.transition(start_bfs_state)
            new_goal_bfs_state = self.transition(goal_bfs_state)

            start_frontier = start_bfs_state.frontier[:start_bfs_state.slice_idx]
            new_start_frontier = new_start_bfs_state.frontier[:new_start_bfs_state.slice_idx]
            new_goal_frontier = new_goal_bfs_state.frontier[:new_goal_bfs_state.slice_idx]

            start_mask_01 = self.filter.query(
                start_frontier.state,
                new_goal_bfs_state.filter_state
            )
            goal_mask_01 = self.filter.query(
                new_goal_frontier.state,
                start_bfs_state.filter_state
            )
            start_mask_11 = self.filter.query(
                new_start_frontier.state,
                new_goal_bfs_state.filter_state
            )
            goal_mask_11 = self.filter.query(
                new_goal_frontier.state,
                new_start_bfs_state.filter_state
            )

            contact_00 = jnp.any(start_bfs_state.contact_mask) & jnp.any(goal_bfs_state.contact_mask)
            contact_01 = ~contact_00 & jnp.any(start_mask_01) & jnp.any(goal_mask_01)
            contact_case = (contact_00 + 2*contact_01 + 3*(~contact_00 & ~contact_01) - 1).astype(jnp.int32)

            return jax.lax.switch(
                contact_case,
                [
                    lambda _: (start_bfs_state, goal_bfs_state),
                    lambda _: (
                        start_bfs_state.replace(
                            contact_mask=start_bfs_state.contact_mask.at[:start_bfs_state.slice_idx].set(start_mask_01)
                        ),
                        new_goal_bfs_state.replace(
                            contact_mask=new_goal_bfs_state.contact_mask.at[:new_goal_bfs_state.slice_idx].set(goal_mask_01)
                        )
                    ),
                    lambda _: (
                        new_start_bfs_state.replace(
                            contact_mask=new_start_bfs_state.contact_mask.at[:new_start_bfs_state.slice_idx].set(start_mask_11)
                        ),
                        new_goal_bfs_state.replace(
                            contact_mask=new_goal_bfs_state.contact_mask.at[:new_goal_bfs_state.slice_idx].set(goal_mask_11)
                        )
                    )
                ]
            )

        start = Node(
            state=start_state,
            path=(-1)*jnp.ones(depth, dtype=jnp.int32),
            next_idx=0
        )
        goal = Node(
            state=goal_state,
            path=(-1)*jnp.ones(depth, dtype=jnp.int32),
            next_idx=0
        )
        contact = jnp.all(start_state == goal_state)
        start_bfs_state = self.init(capacity=self.env.n_actions**depth, root=start, contact=contact)
        goal_bfs_state = self.init(capacity=self.env.n_actions**depth, root=goal, contact=contact)
        init_carry = (start_bfs_state, goal_bfs_state)
        start_bfs_state, goal_bfs_state = jax.lax.fori_loop(0, depth, bfs_loop, init_carry)

        return start_bfs_state, goal_bfs_state
    
    # Not Jittable
    def get_paths(self, start_bfs_state: BFSState, goal_bfs_state: BFSState) -> chex.Array:
        start_frontier = start_bfs_state.frontier[:start_bfs_state.slice_idx]
        goal_frontier = goal_bfs_state.frontier[:goal_bfs_state.slice_idx]

        start_contact = start_frontier[start_bfs_state.contact_mask]
        goal_contact = goal_frontier[goal_bfs_state.contact_mask]

        matches = vmap(
            lambda x: vmap(lambda y: jnp.all(x.state == y.state))(goal_contact)
        )(start_contact)

        start_indices, goal_indices = jnp.where(matches) 

        start_paths = start_contact[start_indices].path
        goal_paths = goal_contact[goal_indices].path

        return start_paths, goal_paths


        
        
       









