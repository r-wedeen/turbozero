
import chex
from chex import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BaseExperience:
    """Experience data structure. Stores a training sample.
    - `reward`: reward attained to get to this state
    - `bootstrapped_return`: bootstrapped return for this state
    - `policy_weights`: policy weights
    - `policy_mask`: mask for policy weights (mask out invalid/illegal actions)
    - `observation_nn`: observation for neural network input
    """
    reward: float
    policy_weights: chex.Array
    policy_mask: chex.Array
    observation_nn: chex.Array
    bootstrapped_return: float=0.0


@dataclass(frozen=True)
class ReplayBufferState:
    """State of the replay buffer. Stores objects stored in the buffer 
    and metadata used to determine where to store the next object, as well as 
    which objects are valid to sample from.
    - `next_idx`: index where the next experience will be stored
    - `episode_start_idx`: index where the current episode started, samples are placed in order
    - `buffer`: buffer of experiences
    - `populated`: mask for populated buffer indices
    - `has_return`: mask for buffer indices that have been assigned a return
        - we store samples from in-progress episodes, but don't want to be able to sample them 
        until the episode is complete
    """
    next_idx: chex.Array # type: int32, shape = (batch_size,)
    episode_start_idx: chex.Array # type: int32, shape = (batch_size,)
    buffer: BaseExperience # type: BaseExperience, shape = (batch_size, capacity, ...)
    populated: chex.Array # type: bool, shape = (batch_size, capacity)
    has_return: chex.Array # type: bool, shape = (batch_size, capacity)


class EpisodeReplayBuffer:
    """Replay buffer, stores trajectories from episodes for training.
    
    Compatible with `jax.jit`, `jax.vmap`, and `jax.pmap`."""

    def __init__(self,
        capacity: int,
    ):
        """
        Args:
        - `capacity`: number of experiences to store in the buffer
        """
        self.capacity = capacity


    def get_config(self):
        """Returns the configuration of the replay buffer. Used for logging."""
        return {
            'capacity': self.capacity,
        }


    def add_experience(self, state: ReplayBufferState, experience: BaseExperience) -> ReplayBufferState:
        """Adds an experience to the replay buffer.
        
        Args:
        - `state`: replay buffer state
        - `experience`: experience to add
        
        Returns:
        - (ReplayBufferState): updated replay buffer state"""
        return state.replace(
            buffer = jax.tree_util.tree_map(
                lambda x, y: x.at[state.next_idx].set(y),
                state.buffer,
                experience
            ),
            next_idx = (state.next_idx + 1) % self.capacity,
            populated = state.populated.at[state.next_idx].set(True),
            has_return = state.has_return.at[state.next_idx].set(False)
        )
    

    def assign_returns(self, state: ReplayBufferState, final_value: float, gamma: float=1.0) -> ReplayBufferState:
        """ Assign bootstrapped returns to the current episode.
        
        Args:
        - `state`: replay buffer state
        - `final_value`: value of the final state in the episode
        - `gamma`: discount factor

        Returns:
        - (ReplayBufferState): updated replay buffer state
        """

        shift = self.capacity - state.next_idx
        rewards = jnp.roll(state.buffer.reward, shift=shift)
        arr = jnp.zeros(self.capacity).at[-1].set(final_value)

        mask = jnp.arange(self.capacity) >= (state.episode_start_idx + shift) % self.capacity

        def body_fun(i, arr):
            z = mask[i] * (gamma * arr[i+1] + rewards[i+1]) + (1-mask[i]) * arr[i]
            return arr.at[i].set(z)

        returns = jax.lax.fori_loop(2, self.capacity+1, lambda i, x: body_fun(self.capacity-i, x), arr)
        returns = jnp.roll(returns, shift=-shift)

        state = state.replace(
            episode_start_idx = state.next_idx,
            has_return = jnp.full_like(state.has_return, True),
            buffer = state.buffer.replace(
                bootstrapped_return = jnp.where(
                    ~state.has_return,
                    returns,
                    state.buffer.bootstrapped_return
                )
            )
        )

        return state
    
    # assumes input is batched!! (dont vmap/pmap)
    def sample(self,
        state: ReplayBufferState,
        key: jax.random.PRNGKey,
        sample_size: int
    ) -> chex.ArrayTree:
        """Samples experiences from the replay buffer.

        Assumes the buffer has two batch dimensions, so shape = (devices, batch_size, capacity, ...)
        Perhaps there is a dimension-agnostic way to do this?

        Samples across all batch dimensions, not per-batch/device.
        
        Args:
        - `state`: replay buffer state
        - `key`: rng
        - `sample_size`: size of minibatch to sample

        Returns:
        - (chex.ArrayTree): minibatch of size (sample_size, ...)
        """
        masked_weights = jnp.logical_and(
            state.populated,
            state.has_return
        ).reshape(-1)

        num_partitions = state.populated.shape[0]
        num_batches = state.populated.shape[1]

        indices = jax.random.choice(
            key,
            self.capacity * num_partitions * num_batches,
            shape=(sample_size,),
            replace=False,
            p = masked_weights / masked_weights.sum()
        )

        partition_indices, batch_indices, item_indices = jnp.unravel_index(
            indices,
            (num_partitions, num_batches, self.capacity)
        )
        
        sampled_buffer_items = jax.tree_util.tree_map(
            lambda x: x[partition_indices, batch_indices, item_indices],
            state.buffer
        )

        return sampled_buffer_items
    
    
    def init(self, batch_size: int, template_experience: BaseExperience) -> ReplayBufferState:
        """Initializes the replay buffer state.
        
        Args:
        - `batch_size`: number of parallel environments
        - `template_experience`: template experience data structure
            - just used to determine the shape of the replay buffer data

        Returns:
        - (ReplayBufferState): initialized replay buffer state
        """
        return ReplayBufferState(
            next_idx = jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_start_idx = jnp.zeros((batch_size,), dtype=jnp.int32),
            buffer = jax.tree_util.tree_map(
                lambda x: jnp.zeros((batch_size, self.capacity, *x.shape), dtype=x.dtype),
                template_experience
            ),
            populated = jnp.full((batch_size, self.capacity,), fill_value=False, dtype=jnp.bool_),
            has_return = jnp.full((batch_size, self.capacity,), fill_value=True, dtype=jnp.bool_),
        )
