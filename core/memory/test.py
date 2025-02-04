import jax
import jax.numpy as jnp
from core.memory.replay_memory import EpisodeReplayBuffer, ReplayBufferState, BaseExperience

# Test parameters
batch_size = 1
capacity = 5
gamma = 0.99
final_value = jnp.array([10.0])
num_players = 1

buffer = EpisodeReplayBuffer(capacity)

n_actions = 4
observation_size = 8


template_experience = BaseExperience(
    reward=jnp.array(0.0),
    bootstrapped_return=jnp.array(0.0),
    policy_weights=jnp.ones(n_actions) / n_actions,
    policy_mask=jnp.ones(n_actions, dtype=bool),
    observation_nn=jnp.zeros(observation_size),
    cur_player_id=jnp.array(0)
)

state = buffer.init(batch_size, template_experience)

test_experience = BaseExperience(
    reward = jnp.array([7.0]),
    bootstrapped_return = jnp.array([0.0]),
    policy_weights = jnp.ones((1, n_actions)) / n_actions,
    policy_mask = jnp.ones((1, n_actions), dtype=bool),
    observation_nn = jnp.zeros((1, observation_size)),
    cur_player_id = jnp.array([0])
)

add_experience = jax.vmap(buffer.add_experience)
rewards = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]])

for i in range(capacity):
    experience = BaseExperience(
        reward = rewards[i],
        bootstrapped_return = jnp.array([0.0]),
        policy_weights = jnp.ones((batch_size, n_actions)) / n_actions,
        policy_mask = jnp.ones((batch_size, n_actions), dtype=bool),
        observation_nn = jnp.zeros((batch_size, observation_size)),
        cur_player_id = jnp.array([0])
    )
    state = add_experience(state, experience)

print(state.buffer.bootstrapped_return)
print(state.buffer.reward)

state = buffer.assign_returns(state, final_value, gamma)


print(state.buffer.bootstrapped_return.shape)