from ac_env import ACEnv, State
import jax
import jax.numpy as jnp

env = ACEnv(max_relator_length=8)

goal = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0])
goal_state = State(observation=goal, reward=0, terminated=True, _step_count=0, goal=goal)
state = env.step(goal_state, 6)
terminated = state.terminated
state_arr = state.observation


print(f"state array: {state_arr}")
print(f"terminated: {terminated}")
