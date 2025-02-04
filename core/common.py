
from functools import partial
from typing import Tuple
import chex
from chex import dataclass 

import jax
import jax.numpy as jnp
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.types import EnvInitFn, EnvStepFn, StepMetadata

def partition(
    data: chex.ArrayTree,
    num_partitions: int
) -> chex.ArrayTree:
    """Partition each array in a data structure into num_partitions along the first axis.
    e.g. partitions an array of shape (N, ...) into (num_partitions, N//num_partitions, ...)

    Args:
    - `data`: ArrayTree to partition
    - `num_partitions`: number of partitions

    Returns:
    - (chex.ArrayTree): partitioned ArrayTree
    """
    return jax.tree_map(
        lambda x: x.reshape(num_partitions, x.shape[0] // num_partitions, *x.shape[1:]),
        data
    )


def step_env_and_evaluator(
    key: jax.random.PRNGKey,
    env_state: chex.ArrayTree,
    env_state_metadata: StepMetadata,
    eval_state: chex.ArrayTree,
    params: chex.ArrayTree,
    evaluator: Evaluator,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int,
    reset: bool = True
) -> Tuple[EvalOutput, chex.ArrayTree,  StepMetadata, bool, bool, chex.Array]:
    """
    - Evaluates the environment state with the Evaluator and selects an action.
    - Performs a step in the environment with the selected action.
    - Updates the internal state of the Evaluator.
    - Optionally resets the environment and evaluator state if the episode is terminated or truncated.

    Args:
    - `key`: rng
    - `env_state`: The environment state to evaluate.
    - `env_state_metadata`: Metadata associated with the environment state.
    - `eval_state`: The internal state of the Evaluator.
    - `params`: nn parameters used by the Evaluator.
    - `evaluator`: The Evaluator.
    - `env_step_fn`: The environment step function.
    - `env_init_fn`: The environment initialization function.
    - `max_steps`: The maximum number of environment steps per episode.
    - `reset`: Whether to reset the environment and evaluator state if the episode is terminated or truncated.

    Returns:
    - (EvalOutput, chex.ArrayTree, StepMetadata, bool, bool, chex.Array)
        - `output`: The output of the evaluation.
        - `env_state`: The updated environment state.
        - `env_state_metadata`: Metadata associated with the updated environment state.
        - `terminated`: Whether the episode is terminated.
        - `truncated`: Whether the episode is truncated.
        - `rewards`: Rewards emitted by the environment.
    """
    key, evaluate_key = jax.random.split(key)
    # evaluate the environment state
    output = evaluator.evaluate(
        key=evaluate_key,
        eval_state=eval_state,
        env_state=env_state,
        root_metadata=env_state_metadata,
        params=params,
        env_step_fn=env_step_fn
    )
    # take the selected action
    env_state, env_state_metadata = env_step_fn(env_state, output.action)
    # check for termination and truncation
    terminated = env_state_metadata.terminated
    truncated = env_state_metadata.step > max_steps 
    # reset the environment and evaluator state if the episode is terminated or truncated
    # else, update the evaluator state
    reward = env_state_metadata.reward
    eval_state = jax.lax.cond(
        terminated | truncated,
        evaluator.reset if reset else lambda s: s,
        lambda s: evaluator.step(s, output.action),
        output.eval_state
    )
    # reset the environment if the episode is terminated or truncated
    env_state, env_state_metadata = jax.lax.cond(
        terminated | truncated,
        lambda _: env_init_fn(key) if reset else (env_state, env_state_metadata),
        lambda _: (env_state, env_state_metadata),
        None
    )
    output = output.replace(eval_state=eval_state)
    return output, env_state, env_state_metadata, terminated, truncated, reward