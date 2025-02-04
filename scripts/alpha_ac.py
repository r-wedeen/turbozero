# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
from core.types import StepMetadata
from ac_env import ACEnv
from jax import numpy as jnp

max_relator_length = 8
env = ACEnv(max_relator_length)

def step_fn(state, action):
    new_state = env.step(state, action)
    return new_state, StepMetadata(
        reward = new_state.reward,
        action_mask=jnp.ones(env.n_actions, dtype=jnp.bool_),
        terminated=new_state.terminated,
        step = new_state._step_count
    )

def init_fn(key):
    state = env.init(key, R=0, N=1)
    return state, StepMetadata(
        reward=state.reward,
        action_mask=jnp.ones(env.n_actions, dtype=jnp.bool_),
        terminated=state.terminated,
        step=0
    )

# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
from core.networks.transformer import Config, GPTConfig, Predictor

encoder_config = Config(
    block_size=4*max_relator_length,
    vocab_size=8, # {-2, -1, 0, 1, 2} + 2 = 5 rounded up to 2**3
    n_layer=1,
    n_head=4,
    n_embd=64,
    dropout=0.0,
    bias=True,
    causal=False,
    use_einsum=True,
    n_segments=4
)

decoder_config = Config(
    block_size=max_relator_length,
    vocab_size=16, # {0:pad, 1-12:actions, 13:start, 14:end} = 15 rounded up to 2**4
    n_layer=1,
    n_head=4,
    n_embd=64,
    dropout=0.0,
    bias=True,
    causal=True,
    use_einsum=True,
    n_value_head=128,
    n_quantiles=1
)

config = GPTConfig(encoder_config, decoder_config)
predictor = Predictor(config)

def state_to_nn_input(state):
    return state.observation + 2

# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS

az_evaluator = AlphaZero(MCTS)(
    eval_fn = make_nn_eval_fn(predictor, state_to_nn_input),
    num_iterations = 32,
    max_nodes = 40,
    branching_factor = env.n_actions,
    action_selector = PUCTSelector(),
    temperature = 1.0,
    dirichlet_alpha = 0.3,
    dirichlet_epsilon = 0.25
)

az_evaluator_test = AlphaZero(MCTS)(
    eval_fn = make_nn_eval_fn(predictor, state_to_nn_input),
    num_iterations = 64,
    max_nodes = 80,
    branching_factor = env.n_actions,
    action_selector = PUCTSelector(),
    temperature = 0.0
)

# -----------------------------------------------------------------------------
# Replay Memory Buffer
# -----------------------------------------------------------------------------
from core.memory.replay_memory import EpisodeReplayBuffer

replay_memory = EpisodeReplayBuffer(capacity=100)

# -----------------------------------------------------------------------------
# Trainer Initialization
# -----------------------------------------------------------------------------
from functools import partial
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
import optax

trainer = Trainer(
   batch_size = 2,
   train_batch_size = 16,
   warmup_steps = 0,
   collection_steps_per_epoch = 8,
   train_steps_per_epoch = 4,
   nn = predictor,
   loss_fn = partial(az_default_loss_fn, l2_reg_lambda=0.0),
   optimizer = optax.adam(1e-3),
   evaluator = az_evaluator,
   memory_buffer = replay_memory,
   max_episode_steps = 4,
   env_step_fn = step_fn,
   env_init_fn = init_fn,
   state_to_nn_input_fn = state_to_nn_input,
   testers = [],
   #wandb_project_name='alpha_ac',
   bootstrap_from_mcts=False
)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
output = trainer.train_loop(seed=1, num_epochs=4, eval_every=1)