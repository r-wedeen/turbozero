
from functools import partial
import os
import shutil
from typing import Any, List, Optional, Tuple

import chex
from chex import dataclass
import flax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import optax
import orbax
import wandb

from core.common import partition, step_env_and_evaluator
from core.evaluators.evaluator import Evaluator
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer, ReplayBufferState
from core.testing.tester import BaseTester, TestState
from core.types import DataTransformFn, EnvInitFn, EnvStepFn, ExtractModelParamsFn, LossFn, StateToNNInputFn, StepMetadata


@dataclass(frozen=True)
class CollectionState:
    """Stores state of self-play episode collection. Persists across generations.
    - `eval_state`: state of the evaluator
    - `env_state`: state of the environment
    - `buffer_state`: state of the replay buffer
    - `metadata`: metadata of the current environment state
    """
    eval_state: chex.ArrayTree
    env_state: chex.ArrayTree
    buffer_state: ReplayBufferState
    metadata: StepMetadata

@dataclass(frozen=True)
class TrainLoopOutput:
    """
    Stores the state of the training loop.
    collection_state is included to access replay memory.
    - `collection_state`: state of self-play episode collection.
    - `train_state`: flax TrainState, holds optimizer state, model params
    - `test_states`: states of testers
    - `cur_epoch`: current epoch num
    """
    collection_state: CollectionState
    train_state: TrainState
    test_states: List[TestState]
    cur_epoch: int


class TrainStateWithBS(TrainState):
    """Custom flax TrainState to handle BatchNorm"""
    batch_stats: chex.ArrayTree
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def extract_params(state: TrainState) -> chex.ArrayTree:
    """Extracts model parameters from TrainState.
    
    Args:
    - `state`: TrainState containing model parameters

    Returns:
    - (chex.ArrayTree): model parameters
    """
    if hasattr(state, 'batch_stats'):
        return {'params': state.params, 'batch_stats': state.batch_stats}
    return {'params': state.params}


class Trainer:
    """Implements a training loop for AlphaZero.
    Maintains state across self-play game collection, training, and testing."""

    def __init__(self,
        batch_size: int,
        train_batch_size: int,
        warmup_steps: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        nn: flax.linen.Module,
        loss_fn: LossFn,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,        
        memory_buffer: EpisodeReplayBuffer,
        max_episode_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        state_to_nn_input_fn: StateToNNInputFn,
        testers: List[BaseTester],
        expert_buffer: Optional[EpisodeReplayBuffer] = None,
        evaluator_test: Optional[Evaluator] = None,
        data_transform_fns: List[DataTransformFn] = [],
        extract_model_params_fn: Optional[ExtractModelParamsFn] = extract_params,
        wandb_project_name: str = "",
        ckpt_dir: str = "/tmp/turbozero_checkpoints",
        max_checkpoints: int = 2,
        num_devices: Optional[int] = None,
        wandb_run: Optional[Any] = None,
        extra_wandb_config: Optional[dict] = None,
        bootstrap_from_mcts: bool = False,
        move_length: int = 1,
        temperature: float = 1.0,
        env: Optional[Any] = None,
    ):
        """ 
        Args:
        - `batch_size`: batch size for self-play games
        - `train_batch_size`: minibatch size for training steps
        - `warmup_steps`: # of steps (per batch) to collect via self-play prior to entering the training loop. 
            - This is used to populate the replay memory with some initial samples
        - `collection_steps_per_epoch`: # of steps (per batch) to collect via self-play in each epoch
        - `train_steps_per_epoch`: # of training steps to take in each epoch
        - `nn`: flax.linen.Module containing configured neural network
        - `loss_fn`: loss function for training (see core.training.loss_fns)
        - `optimizer`: optax optimizer
        - `evaluator`: the `Evaluator` to use during self-play
        - `memory_buffer`: replay memory buffer class, used to store self-play experiences
        - `max_episode_steps`: maximum number of steps in an episode
        - `env_step_fn`: environment step function (env_state, action) -> (new_env_state, metadata)
        - `env_init_fn`: environment initialization function (key) -> (env_state, metadata)
        - `state_to_nn_input_fn`: function to convert environment state to neural network input
        - `testers`: list of testers to evaluate the agent against (see core.testing.tester)
        - `evaluator_test`: (optional) evaluator to use during testing. If not provided, `evaluator` is used.
        - `data_transform_fns`: (optional) list of data transform functions to apply to self-play experiences (e.g. rotation, reflection, etc.)
        - `extract_model_params_fn`: (optional) function to extract model parameters from TrainState
        - `wandb_project_name`: (optional) name of wandb project to log to
        - `ckpt_dir`: directory to save checkpoints
        - `max_checkpoints`: maximum number of checkpoints to keep
        - `num_devices`: (optional) number of devices to use, defaults to jax.local_device_count()
        - `wandb_run`: (optional) wandb run object, will continue logging to this run if passed, else a new run is initialized
        - `extra_wandb_config`: (optional) extra config to pass to wandb
        - `move_length`: (sampled alphazero) number of actions in a move 
        - `temperature`: (sampled alphazero) temperature for the softmax function
        - `env`: (sampled alphazero) if provided, step environment with each action/token prediction, otherwise fly blind
        """
        self.num_devices = num_devices if num_devices is not None else jax.local_device_count()
        # environment
        self.env_step_fn = env_step_fn
        self.env_init_fn = env_init_fn
        self.max_episode_steps = max_episode_steps
        self.template_env_state = self.make_template_env_state()
        # nn
        self.state_to_nn_input_fn = state_to_nn_input_fn
        self.nn = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.extract_model_params_fn = extract_model_params_fn
        # selfplay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.collection_steps_per_epoch = collection_steps_per_epoch
        self.memory_buffer = memory_buffer
        self.expert_buffer = expert_buffer
        self.evaluator_train = evaluator
        self.transform_fns = data_transform_fns
        self.step_train = partial(step_env_and_evaluator,
            evaluator=self.evaluator_train,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=self.max_episode_steps
        )
        self.bootstrap_from_mcts = bootstrap_from_mcts
        # training
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_batch_size = train_batch_size
        # testing
        self.testers = testers
        self.evaluator_test = evaluator_test if evaluator_test is not None else evaluator
        self.step_test = partial(step_env_and_evaluator,
            evaluator=self.evaluator_test,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=self.max_episode_steps
        )
        # checkpoints
        self.ckpt_dir = ckpt_dir
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            ckpt_dir, orbax_checkpointer, options)
        # wandb
        self.wandb_project_name = wandb_project_name
        self.use_wandb = wandb_project_name != ""
        if self.use_wandb:
            if wandb_run is not None:
                self.run = wandb_run
            else:
                self.run = self.init_wandb(wandb_project_name, extra_wandb_config)
        else:
            self.run = None
        # sampled alphazero
        self.move_length = move_length
        self.temperature = temperature
        self.env = env
        # check batch sizes, etc. are compatible with number of devices
        self.check_size_compatibilities()


    def init_wandb(self, project_name: str, extra_wandb_config: Optional[dict]):
        """Initializes wandb run.
        Args: 
        - `project_name`: name of wandb project
        - `extra_wandb_config`: (optional) extra config to pass to wandb
        
        Returns:
        - (wandb.Run): wandb run
        """
        if extra_wandb_config is None:
            extra_wandb_config = {} 
        return wandb.init(
            project=project_name,
            config={**self.get_config(), **extra_wandb_config}
        )
    

    def check_size_compatibilities(self):
        """Checks if batch sizes, etc. are compatible with number of devices.
        Calls check_size_compatibilities on each tester."""

        err_fmt = "Batch size must be divisible by the number of devices. Got {b} batch size and {d} devices."
        # check train batch size
        if self.train_batch_size % self.num_devices != 0:
            raise ValueError(err_fmt.format(b=self.train_batch_size, d=self.num_devices))
        # check collection batch size
        if self.batch_size % self.num_devices != 0:
            raise ValueError(err_fmt.format(b=self.batch_size, d=self.num_devices))
        # check testers 
        for tester in self.testers:
            tester.check_size_compatibilities(self.num_devices)


    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0,))
    def init_train_state(self, key: jax.random.PRNGKey) -> TrainState:
        """Initializes the training state (params, optimizer, etc.) partitions across devices.
        
        Args:
        - `key`: rng

        Returns:
        - (TrainState): initialized training state
        """
        # get template env state
        sample_env_state = self.make_template_env_state()
        # get sample nn input
        sample_obs = self.state_to_nn_input_fn(sample_env_state)
        # initialize nn parameters
        variables = self.nn.init(key, sample_obs[None, ...], train=False)
        params = variables['params']
        # initialize apply_fn
        apply_fn = partial(self.nn.apply, move_length=self.move_length, temperature=self.temperature, env=self.env)
        # handle batchnorm
        if 'batch_stats' in variables:
            return TrainStateWithBS.create(
                apply_fn=apply_fn,
                params=params,
                tx=self.optimizer,
                batch_stats=variables['batch_stats']
            )
        # init TrainState
        return TrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx=self.optimizer,
        )

        
    def get_config(self):
        """Returns a dictionary of the configuration of the trainer. Used for logging/wandb."""
        return {
            'batch_size': self.batch_size,
            'train_batch_size': self.train_batch_size,
            'warmup_steps': self.warmup_steps,
            'collection_steps_per_epoch': self.collection_steps_per_epoch,
            'train_steps_per_epoch': self.train_steps_per_epoch,
            'num_devices': self.num_devices,
            'evaluator_train': self.evaluator_train.__class__.__name__,
            'evaluator_train_config': self.evaluator_train.get_config(),
            'evaluator_test': self.evaluator_test.__class__.__name__,
            'evaluator_test_config': self.evaluator_test.get_config(),
            'memory_buffer': self.memory_buffer.__class__.__name__,
            'memory_buffer_config': self.memory_buffer.get_config(),
        }
    
    
    def collect(self,
        key: jax.random.PRNGKey,
        state: CollectionState,
        params: chex.ArrayTree,
        bootstrap_from_mcts: bool = False
    ) -> CollectionState:
        """
        - Collects self-play data for a single step. 
        - Stores experience in replay buffer.
        - Resets environment/evaluator if episode is terminated.
        
        Args:
        - `key`: rng
        - `state`: current collection state (environment, evaluator, replay buffer)
        - `params`: model parameters
        
        Returns:
        - (CollectionState): updated collection state
        """

        # step environment and evaluator
        eval_output, new_env_state, new_metadata, terminated, truncated, reward = \
            self.step_train(
                key = key,
                env_state = state.env_state,
                env_state_metadata = state.metadata,
                eval_state = state.eval_state,
                params = params
            )
            
        # store experience in replay buffer
        buffer_state = self.memory_buffer.add_experience(
            state = state.buffer_state,
            experience = BaseExperience(
                observation_nn=self.state_to_nn_input_fn(state.env_state),
                policy_mask=state.metadata.action_mask,
                policy_weights=eval_output.policy_weights,
                reward=state.metadata.reward,
                bootstrapped_return=jnp.float32(0.0) # placeholder value; to be updated later
            )
        )

        # apply transforms. NOTE: not used for AC
        for transform_fn in self.transform_fns:
            t_policy_mask, t_policy_weights, t_env_state = transform_fn(
                state.metadata.action_mask,
                eval_output.policy_weights,
                state.env_state
            )
            buffer_state = self.memory_buffer.add_experience(
                state = buffer_state,
                experience = BaseExperience(
                    observation_nn=self.state_to_nn_input_fn(t_env_state),
                    policy_mask=t_policy_mask,
                    policy_weights=t_policy_weights,
                    reward=state.metadata.reward,
                    bootstrapped_return=jnp.float32(0.0),
                )
            )
        
        # assign returns to buffer if episode is terminated: bootstrap from -1 (env resets after termination so don't bootstrap from 0)
        buffer_state = jax.lax.cond(
            terminated,
            lambda s: self.memory_buffer.assign_returns(s, final_value=-1.0),
            lambda s: s,
            buffer_state
        )
        # assign returns to buffer if episode is truncated: bootstrap from predicted value
        value_estimate = jax.lax.cond(
            bootstrap_from_mcts,
            lambda: self.evaluator_train.get_value(eval_output.eval_state),
            lambda: eval_output.root_value
        )
        buffer_state = jax.lax.cond(
            truncated,
            lambda s: self.memory_buffer.assign_returns(s, final_value=value_estimate),
            lambda s: s,
            buffer_state
        )

        state_arr ,goal_arr = jnp.split(state.env_state.observation, 2)
        jax.debug.print("state array: {x}", x=state_arr)
        jax.debug.print("goal array: {x}", x=goal_arr)
        jax.debug.print("episode start idx: {x}", x=state.buffer_state.episode_start_idx)
        jax.debug.print("current idx: {x}", x=state.buffer_state.next_idx)
        jax.debug.print("terminated: {x}", x=terminated)
        jax.debug.print("truncated: {x}", x=truncated)
        jax.debug.print("rewards: {x}", x=buffer_state.buffer.reward.reshape(-1))
        jax.debug.print("bootstrapped returns: {x}", x=buffer_state.buffer.bootstrapped_return.reshape(-1))

        # return new collection state
        return state.replace(
            eval_state=eval_output.eval_state,
            env_state=new_env_state,
            buffer_state=buffer_state,
            metadata=new_metadata
        )

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 4, 5))
    def collect_steps(self,
        key: chex.PRNGKey,
        state: CollectionState,
        params: chex.ArrayTree,
        num_steps: int,
        bootstrap_from_mcts: bool = False
    ) -> CollectionState:
        """Collects self-play data for `num_steps` steps. Mapped across devices.
        
        Args:
        - `key`: rng
        - `state`: current collection state
        - `params`: model parameters
        - `num_steps`: number of self-play steps to collect

        Returns:
        - (CollectionState): updated collection state
        """
        if num_steps > 0:
            collect = partial(self.collect, params=params, bootstrap_from_mcts=bootstrap_from_mcts)
            keys = jax.random.split(key, num_steps)
            return jax.lax.fori_loop(
                0, num_steps, 
                lambda i, s: collect(keys[i], s), 
                state
            )
        return state
    

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0,))
    def one_train_step(self, ts: TrainState, batch: BaseExperience) -> Tuple[TrainState, dict]:
        """Make a single training step.
        
        Args:
        - `ts`: TrainState
        - `batch`: minibatch of experiences 

        Returns:
        - (TrainState, dict): updated TrainState and metrics
        """
        # calculate loss, get gradients
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, (metrics, updates)), grads = grad_fn(ts.params, ts, batch)
        # apply gradients
        grads = jax.lax.pmean(grads, axis_name='d')
        ts = ts.apply_gradients(grads=grads)
        # update batchnorm stats
        if hasattr(ts, 'batch_stats'):
            ts = ts.replace(batch_stats=jax.lax.pmean(updates['batch_stats'], axis_name='d'))
        # return updated train state and metrics
        metrics = {
            **metrics,
            'loss': loss
        }
        return ts, metrics


    def train_steps(self,
        key: chex.PRNGKey,
        collection_state: CollectionState,
        train_state: TrainState,
        num_steps: int
    ) -> Tuple[CollectionState, TrainState, dict]:
        """Performs `num_steps` training steps.
        Each step consists of sampling a minibatch from the replay buffer and updating the parameters.
        
        Args:
        - `key`: rng
        - `collection_state`: current collection state
        - `train_state`: current training state
        - `num_steps`: number of training steps to perform
        
        Returns:
        - (CollectionState, TrainState, dict): 
            - updated collection state
            - updated training state
            - metrics
        """
        # get replay memory buffer 
        buffer_state = collection_state.buffer_state
        
        batch_metrics = []
        
        for _ in range(num_steps):
            step_key, key = jax.random.split(key)
            # sample from replay memory
            batch = self.memory_buffer.sample(buffer_state, step_key, self.train_batch_size)
            # reshape into minibatch
            batch = jax.tree_map(lambda x: x.reshape((self.num_devices, -1, *x.shape[1:])), batch)
            # make training step
            train_state, metrics = self.one_train_step(train_state, batch)
            # append metrics from step
            if metrics:
                batch_metrics.append(metrics)
        # take mean of metrics across all training steps
        if batch_metrics:
            metrics = {k: jnp.stack([m[k] for m in batch_metrics]).mean() for k in batch_metrics[0].keys()}
        else:
            metrics = {}
        # return updated collection state, train state, and metrics
        return collection_state, train_state, metrics
    
    
    def log_metrics(self, metrics: dict, epoch: int, step: Optional[int] = None):
        """Logs metrics to console and wandb.
        
        Args:
        - `metrics`: dictionary of metrics
        - `epoch`: current epoch
        - `step`: current step
        """
        # log to console
        metrics_str = {k: f"{v.item():.4f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {metrics_str}")
        # log to wandb
        if self.use_wandb:
            wandb.log(metrics, step)


    def save_checkpoint(self, train_state: TrainState, epoch: int, **kwargs) -> None:
        """Saves an orbax checkpoint of the training state.
        
        Args:
        - `train_state`: current training state
        - `epoch`: current epoch
        """
        # get params from single device, because
        # params are copied to all devices
        ckpt = {'train_state': jax.tree_map(lambda x: x[0], train_state)}
        # save checkpoint
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})

    def load_train_state_from_checkpoint(self, path_to_checkpoint: str) -> TrainState:
        """Loads a training state from a checkpoint.
        
        Args:
        - `path_to_checkpoint`: path to checkpoint
        
        Returns:
        - (TrainState): loaded training state
        """
        ckpt = self.checkpoint_manager.restore(path_to_checkpoint)
        return ckpt['train_state']
    

    def make_template_env_state(self) -> chex.ArrayTree:
        """Create a template environment state used for initializing data structures 
        that hold environment states to the correct shape.
        
        Returns:
        - (chex.ArrayTree): template environment state
        """
        env_state, _ = self.env_init_fn(jax.random.PRNGKey(0))
        return env_state
    
    
    def make_template_experience(self) -> BaseExperience:
        """Create a template experience used for initializing data structures
        that hold experiences to the correct shape.
        
        Returns:
        - (BaseExperience): template experience
        """
        env_state, metadata = self.env_init_fn(jax.random.PRNGKey(0))
        return BaseExperience(
            observation_nn=self.state_to_nn_input_fn(env_state),
            policy_mask=metadata.action_mask,
            policy_weights=jnp.zeros_like(metadata.action_mask, dtype=jnp.float32),
            reward=jnp.float32(0.0),
            bootstrapped_return=jnp.float32(0.0)
        )
    

    def init_collection_state(self, key: jax.random.PRNGKey, batch_size: int) -> CollectionState:
        """Initializes the collection state (see CollectionState).
        
        Args:
        - `key`: rng
        - `batch_size`: number of parallel environments

        Returns:
        - (CollectionState): initialized collection state
        """
        # make template experience
        template_experience = self.make_template_experience()
        # init buffer state
        buffer_state = self.memory_buffer.init(batch_size, template_experience)
        # init env state
        env_init_key, key = jax.random.split(key)
        env_keys = jax.random.split(env_init_key, batch_size)
        env_state, metadata = jax.vmap(self.env_init_fn)(env_keys)
        # init evaluator state
        eval_state = self.evaluator_train.init_batched(batch_size, template_embedding=self.template_env_state)
        # return collection state
        return CollectionState(
            eval_state=eval_state,
            env_state=env_state,
            buffer_state=buffer_state,
            metadata=metadata
        )
    

    def train_loop(self,
        seed: int,
        num_epochs: int,
        eval_every: int = 1,
        initial_state: Optional[TrainLoopOutput] = None
    ) -> Tuple[CollectionState, TrainState]:
        """Runs the training loop for `num_epochs` epochs. Mostly configured by the Trainer's attributes.
        - Collects self-play episdoes across a batch of environments.
        - Trains the neural network on the collected experiences.
        - Tests the agent on a set of Testers, which evaluate the agent's performance.

        Args:
        - `seed`: rng seed (int)
        - `num_epochs`: number of epochs to run the training loop for
        - `eval_every`: number of epochs between evaluations
        - `initial_state`: (optional) TrainLoopOutput, used to continue training from a previous state

        Returns:
        - (TrainLoopOutput): contains train_state, collection_state, test_states, cur_epoch after training loop
        """
        # init rng
        key = jax.random.PRNGKey(seed)

        # initialize states
        if initial_state:
            collection_state = initial_state.collection_state
            train_state = initial_state.train_state
            tester_states = initial_state.test_states
            cur_epoch = initial_state.cur_epoch
        else:
            cur_epoch = 0
            # initialize collection state
            init_key, key = jax.random.split(key)
            collection_state = partition(self.init_collection_state(init_key, self.batch_size), self.num_devices)

            # initialize train state
            init_key, key = jax.random.split(key)
            init_keys = jnp.tile(init_key[None], (self.num_devices, 1))
            train_state = self.init_train_state(init_keys)
            params = self.extract_model_params_fn(train_state)
            # initialize tester states
            tester_states = []
            for tester in self.testers:
                state = jax.pmap(tester.init, axis_name='d')(params=params)
                tester_states.append(state)
        
        # warmup
        # populate replay buffer with initial self-play games
        collect = jax.vmap(self.collect_steps, in_axes=(1, 1, None, None, None), out_axes=1)
        params = self.extract_model_params_fn(train_state)
        collect_key, key = jax.random.split(key)
        collect_keys = partition(jax.random.split(collect_key, self.batch_size), self.num_devices)
        collection_state = collect(collect_keys, collection_state, params, self.warmup_steps, self.bootstrap_from_mcts)

        # training loop
        while cur_epoch < num_epochs:
            # collect self-play games
            collect_key, key = jax.random.split(key)
            collect_keys = partition(jax.random.split(collect_key, self.batch_size), self.num_devices)
            collection_state = collect(collect_keys, collection_state, params, self.collection_steps_per_epoch, self.bootstrap_from_mcts)
            # train
            train_key, key = jax.random.split(key)
            collection_state, train_state, metrics = self.train_steps(train_key, collection_state, train_state, self.train_steps_per_epoch)
            # log metrics
            collection_steps = self.batch_size * (cur_epoch+1) * self.collection_steps_per_epoch
            self.log_metrics(metrics, cur_epoch, step=collection_steps)

            # test 
            if cur_epoch % eval_every == 0:
                params = self.extract_model_params_fn(train_state)
                for i, test_state in enumerate(tester_states):
                    run_key, key = jax.random.split(key)
                    new_test_state, metrics, rendered = self.testers[i].run(
                        key=run_key, epoch_num=cur_epoch, max_steps=self.max_episode_steps, num_devices=self.num_devices,
                        env_step_fn=self.env_step_fn, env_init_fn=self.env_init_fn, evaluator=self.evaluator_test,
                        state=test_state, params=params)
                        
                    metrics = {k: v.mean() for k, v in metrics.items()}
                    self.log_metrics(metrics, cur_epoch, step=collection_steps)
                    if rendered and self.run is not None:
                        self.run.log({f'{self.testers[i].name}_game': wandb.Video(rendered)}, step=collection_steps)
                    tester_states[i] = new_test_state
            # save checkpoint
            self.save_checkpoint(train_state, cur_epoch)
            # next epoch
            cur_epoch += 1
            
        # return state so that training can be continued!
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            test_states=tester_states,
            cur_epoch=cur_epoch
        )
