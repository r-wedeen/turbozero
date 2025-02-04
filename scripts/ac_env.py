import jax.numpy as jnp
import numpy as np
import jax
from jax import jit, lax, debug
from functools import partial
import chex
from chex import dataclass

@dataclass
class State:
    observation: chex.Array # concatenated current and goal states; length = 4*max_relator_length
    reward: float
    terminated: bool
    _step_count: int

class ACEnv:
    def __init__(self, max_relator_length=128):
        
        self.action_methods = (
            self._concatenate1,
            self._concatenate2, 
            lambda s: self._conjugate1(s, 1), 
            lambda s: self._conjugate1(s, -1),
            lambda s: self._conjugate1(s, 2),
            lambda s: self._conjugate1(s, -2),
            lambda s: self._conjugate2(s, 1),
            lambda s: self._conjugate2(s, -1),
            lambda s: self._conjugate2(s, 2),
            lambda s: self._conjugate2(s, -2),
            self._invert1, 
            self._invert2
        )

        self.inverse_action_methods = (
            self._concatenate1i,
            self._concatenate2i,
            lambda s: self._conjugate1(s, -1), 
            lambda s: self._conjugate1(s, 1),
            lambda s: self._conjugate1(s, -2),
            lambda s: self._conjugate1(s, 2),
            lambda s: self._conjugate2(s, -1),
            lambda s: self._conjugate2(s, 1),
            lambda s: self._conjugate2(s, -2),
            lambda s: self._conjugate2(s, 2),
            self._invert1,
            self._invert2
        )

        self.n_actions = len(self.action_methods)
        self.max_relator_length = max_relator_length
        self.state_shape = (2*max_relator_length,)

    # ------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------

    def init(self, key, R=50, N=50) -> State:
        """Initialize a State object.
        Random walk from ((1,), (2,)) to goal with R steps, then to start with N steps. 
        If start == goal then perturb."""
        def random_walk(arr, num_steps):
    
            def random_step(i, carry):
                key, cur_arr = carry
                _, subkey = jax.random.split(key)
                action = jax.random.choice(subkey, jnp.arange(self.n_actions))
                next_arr = jax.lax.switch(action, self.action_methods, cur_arr)
                return subkey, next_arr
            
            init_carry = (key, arr)
            _, final_arr = lax.fori_loop(0, num_steps, random_step, init_carry)
            return final_arr
        
        def perturb(arr):
            _, subkey = jax.random.split(key)
            actions = jax.random.permutation(subkey, jnp.arange(self.n_actions))
            def body_fn(i, cur_arr):
                return jax.lax.cond(
                    jnp.all(arr == cur_arr), 
                    lambda x: jax.lax.switch(actions[i], self.action_methods, x), 
                    lambda x: x, 
                    cur_arr
                )
            return lax.fori_loop(0, self.n_actions, body_fn, arr)

        terminal_arr = jnp.zeros(2*self.max_relator_length, dtype=jnp.int32)
        terminal_arr = terminal_arr.at[0].set(1)
        terminal_arr = terminal_arr.at[self.max_relator_length].set(2)
        goal_arr = random_walk(terminal_arr, R)
        start_arr = random_walk(goal_arr, N)
        start_arr = lax.cond(jnp.all(start_arr == goal_arr), perturb, lambda arr: arr, start_arr)
        return State(
            observation=jnp.concatenate([start_arr, goal_arr]),
            reward=jnp.float32(0.0),
            terminated=jnp.all(start_arr == goal_arr),
            _step_count=jnp.int32(0)
        )
    
    def step(self, state: State, action: int) -> State:
        cur_arr, goal_arr = jnp.split(state.observation, 2)
        next_arr = self.transition(cur_arr, action)
        next_observation = jnp.concatenate([next_arr, goal_arr])
        return State(
            observation=next_observation,
            reward=jnp.float32(-1.0),
            terminated=jnp.all(next_arr == goal_arr),
            _step_count=state._step_count + 1,
        )
    
    # ------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------

    @partial(jit, static_argnames=['self'])
    def transition(self, state_arr, action, inverse=False):
        return lax.cond(
            inverse,
            lambda x: lax.switch(action, self.inverse_action_methods, x),
            lambda x: lax.switch(action, self.action_methods, x),
            operand=state_arr
        )

    def _concatenate1(self, state):
        lax.cond(
            False, #jnp.sum(state!=0) > self.max_relator_length, 
            lambda: debug.print(f"len(r1+r2) > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r1, r2 = jnp.split(state, 2)
        r1_num_nonzero = jnp.sum(r1 != 0)
        result = jnp.zeros(2*self.max_relator_length, dtype=state.dtype)
        result = result.at[:self.max_relator_length].set(r1)
        result = lax.dynamic_update_slice(result, r2, (r1_num_nonzero,))
        result = result.at[self.max_relator_length:].set(r2)
        return self._reduce(result)

    def _concatenate1i(self, state):
        return self._invert2(self._concatenate1(self._invert2(state)))

    def _concatenate2(self, state):
        lax.cond(
            #jnp.sum(state!=0) > self.max_relator_length, 
            False,
            lambda: debug.print(f"len(r1+r2) > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r1, r2 = jnp.split(state, 2)
        r2_num_nonzero = jnp.sum(r2 != 0)
        temp = jnp.zeros(3*self.max_relator_length, dtype=state.dtype)
        temp = temp.at[0:self.max_relator_length].set(r1)
        temp = temp.at[self.max_relator_length:2*self.max_relator_length].set(r2)
        temp = lax.dynamic_update_slice(temp, r1, (self.max_relator_length + r2_num_nonzero,))
        result = temp[:2*self.max_relator_length]
        return self._reduce(result)

    def _concatenate2i(self, state):
        return self._invert1(self._concatenate2(self._invert1(state)))

    def _conjugate1(self, state, i):
        [r1, r2] = jnp.split(state, 2)
        lax.cond(
            False, #jnp.sum(r1!=0) + 2 > self.max_relator_length, 
            lambda: debug.print(f"len(r1)+2 > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r1_zero_ix = jnp.where(r1 == 0, size=1, fill_value=-1)[0][0]
        r1 = r1.at[r1_zero_ix].set(-i)
        r1 = jnp.roll(r1, 1)
        r1 = r1.at[0].set(i)
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _conjugate2(self, state, i): 
        [r1, r2] = jnp.split(state, 2)
        lax.cond(
            False, #jnp.sum(r2!=0) + 2 > self.max_relator_length, 
            lambda: debug.print(f"len(r2)+2 > max_relator_length={self.max_relator_length}"),
            lambda: None
        )
        r2_num_nonzero = jnp.sum(r2 != 0)
        r2 = r2.at[r2_num_nonzero].set(-i)
        r2 = jnp.roll(r2, 1)
        r2 = r2.at[0].set(i)
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _invert1(self, state):  
        [r1, r2] = jnp.split(state, 2)
        r1_num_zeros = jnp.sum(r1 == 0)
        r1 = jnp.roll(r1, r1_num_zeros)
        r1 = -r1[::-1]
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _invert2(self, state):
        [r1, r2] = jnp.split(state, 2)
        r2_num_zeros = jnp.sum(r2 == 0)
        r2 = jnp.roll(r2, r2_num_zeros)
        r2 = -r2[::-1]
        result = jnp.concatenate([r1, r2])
        return self._reduce(result)

    def _reduce(self, state):
        [r1, r2] = jnp.split(state, 2)
        
        def reduce_relator(r):
            r = jnp.concatenate([r, jnp.array([0, 0])])
            ix = jnp.where(r[:-1] + r[1:] == 0, size=1, fill_value=-1)[0][0]
            r = jnp.roll(r, -ix)
            r = r[2:]
            r = jnp.roll(r, ix)
            return r
        
        def cond_fn(carry):
            r, new_r = carry
            return jnp.any(r != new_r)

        def body_fn(carry):
            _, r = carry
            new_r = reduce_relator(r)
            return r, new_r

        init_val = (jnp.zeros_like(r1), r1)
        (_, r1) = lax.while_loop(cond_fn, body_fn, init_val)
        init_val = (jnp.zeros_like(r2), r2)
        (_, r2) = lax.while_loop(cond_fn, body_fn, init_val)
        return jnp.concatenate((r1, r2))