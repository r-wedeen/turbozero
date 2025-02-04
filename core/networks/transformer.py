# TODO: Flash Attention for JAX?

import jax
import optax
import flax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass
from scripts.ac_env import State

# ----------------- Configs -----------------
@dataclass
class Config:
    block_size: int = 4 * 128 # E: max_relator_length=128, D: max_action_path_length=128
    vocab_size: int = 5 # E: {-2, -1, 0, 1, 2} + 2 ==> pad=2, D: {0:pad, 1-12:actions, 13:start, 14:end}
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal: bool = False
    use_einsum: bool = True
    n_segments: int = 4 # ENCODER ONLY
    n_value_head: int = 128 # DECODER ONLY
    n_quantiles: int = 1 # DECODER ONLY
    

@dataclass
class GPTConfig:
    encoder: Config
    decoder: Config


# ----------------- Layer Norm -----------------
class LayerNorm(nn.Module):
    features: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm(
            epsilon=1e-5,
            use_bias=self.bias,
            use_scale=True
        )(x)

# ----------------- Attentions -----------------
class SelfAttention(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, y, train=False, rng1=None, rng2=None):
        assert self.config.n_embd % self.config.n_head == 0
       
        B, T, C = y.shape
        
        q, k, v = jnp.split(nn.Dense(self.config.n_embd * 3, name="c_attn")(y), 3, axis=-1)
        k = k.reshape(B, T, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        q = q.reshape(B, T, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        
        att = (jnp.einsum('bhts,bhrs->bhtr', q, k, optimize=True) if self.config.use_einsum else jnp.matmul(q, k.swapaxes(-2, -1))) * (1.0 / jnp.sqrt(k.shape[-1]))
        if self.config.causal:
            mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))
            att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(self.config.dropout, name='attn_dropout', deterministic=not train)(att, rng=rng1)
        
        y = jnp.einsum('bhts,bhsq->bhtq', att, v, optimize=True) if self.config.use_einsum else jnp.matmul(att, v)   # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(B, T, C)  
        y = nn.Dense(self.config.n_embd, name='c_proj')(y)
        y = nn.Dropout(self.config.dropout, name='resid_dropout', deterministic=not train)(y, rng=rng2)

        return y
    
class CrossAttention(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, y, train=False, rng1=None, rng2=None):
        assert self.config.n_embd % self.config.n_head == 0

        B, T, C = x.shape
        B, R, _ = y.shape

        q = nn.Dense(C, name="x_q")(x)
        k, v = jnp.split(nn.Dense(2 * C, name="y_kv")(y), 2, axis=-1)
        q = q.reshape(B, T, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) #(B, nh, T, hs)
        k = k.reshape(B, R, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) #(B, nh, R, hs)
        v = v.reshape(B, R, self.config.n_head, C // self.config.n_head).swapaxes(1, 2) #(B, nh, R, hs)

        att = (jnp.einsum('bhts,bhrs->bhtr', q, k, optimize=True) if self.config.use_einsum else jnp.matmul(q, k.swapaxes(-2, -1))) * (1.0 / jnp.sqrt(k.shape[-1])) # (B, nh, T, R)
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(self.config.dropout, name='attn_dropout', deterministic=not train)(att, rng=rng1)

        x = jnp.einsum('bhtr,bhrs->bhts', att, v, optimize=True) if self.config.use_einsum else jnp.matmul(att, v)   # (B, nh, T, R) x (B, nh, R, hs) -> (B, nh, T, hs)
        x = x.swapaxes(1, 2).reshape(B, T, C)  
        x = nn.Dense(self.config.n_embd, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout, name='resid_dropout', deterministic=not train)(x, rng=rng2)

        return x
    
# ----------------- MLPs -----------------
class MLP(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, train=False, rng=None):
        x = nn.Dense(4 * self.config.n_embd, name="c_fc")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_embd, name="c_proj")(x)
        x = nn.Dropout(self.config.dropout, name='resid_dropout', deterministic=not train)(x, rng=rng)
        return x

# ----------------- Encoder -----------------
class EncoderBlock(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, y, train=False, rng1=None, rng2=None, rng3=None):
        y = y + SelfAttention(self.config, name='attn')(nn.LayerNorm(use_bias=self.config.bias, name='ln_1')(y), train=train, rng1=rng1, rng2=rng2)
        y = y + MLP(self.config, name='mlp')(nn.LayerNorm(use_bias=self.config.bias, name='ln_2')(y), train=train, rng=rng3)
        return y

class Encoder(nn.Module):
    config: Config

    def setup(self):
        assert self.config.vocab_size is not None
        assert self.config.block_size is not None

        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd)
        self.wse = nn.Embed(self.config.n_segments, self.config.n_embd)
        self.drop = nn.Dropout(self.config.dropout)
        self.h = [EncoderBlock(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(use_bias=self.config.bias)

    def __call__(self, y, targets=None, train=False, rng=jax.random.key(0)):
        _, t = y.shape
        assert t <= self.config.block_size, f"Cannot forward encoder input of length {t}, block size is only {self.config.block_size}"
        assert t % 4 == 0, "Encoder input must be divisible by 4"
        pos = jnp.arange(t, dtype=jnp.int32)
        seg0 = jnp.zeros(t//4, dtype=jnp.int32)
        seg1 = jnp.ones(t//4, dtype=jnp.int32)
        seg2 = 2 * jnp.ones(t//4, dtype=jnp.int32)
        seg3 = 3 * jnp.ones(t//4, dtype=jnp.int32)
        seg = jnp.concatenate([seg0, seg1, seg2, seg3], axis=0)

        tok_emb = self.wte(y) # (B, T, C)
        pos_emb = self.wpe(pos) # (T, C)
        seg_emb = self.wse(seg) # (T, C)

        rng0, rng1, rng2, rng3 = jax.random.split(rng, 4)
        y = self.drop(tok_emb + pos_emb + seg_emb, deterministic=False, rng=rng0)
        for block in self.h:
            y = block(y, train=train, rng1=rng1, rng2=rng2, rng3=rng3)
        y = self.ln_f(y)
        return y

# ----------------- Decoder -----------------
class DecoderBlock(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, y, train=False, rng1=None, rng2=None, rng3=None, rng4=None, rng5=None):
        x = x + SelfAttention(self.config, name='attn')(
            nn.LayerNorm(use_bias=self.config.bias, name='ln_1')(x), 
            train=train,
            rng1=rng1, 
            rng2=rng2
        )
        x = x + CrossAttention(self.config, name='cross_attn')(
            nn.LayerNorm(use_bias=self.config.bias, name='ln_2')(x), 
            y, 
            train=train, 
            rng1=rng3, 
            rng2=rng4
        )
        x = x + MLP(self.config, name='mlp')(
            nn.LayerNorm(use_bias=self.config.bias, name='ln_3')(x), 
            train=train, 
            rng=rng5
        )
        return x
    
class ValueHead(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Dense(self.config.n_value_head, name="c_fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_value_head, name="c_fc2")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_value_head, name="c_fc3")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_quantiles, name="c_proj")(x)
        return x
    
class Decoder(nn.Module):
    config: Config

    def setup(self):
        assert self.config.vocab_size is not None
        assert self.config.block_size is not None

        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.dropout)
        self.h = [DecoderBlock(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(use_bias=self.config.bias)
        self.value_head = ValueHead(self.config)

    def __call__(self, x, y, targets=None, train=False, rng=jax.random.key(0)):
        _, t = x.shape
        assert t <= self.config.block_size, f"Cannot forward decoder input of length {t}, block size is only {self.config.block_size}"

        pos = jnp.arange(t, dtype=jnp.int32)
        tok_emb = self.wte(x) # (B, T, C)
        pos_emb = self.wpe(pos) # (T, C)

        rng0, rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 6)
        x = self.drop(tok_emb + pos_emb, deterministic=False, rng=rng0)
        for block in self.h:
            x = block(x, y, train=train, rng1=rng1, rng2=rng2, rng3=rng3, rng4=rng4, rng5=rng5)
        x = self.ln_f(x)

        # Policy head:
        policy_logits = self.wte.attend(x) # (B, T, vocab_size); weight tying
        # Value head:
        x_v = x[:, 0, :] # (B, C)
        value_quantiles = self.value_head(x_v, train=train) # (B, Q)

        return policy_logits, value_quantiles

# ----------------- GPT -----------------
class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        self.encoder = Encoder(self.config.encoder)
        self.decoder = Decoder(self.config.decoder)

    def __call__(self, x, y, targets=None, train=False, rng=jax.random.key(0)):
        rng0, rng1 = jax.random.split(rng)
        y = self.encoder(y, train=train, rng=rng0)
        policy_logits, value_quantiles = self.decoder(x, y, targets=targets, train=train, rng=rng1)
        return policy_logits, value_quantiles
    
    @staticmethod
    def get_num_params(params, non_embedding=True):
        n_params = sum(p.size for p in jax.tree.leaves(params))
        if non_embedding:
            encoder_wpe_params = params['encoder']['wpe']['embedding']
            encoder_wse_params = params['encoder']['wse']['embedding']
            decoder_wpe_params = params['decoder']['wpe']['embedding']
            n_params -= encoder_wpe_params.size + encoder_wse_params.size + decoder_wpe_params.size
        return n_params
    
    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        # Only implement weight decay on weights that are involved in matmul.
        # Reference: https://stats.stackexchange.com/questions/576463/why-not-perform-weight-decay-on-layernorm-embedding
        label_fn = lambda path, value: "no_decay" if (value.ndim < 2) or ('embedding' in path) else "decay"

        # Create optimization groups
        decay_opt = optax.adamw(learning_rate, weight_decay=weight_decay, b1=betas[0], b2=betas[1])
        nodecay_opt = optax.adam(learning_rate, b1=betas[0], b2=betas[1])

        tx = optax.multi_transform({
            'decay': decay_opt,
            'no_decay': nodecay_opt
        }, flax.traverse_util.path_aware_map(label_fn, params))

        return tx
    
# ----------------- Predictor -----------------
class Predictor(nn.Module):
    config: GPTConfig

    def setup(self):
        self.transformer = GPT(self.config)

    def __call__(self, x, train=False, rng=jax.random.key(0)):
        batch_size = x.shape[0]
        x_d = jnp.full((batch_size, 1), 13, dtype=jnp.int32)
        policy_logits, value_quantiles = self.transformer(x_d, x, train=train, rng=rng)
        action_logits = policy_logits[:, 0, 1:13]
        policy = jax.nn.softmax(action_logits)
        return policy, value_quantiles

class SampledPredictor(nn.Module):
    config: GPTConfig

    def setup(self):
        self.transformer = GPT(self.config)

    def __call__(self, 
            x, 
            train=False, 
            rng=jax.random.key(0), 
            move_length=1, 
            temperature=1.0, 
            env=None,
            branching_factor=12
        ):
        """
        Args:
            - `x`: encoder input
            - `train`: whether to train the model
            - `rng`: random number generator (for dropout and sampling actions)
            - `move_length`: number of actions in a move
            - `temperature`: temperature for the softmax function
            - `env`: if provided, step environment with each action/token prediction, otherwise fly blind
        """
        batch_size = x.shape[0]
        # start token decoder input
        x_d = jnp.full((batch_size, 1), 13, dtype=jnp.int32)
        # generate composite move comprised of `move_length` actions
        pred_policy = {}
        empirical_policy = {}
        for i in range(branching_factor):
            move = jnp.zeros((batch_size, move_length), dtype=jnp.int32)
            prob = jnp.ones((batch_size, 1))
            for j in range(move_length):
                rng1, rng2 = jax.random.split(rng)
                policy_logits, value_quantiles = self.transformer(x_d, x, train=train, rng=rng1)
                action_indices = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
                action_logits = policy_logits[:, 0, action_indices]
                action_probs = jax.nn.softmax(action_logits / temperature)
                action = jax.random.categorical(rng2, action_probs, shape=(batch_size, 1))
                if action == 14: break
                move = move.at[:, j].set(action)
                prob = prob * action_probs[:, action]
                if env is not None:
                    state = State(
                        observation=(x - 2), # x is the encoder input = [start_arr, goal_arr] + 2
                        reward=jnp.float32(0.0),
                        terminated=False,
                        _step_count=jnp.int32(0)
                    )
                    state = env.step(state, action)
                    x = state.observation + 2 # x is the encoder input = [start_arr, goal_arr] + 2
                if env is None:
                    x_d = jnp.concatenate([x_d, action], axis=1)
            pred_policy[move] = prob
            empirical_policy[move] = empirical_policy.get(move, 0) + 1
        return pred_policy, empirical_policy, value_quantiles
    