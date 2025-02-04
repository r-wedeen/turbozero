import jax
from jax import vmap
import jax.numpy as jnp
import chex
from chex import dataclass

@dataclass(frozen=True)
class BloomFilterState:
    array: chex.Array

class BloomFilter:
    """Bloom filter for fast membership testing. 
    
    Compatible with `jax.jit`, `jax.vmap`, and `jax.pmap`."""

    def __init__(self, 
            num_hashes: int=32, 
            num_words: int=2**27, 
            b1: int=7,
            b2: int=31,
            m: int=4294967291
        ):
        """
        Args:
        - `num_hashes`: number of hash functions
        - `num_words`: number of words in the bloom filter; 
            number of bits = `num_words * (bits_per_word=32)`
        - `b1`: base for the first hash function
        - `b2`: base for the second hash function
        - `m`: modulus for the hash functions
        """
        self.num_hashes = num_hashes
        self.num_words = num_words
        self.bits_per_word = 32
        self.b1 = jnp.uint32(b1)
        self.b2 = jnp.uint32(b2)
        self.m = jnp.uint32(m)

    def hash(self, arr, b, m):
        """Hash a numpy array using a polynomial rolling hash function.

        Args:
        - `arr`: numpy array to hash
        - `b`: base for the polynomial hash function
        - `m`: modulus for the polynomial hash function
        
        Returns:
        - `hash_value`: uint32 hash value of the array"""

        arr = arr.astype(jnp.uint32)
        def hash_fold(carry, x):
            return (carry * b + x) % m, None
        hash_value, _ = jax.lax.scan(hash_fold, jnp.uint32(0), arr)
        return hash_value
    
    def compute_hashes(self, arr: chex.Array) -> chex.Array:
        """Compute multiple hash values for a state.
        
        Args:
        - `arr`: array to hash
        
        Returns:
        - `hashes`: array of `num_hashes` uint32 hashes of `arr`"""

        # Compute two base hash values using different bases
        h1 = self.hash(arr, self.b1, self.m)
        h2 = self.hash(arr, self.b2, self.m)
        # Generate multiple hashes
        hashes = (h1 + jnp.arange(self.num_hashes, dtype=jnp.uint32) * h2) % self.m
        return hashes

    def insert(self, states: chex.Array, filter: BloomFilterState) -> BloomFilterState:
        """Insert states into the bloom filter. 
        
        Assumes `states` has a batch dimension in axis=0.

        Args:
        - `states`: states to insert, shape = (num_states, ...)
        - `bloom_filter`: bloom filter to insert into
        
        Returns:
        - `bloom_filter`: updated bloom filter"""

        indices = vmap(self.compute_hashes)(states).flatten() #(num_states * num_hashes,)
        word_indices = indices // self.bits_per_word #(num_states * num_hashes,)
        bit_positions = indices % self.bits_per_word #(num_states * num_hashes,)
        masks = jnp.left_shift(jnp.uint32(1), bit_positions) #(num_states * num_hashes,)
        return filter.replace(
            array = filter.array.at[word_indices].set(filter.array[word_indices] | masks)
        )

    def query(self, states: chex.Array, filter: BloomFilterState) -> bool:
        """Check if states are present in the bloom filter.

        Assumes `states` has batch dimension in axis=0.
        
        Args:
        - `states`: states to check membership
        - `bloom_filter`: bloom filter to query
        
        Returns:
        - (boolean array): 0: state not in filter, 1: state in filter, shape=(batch_size,)"""

        indices = vmap(self.compute_hashes)(states) #(batch_size, num_hashes)
        word_indices = indices // self.bits_per_word #(batch_size, num_hashes)
        bit_positions = indices % self.bits_per_word #(batch_size, num_hashes)
        masks = jnp.left_shift(jnp.uint32(1), bit_positions) #(batch_size, num_hashes)

        # Retrieve the bits from the bloom_filter
        bits = filter.array[word_indices] & masks #(batch_size, num_hashes)

        # Check if all bits are set
        return jnp.all(bits == 1, axis=1)

    def init(self) -> BloomFilterState:
        """Initialize the bloom filter state.
        
        Returns:
        - `bloom_filter`: initialized bloom filter state"""

        return BloomFilterState(
            array = jnp.zeros(self.num_words, dtype=jnp.uint32)
        )