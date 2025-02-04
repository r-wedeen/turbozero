import jax

def true_fn():
    return None

def false_fn():
    jax.debug.print("hello world")
    return None

jax.lax.cond(False, true_fn, false_fn)
