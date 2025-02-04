from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from core.common import GameFrame
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn

class SinglePlayerBaseline(BaseTester):
    """Implements a tester that evaluates an agent against a baseline evaluator in a single-player game."""
