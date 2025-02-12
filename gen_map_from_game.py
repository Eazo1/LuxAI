import json
import time

import numpy as np
import flax
import flax.serialization
from luxai_s3.params import EnvParams
from luxai_s3.state import EnvState, serialize_env_actions, serialize_env_states

import jax
import jax.numpy as jnp

from luxai_s3.env import LuxAIS3Env

def gen_game_map(seed):
    # from luxai_s3.wrappers import RecordEpisode

    # Create the environment
    env = LuxAIS3Env(auto_reset=False)
    env_params = EnvParams(map_type=0, max_steps_in_match=50)

    # Initialize a random key
    key = jax.random.key(seed)

    # Reset the environment
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params=env_params)
    # Take a random action
    key, subkey = jax.random.split(key)
    action = env.action_space(env_params).sample(subkey)
    # Step the environment and compile. Not sure why 2 steps? are needed
    for _ in range(2):
        key, subkey = jax.random.split(key)
        obs, state, reward, terminated, truncated, info = env.step(
            subkey, state, action, params=env_params
        )

    MAP = []
    RELIC_NODES = []
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, params=env_params)
    N = env_params.max_steps_in_match * env_params.match_count_per_episode
    for _ in range(N):
        key, subkey = jax.random.split(key)
        action = env.action_space(env_params).sample(subkey)
        obs, state, reward, terminated, truncated, info = env.step(
            subkey, state, action, params=env_params
        )
        MAP.append(np.array(state.map_features.tile_type))
        RELIC_STEP = []
        for i in range(6):
            if state.relic_nodes_mask[i]:
                x, y = state.relic_nodes[i, :2]
                RELIC_STEP.append(np.array([x,y]))
        RELIC_NODES.append(RELIC_STEP)

    PARAMS = json.dumps(env_params.__dict__) 
    return MAP,RELIC_NODES,PARAMS
