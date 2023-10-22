import highway_env.envs
# Hide pygame support prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# Import the envs module so that envs register themselves
from gymnasium.envs.registration import register

def register_highway_envs():
    """Import the envs module so that envs register themselves."""

    register(
    id='merge_in-v0',
    entry_point='highway_env.envs:MergeInEnv',
)

register(
    id='merge_in-v1',
    entry_point='highway_env.envs:MergeInEnvReward2',
)

register(
    id='merge_in-v2',
    entry_point='highway_env.envs:DiscreteMergeInEnvReward2',
)

register(
    id='merge_in-v3',
    entry_point='highway_env.envs:DiscreteMergeInEnvReward1',
)