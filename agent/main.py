# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:06:40 2025

@author: herbi
"""


import json
from argparse import Namespace
from agent import Agent
from agent2 import Agent2
from lux.kit import from_json

### DO NOT REMOVE THE FOLLOWING CODE ###
# store potentially multiple dictionaries as kaggle imports code directly
agent_dict = dict()
agent_prev_obs = dict()

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        if player == "player_0":
            agent_dict[player] = Agent(player, configurations["env_cfg"])
        elif player == "player_1":        
            agent_dict[player] = Agent2(player, configurations["env_cfg"])
    agent = agent_dict[player]
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    while True:
        inputs = read_input()
        
        raw_input = json.loads(inputs)
        
        observation = Namespace(
            **dict(
                step=raw_input["step"],
                obs=raw_input["obs"],
                remainingOverageTime=raw_input["remainingOverageTime"],
                player=raw_input["player"],
                info=raw_input["info"],
            )
        )
      
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))
        


# Run the Lux AI S3 command
#command = ["luxai-s3", "agent/main.py", "agent/main.py", "--output=replay.html"]
#subprocess.run(command)
