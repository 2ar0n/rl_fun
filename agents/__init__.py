from .reinforce import Reinforce
from .actor_critic import ActorCritic
from .actor_critic_continuous import ActorCriticContinuous
from .ppo import PPO

def create_agent(agent_type: str, env):
    if agent_type == "reinforce":
         return Reinforce(env)
    elif agent_type == "actor-critic":
         return ActorCritic(env)
    elif agent_type == "actor-critic-continuous":
         return ActorCriticContinuous(env)
    elif agent_type == "ppo":
         return PPO(env)
    else:
        raise RuntimeError(f"Agent type {agent_type} not found")
