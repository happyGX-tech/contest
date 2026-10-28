"""Thin adapter for the existing AgentManager and TrainingEnv deployment API."""
import numpy as np
from user_agents.base_agent import BaseAgent, AgentType
from scenarios.grid_maps.deployment import compile_actions


class GridDeployAgent(BaseAgent):
    def __init__(self, agent_id, entity_id, init_observation, coordinates, coordinates_hm, plan):
        super().__init__(agent_id, entity_id, AgentType.DEPLOY, init_observation)
        # Validate the entire submission before AgentManager's exception-catching loop.
        self.actions = compile_actions(plan, init_observation['entities'], coordinates, coordinates_hm)
        self.index = 0

    def set_observation(self, observation):
        """Keep the scenario IDs and heights validated at registration."""

    def get_action(self):
        batch = self.actions[self.index:self.index+10]
        self.index += len(batch)
        return np.asarray(batch if batch else [[1,-1,0,0,0]], dtype=np.float64)

    def reset(self):
        super().reset()
        self.index = 0
