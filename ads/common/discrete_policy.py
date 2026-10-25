"""Explicit episode lifecycle for the stateless SB3 DQN and PPO policies."""

from ads.common.randomness import PolicyRandomContext


class DiscretePolicy:
    def __init__(self, model, algorithm):
        self.model, self.algorithm, self.device = model, algorithm, model.device
        self.reset(PolicyRandomContext(algorithm))

    def reset(self, context):
        self.random_context, self.decision = context, 0
        self.model.action_space.seed(context.seed_for(0, "semantic_action"))

    def predict(self, observation, deterministic=True):
        self.model.action_space.seed(
            self.random_context.seed_for(self.decision, "semantic_action")
        )
        with self.random_context.step(
            self.decision, self.device, use="semantic_policy"
        ):
            output = self.model.predict(observation, deterministic=deterministic)
        self.decision += 1
        return output

    def save(self, path):
        self.model.save(path)
