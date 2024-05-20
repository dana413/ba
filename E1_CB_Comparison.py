import numpy as np
from sklearn.linear_model import LogisticRegression
from obp.policy.offline import IPWLearner
from obp.dataset import logistic_sparse_reward_function
from obp.policy import EpsilonGreedy, LinEpsilonGreedy, LinUCB, LinTS, LogisticUCB, LogisticTS
from sklearn.ensemble import RandomForestClassifier as RandomForest
from obp.simulator.simulator import BanditEnvironmentSimulator, BanditPolicySimulator
import seaborn as sns
import pandas as pd

experiment_round_size = [1000, 10000, 100000]
n_runs_per_round_size = 20
n_actions = 7
dim_context = 5

bandit_policies = [
    (LinEpsilonGreedy, {"dim": dim_context, "n_actions": n_actions, "epsilon": 0.1, "random_state": 12345}),
    (LinEpsilonGreedy, {"dim": dim_context, "n_actions": n_actions, "epsilon": 0.5, "random_state": 12345}),
    (LinUCB, {"dim": dim_context, "n_actions": n_actions, "epsilon": 0.1, "random_state": 12345}),
    (LinUCB, {"dim": dim_context, "n_actions": n_actions, "epsilon": 0.5, "random_state": 12345}),
    (LinTS, {"dim": dim_context, "n_actions": n_actions, "random_state": 12345}),
    (EpsilonGreedy, {"n_actions": n_actions, "epsilon": 0.1, "random_state": 12345}),
    (LogisticTS, {"dim": dim_context, "n_actions": n_actions, "random_state": 12345}),
      (LogisticUCB, {"dim": dim_context, "n_actions": n_actions, "epsilon": 0.1, "random_state": 12345}),
    (LogisticUCB, {"dim": dim_context, "n_actions": n_actions, "epsilon": 0.5, "random_state": 12345})
]

dummy_policies = [policy_class(**args) for policy_class, args in bandit_policies]
train_rewards = {policy.policy_name: [] for policy in dummy_policies}
eval_rewards = {f"{IPWLearner.__name__} {policy.policy_name}": [] for policy in dummy_policies}
train_rewards["n_rounds"] = []
eval_rewards["n_rounds"] = []

env = BanditEnvironmentSimulator(
    n_actions=7,
    dim_context=5,
    reward_type="binary",
    reward_function=logistic_sparse_reward_function,
    random_state=12345,
)

for n_rounds in experiment_round_size:
    for experiment in range(n_runs_per_round_size):
        training_bandit_batch = env.next_bandit_round_batch(n_rounds=n_rounds)
        evaluation_bandit_batch = env.next_bandit_round_batch(n_rounds=n_rounds)
        for policy_class, args in bandit_policies:
            policy = policy_class(**args)

            training_simulator = BanditPolicySimulator(policy=policy)

            training_simulator.steps(batch_bandit_rounds=training_bandit_batch)
            train_rewards[policy.policy_name].append(training_simulator.total_reward)

            propensity_model = LogisticRegression(random_state=12345, max_iter=1000000000000)
            propensity_model.fit(training_simulator.contexts, training_simulator.selected_actions)
            pscores = propensity_model.predict_proba(training_simulator.contexts)

            ipw_learner = IPWLearner(n_actions=env.n_actions,
                                     base_classifier=RandomForest(n_estimators=30, min_samples_leaf=10, random_state=12345))

            ipw_learner.fit(
                context=training_simulator.contexts,
                action=training_simulator.selected_actions,
                reward=training_simulator.obtained_rewards,
                pscore=np.choose(training_simulator.selected_actions, pscores.T)
            )
            eval_action_dists = ipw_learner.predict(
                context=evaluation_bandit_batch.context
            )

            eval_rewards[f"{ipw_learner.policy_name} {policy.policy_name}"].append(
                np.sum(eval_action_dists.squeeze(axis=2) * evaluation_bandit_batch.rewards)
            )

        train_rewards["n_rounds"].append(n_rounds)
        eval_rewards["n_rounds"].append(n_rounds)


def plot_average_reward_per_n_rounds(rewards):
    rewards_pd = pd.DataFrame(rewards)
    rewards_pd = pd.melt(rewards_pd, ['n_rounds'])
    rewards_pd["average reward"] = rewards_pd["value"] / rewards_pd["n_rounds"]

    plot = sns.lineplot(data=rewards_pd, x="n_rounds", y="average reward", style="variable", hue="variable", markers=True, dashes=False)
    plot.set(xscale='log')
    plot.legend(bbox_to_anchor=(1.1, 1.05))

plot_average_reward_per_n_rounds(eval_rewards)