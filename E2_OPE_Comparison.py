from sklearn.linear_model import LogisticRegression
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_sparse_reward_function
)
from obp.policy import IPWLearner
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust
)

dataset = SyntheticBanditDataset(
    n_actions=7,
    dim_context=5,
    beta=1.0,
    reward_type="binary",
    reward_function=logistic_sparse_reward_function,
    random_state=12345,
)
n_rounds_train, n_rounds_test = 500000, 500000
bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_train)
bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds_test)
ipw_lr = IPWLearner(
    n_actions=dataset.n_actions,
    base_classifier=LogisticRegression(C=100, random_state=12345)
)

ipw_lr.fit(
    context=bandit_feedback_train["context"],
    action=bandit_feedback_train["action"],
    reward=bandit_feedback_train["reward"],
    pscore=bandit_feedback_train["pscore"]
)

action_dist_ipw_lr = ipw_lr.predict(context=bandit_feedback_test["context"])
regression_model = RegressionModel(
    n_actions=dataset.n_actions,
    action_context=dataset.action_context,
    base_model=LogisticRegression(random_state=12345),
)
estimated_rewards_by_reg_model = regression_model.fit_predict(
    context=bandit_feedback_test["context"],
    action=bandit_feedback_test["action"],
    reward=bandit_feedback_test["reward"],
    n_folds=3,
    random_state=12345,
)
ope = OffPolicyEvaluation(
    bandit_feedback=bandit_feedback_test,
    ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()]
)

estimated_policy_value_a, estimated_interval_a = ope.summarize_off_policy_estimates(
    action_dist=action_dist_ipw_lr,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
)
print(estimated_interval_a, '\n')

ope.visualize_off_policy_estimates(
    action_dist=action_dist_ipw_lr,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    n_bootstrap_samples=1000,
    random_state=12345,
)

relative_ee_for_ipw_lr = ope.summarize_estimators_comparison(
    ground_truth_policy_value=dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=action_dist_ipw_lr,
    ),
    action_dist=action_dist_ipw_lr,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    metric="relative-ee",
)

print(relative_ee_for_ipw_lr)