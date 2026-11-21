# Evaluation package for multi-agent coordination.

from evaluation.metrics import (
    calculate_success_rate,
    calculate_mean_completion_time,
    calculate_mean_return,
    calculate_recovery_ratio,
    calculate_interaction,
    calculate_redundant_visits,
)
from evaluation.mutual_information import (
    compute_nearest_target_direction,
    calculate_mutual_information,
    bootstrap_confidence_interval,
    paired_bootstrap_difference,
)
from evaluation.baselines import (
    GreedyHeuristicPolicy,
)
from evaluation.evaluate import (
    load_checkpoint,
    evaluate_agent,
    evaluate_heuristic_baseline,
    evaluate_checkpoint,
    evaluate_conditions,
    format_obs,
)
from evaluation.statistics import (
    compute_mean_and_ci,
    compute_paired_comparison,
    aggregate_experiment_results,
)
