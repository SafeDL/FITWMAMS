import inspect

from bayesian_ma_idm.src.full_evaluation import evaluate_full_population
from bayesian_ma_idm.src.full_model import fit_full_population


def test_class_conditional_population_interfaces_are_explicit() -> None:
    assert "training_vehicle_classes" in inspect.signature(fit_full_population).parameters
    assert "evaluation_vehicle_classes" in inspect.signature(evaluate_full_population).parameters
