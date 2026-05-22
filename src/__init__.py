
                                                                          
from .importance import compute_importance, l2_normalize, importance_to_dict
from .tuning import freeze_gates, set_gates_constant, update_gates_from_importance
from .dropping import evaluate_all_strategies
from .model_utils import load_dataset, build_model, evaluate_model
