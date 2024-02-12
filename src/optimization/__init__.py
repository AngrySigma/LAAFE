from src.optimization.models.model_training import (
    CatboostClassifierModel,
    LinearClassifierModel,
    RandomForestClassifierModel,
    SVMClassifierModel,
)

MODELS = {
    "catboost_classifier": CatboostClassifierModel,
    "linear_classifier": LinearClassifierModel,
    "svm_classifier": SVMClassifierModel,
    "random_forest_classifier": RandomForestClassifierModel,
}
