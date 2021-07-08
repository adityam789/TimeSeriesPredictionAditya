from .local_explanability.LIME.limeExplainer import lime_explainer_function
from .local_explanability.DeepExplainMain.deepExplainerMethod import deep_explainer

def local_explainer_functions():
    lime_explainer_function()
    deep_explainer()