def local_explainer_functions():

    try:
        from .local_explanability.LIME.limeExplainer import lime_explainer_function
        print("\nStarting Lime Explainer \n")
        lime_explainer_function()
        print("\nLime Explainer stage successful\n")
    except Exception as e:
        print(e)
        print("\nError in Lime Explainer stage\n")

    try:
        from .local_explanability.DeepExplainMain.deepExplainerMethod import deep_explainer
        print("\nStarting Deep Explainer \n")
        deep_explainer()
        print("\nDeep Explainer stage successful\n")
    except Exception as e:
        print(e)
        print("\nError in Deep Explainer stage\n")

    try:
        from .local_explanability.log_return_explain.log_return_explainer import log_return_explain_function
        print("\nStarting log return explain Explainer \n")
        log_return_explain_function()
        print("\nlog return explain stage successful\n")
    except Exception as e:
        print(e)
        print("\nError in log return explain stage\n")

# local_explainer_functions()