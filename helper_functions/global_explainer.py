def global_explainer_functions():

    try:
        from .global_explanability.xaiExplanability.xai_explain import xai_explainer
        print("\nStarting XAI Explainer \n")
        xai_explainer()
        print("\nXAI Explainer stage successful\n")
    except Exception as e:
        print(e)
        print("\nError in XAI Explainer stage\n")

    try:
        from .global_explanability.ALE_explain.ALE_explainer import ALE_explainer_function
        print("\nStarting ALE Explainer \n")
        ALE_explainer_function()
        print("\nALE Explainer stage successful\n")
    except Exception as e:
        print(e)
        print("\nError in ALE Explainer stage\n")

    try:
        from .global_explanability.linearity_measure.linearity_measure_explainer2 import linearity_measure_explainer_function
        print("\nStarting linearity measure \n")
        linearity_measure_explainer_function()
        print("\nlinearity measure stage successful\n")
    except Exception as e:
        print(e)
        print("\nError in linearity measure stage\n")

    # try:
    #     from .global_explanability.SHAP_global.shap_global_explainer import shap_global_explainer_function
    #     print("\nStarting SHAP global Explainer \n")
    #     shap_global_explainer_function()
    #     print("\nSHAP global Explainer stage successful\n")
    # except Exception as e:
    #     print(e)
    #     print("\nError in SHAP global Explainer stage\n")

# global_explainer_functions()