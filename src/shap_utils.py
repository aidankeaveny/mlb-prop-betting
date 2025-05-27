import shap
import pandas as pd
import numpy as np
import torch

def explain_shap(model, X_test, feature_names):
    explainer = shap.Explainer(lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(), X_test, feature_names=feature_names)
    shap_values = explainer(X_test, max_evals=2000)
    return shap_values

def get_top_shap_features(shap_values, feature_names, top_n=20):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': feature_names, 'MeanAbsSHAP': mean_abs_shap})
    top_features = shap_df.sort_values("MeanAbsSHAP", ascending=False).head(top_n)['Feature'].tolist()
    return top_features, shap_df
