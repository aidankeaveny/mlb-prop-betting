import torch
import numpy as np
import os

def load_ensemble_and_predict(X_input_np, device, model_class, configs, model_dir, feature_count):
    """
    Load and apply an ensemble of MLP models to pre-scaled input data.

    Args:
        X_input_np (np.ndarray): Scaled input data of shape (N, D).
        device (torch.device): 'cpu' or 'cuda' or 'mps'.
        model_class (nn.Module): Model class, e.g., PlayerMLP.
        configs (list): List of 5 config tuples used to initialize models.
        model_dir (str): Path to directory with saved model weights.
        feature_count (int): Number of input features expected by the model.

    Returns:
        np.ndarray: Ensemble-averaged predictions.
    """
    X_tensor = torch.tensor(X_input_np, dtype=torch.float32).to(device)

    ensemble_preds = []
    for i, cfg in enumerate(configs):
        hidden_dims, lr, epochs, batch_size, dropout, val_split, activation, scheduler_type = cfg

        model = model_class(
            input_dim=feature_count,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation
        ).to(device)

        model_path = os.path.join(model_dir, f"mlp_model_{i}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().flatten()
            ensemble_preds.append(preds)

    return np.mean(ensemble_preds, axis=0)
