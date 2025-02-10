import torch

def get_embedding_size(model_path):
    """
    Loads the model and returns its embedding size.
    
    Supports:
    - Full model saving (with embedding layer as an attribute).
    - State dictionary (PyTorch models saved using `model.state_dict()`).
    - Checkpoint files that contain 'embedding_size' or an embedding layer.
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # If the checkpoint is a dictionary, check for possible keys
        if isinstance(checkpoint, dict):
            # Direct embedding size key
            if "embedding_size" in checkpoint:
                return checkpoint["embedding_size"]

            # If the model is stored as 'state_dict'
            state_dict = checkpoint.get("state_dict", checkpoint)  

            # Check for an embedding layer dynamically
            for key, value in state_dict.items():
                if any(x in key.lower() for x in ["embedding", "projection", "linear"]):
                    return value.shape[-1]

        # If the checkpoint is a full model, check for embedding layer
        if hasattr(checkpoint, "embedding"):
            return checkpoint.embedding.weight.shape[-1]

        return "❌ Could not determine embedding size!"
    
    except Exception as e:
        return f"❌ Error loading model: {e}"
