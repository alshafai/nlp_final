import json
import numpy as np
import torch
import torch.nn.functional as F

def get_scores(eval_pred_file, as_probs= False):

    predicted_scores = []
    with open(eval_pred_file, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_obj = json.loads(line)
            # Extract "predicted_scores"
            if "predicted_scores" in json_obj:
                predicted_scores.append(json_obj["predicted_scores"])

    # Convert to a PyTorch tensor
    predicted_scores_tensor = torch.tensor(predicted_scores)
    if as_probs:
        softmaxed_tensor = F.softmax(predicted_scores_tensor, dim=1)
        return softmaxed_tensor.tolist()
    else:
        return predicted_scores_tensor.tolist()
    
def add_score(example, predicted_scores, idx):
    example['predicted_scores'] = predicted_scores[idx]
    return example