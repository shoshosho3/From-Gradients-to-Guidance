import torch
import torch.nn.functional as F

def modulate_scores_by_entropy(scores, logits):
    """
    Modulates a set of scores by the predictive entropy of the model's logits.
    Samples with high uncertainty (high entropy) will have their scores boosted.

    :param scores: A tensor of base scores (e.g., gradient norms).
    :param logits: The raw logit outputs from the model.
    :return: A tensor of modulated scores.
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)

    # modulating the score with entropy (we add 1 to avoid multiplying by zero for certain samples)
    modulated_scores = scores * (1 + entropy)
    return modulated_scores