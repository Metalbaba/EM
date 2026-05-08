import torch
import torch.nn.functional as F

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the log probabilities of the actual sequence of tokens.
    logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    """
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    
    log_probs = -F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)), 
        shifted_labels.view(-1), 
        reduction='none'
    )
    return log_probs.view(shifted_labels.shape[0], -1).sum(dim=-1)

def simple_dpo_loss(
    pi_logps_A: torch.Tensor, 
    pi_logps_B: torch.Tensor, 
    ref_logps_A: torch.Tensor, 
    ref_logps_B: torch.Tensor, 
    beta: float = 0.1
):
    """
    BASELINE DPO: Blindly assumes A is the absolute winner.
    Used to prove how badly the model fails when trained on raw, noisy crowdsourced data.
    """
    pi_logratios = pi_logps_A - pi_logps_B
    ref_logratios = ref_logps_A - ref_logps_B
    
    logits = pi_logratios - ref_logratios
    
    # Standard DPO loss assumes A is the preferred response 100% of the time
    loss = -F.logsigmoid(beta * logits)
    
    return loss.mean()

def projected_dpo_loss(
    pi_logps_A: torch.Tensor, 
    pi_logps_B: torch.Tensor, 
    ref_logps_A: torch.Tensor, 
    ref_logps_B: torch.Tensor, 
    trust_weights_A: torch.Tensor,
    beta: float = 0.1
):
    """
    PROJECTED DPO: Uses the EM algorithm's confidence (trust_weights_A) 
    to scale and invert gradients, neutralizing spammers and adversaries.
    """
    pi_logratios = pi_logps_A - pi_logps_B
    ref_logratios = ref_logps_A - ref_logps_B
    
    logits = pi_logratios - ref_logratios
    
    # Loss if A is the true winner
    loss_if_A = -F.logsigmoid(beta * logits)
    
    # Loss if B is the true winner (invert the logits)
    loss_if_B = -F.logsigmoid(-beta * logits)
    
    # Project the expectation over the EM posterior probabilities
    expected_loss = (trust_weights_A * loss_if_A) + ((1.0 - trust_weights_A) * loss_if_B)
    
    return expected_loss.mean()