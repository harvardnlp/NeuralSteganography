import torch
import torch.nn.functional as F

from utils import limit_past, kl, entropy

def sample(model, enc, length, context, temperature=1.0, device='cuda', topk=-1):
    assert length > 0

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    prev = context
    output = context
    past = None

    total_log_probs = 0
    total_entropy_ptau = 0
    total_num = 0
    total_kl = 0 # in bits

    with torch.no_grad():
        while total_num < length:
            if past and past[0].shape[3] >= 1023:
                raise RuntimeError

            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e10 # endoftext can't happen
            logits[0, -1, 628] = -1e10 # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            base_log_probs = F.log_softmax(logits, dim=-1)

            if topk > 0:
                logits = logits[:topk]

            logits = logits / temperature
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            total_kl += kl(probs, log_probs, base_log_probs[:topk])

            selection = torch.multinomial(probs, num_samples=1).item()
            log_prob_chosen = base_log_probs[selection]
            total_log_probs += log_prob_chosen.item()
            
            total_entropy_ptau += entropy(probs, log_probs)

            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

    avg_NLL = -total_log_probs/total_num
    avg_KL = total_kl/total_num
    avg_Hq = total_entropy_ptau/total_num

    return output[len(context):].tolist(), avg_NLL, avg_KL, avg_Hq
