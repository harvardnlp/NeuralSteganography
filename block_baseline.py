import torch
import torch.nn.functional as F

import numpy as np
from utils import kl, entropy, is_sent_finish, limit_past, bits2int, int2bits

# number of bins is 2^block_size
# each bin contains vocab_size/2^block_size words
def get_bins(vocab_size, block_size):
    num_bins = 2**block_size
    words_per_bin = vocab_size/num_bins 

    vocab_ordering = np.arange(vocab_size)
    np.random.seed(block_size)
    np.random.shuffle(vocab_ordering)

    bin2words = [vocab_ordering[int(i*words_per_bin):int((i+1)*words_per_bin)] for i in range(num_bins)]
    bin2words = [np.array(words) for words in bin2words]
    words2bin_list = [{i: j for i in bin2words[j]} for j in range(num_bins)]
    words2bin = {}
    for d in words2bin_list:
        words2bin.update(d)

    return bin2words, words2bin

def encode_block(model, enc, message, context, block_size, bin2words, words2bin, finish_sent=False, device='cuda'):
    length = len(message)

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    
    prev = context
    output = context
    past = None
    
    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0 # in bits
    total_num_sents = 0
    
    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < length or (finish_sent and not sent_finish):
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e10 # endoftext can't happen
            logits[0, -1, 628] = -1e10 # 2 newlines can't happen
            logits = logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            filtered_logits = logits.clone()
            filtered_logits[:] = -1e10 # first set all to 0

            if i >= length:
                _, indices = logits.sort(descending=True)
                sent_finish = is_sent_finish(indices[0].item(), enc)
            else:
                # First calculate logq
                logq = logits.clone()
                logq[:] = -1e10 # first set all to 0

                for bin_val in range(2**block_size):
                    filtered_logits = logits.clone()
                    filtered_logits[:] = -1e10 # first set all to 0
                    available_tokens = bin2words[bin_val]
                    filtered_logits[available_tokens] = logits[available_tokens]
                    filtered_logits, indices = filtered_logits.sort(descending=True)

                    logq[indices[0]] = -block_size # in bits

                logq = logq*0.69315 # in nats
                q = torch.exp(logq)

                # Then find the actual word for the right bin
                m_part = message[i:i+block_size]

                filtered_logits = logits.clone()
                filtered_logits[:] = -1e10 # first set all to 0
                available_tokens = bin2words[bits2int(m_part)]
                filtered_logits[available_tokens] = logits[available_tokens]
                filtered_logits, indices = filtered_logits.sort(descending=True)

                total_kl += kl(q, logq, log_probs)
                total_log_probs += log_probs[indices[0]].item()
                i += block_size
                total_num_for_stats += 1
                

            total_num += 1
            prev = indices[0].view(1)
            output = torch.cat((output, prev))

    avg_NLL = -total_log_probs/total_num_for_stats
    avg_KL = total_kl/total_num_for_stats
    words_per_bit = total_num_for_stats/i

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit

def decode_block(model, enc, text, context, block_size, bin2words, words2bin, device='cuda'):
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)
    i = 0
    while i < len(inp):
        if inp[i] == 628:
            inp[i] = 198
            inp[i+1:i+1] = [198]
            i += 2
        else:
            i += 1

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    prev = context
    past = None

    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            if past and past[0].shape[3] >= 1023:
                raise RuntimeError
            bin_num = words2bin[inp[i]]

            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e10 # endoftext can't happen
            logits[0, -1, 628] = -1e10 # 2 newlines can't happen

            logits = logits[0, -1, :]
            filtered_logits = logits.clone()
            filtered_logits[:] = -1e10 # first set all to 0

            available_tokens = bin2words[bin_num]
            filtered_logits[available_tokens] = logits[available_tokens]
            filtered_logits, indices = filtered_logits.sort(descending=True)
            
            rank = (indices == inp[i]).nonzero().item()

            # Handle errors that could happen because of BPE
            if rank > 0:
                true_token_text = enc.decoder[inp[i]]
                for bin_num in range(len(bin2words)):
                    filtered_logits = logits.clone()
                    filtered_logits[:] = -1e10 # first set all to 0

                    available_tokens = bin2words[bin_num]
                    filtered_logits[available_tokens] = logits[available_tokens]
                    filtered_logits, indices = filtered_logits.sort(descending=True)

                    prop_token_text = enc.decoder[indices[0].item()]
                    #print(true_token_text, prop_token_text)

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) < len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix) # a list
                        inp[i] = indices[0].item()
                        inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                              true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i+num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            inp[i] = indices[0].item()
                            for j in range(1, num_extra):
                                del inp[i+j]
                        
                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix) # a list
                                inp[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))

            tokens_t = int2bits(bin_num, block_size)

            message.extend(tokens_t)
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            i += 1

    return message

if __name__ == '__main__':
    np.random.seed(123)

    bin2words, words2bin = get_bins(50257, 5)
    print(words2bin[153])
