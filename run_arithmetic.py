import bitarray
import math

from utils import get_model, encode_context

from arithmetic import encode_arithmetic, decode_arithmetic

def run_single(model_name):
    enc, model = get_model(model_name=model_name)
    print(model_name)

    
    ## PARAMETERS
    message_str = "This is a very secret message!"

    unicode_enc = False
    temp = 0.9
    precision = 26
    topk = 300
    finish_sent = False

    context = \
"""Tensions between the two countries have escalated in recent weeks, prompting international calls for diplomatic intervention. Both leaders are expected to meet in Geneva later this month to discuss a potential resolution.


"""

    context_tokens = encode_context(context, enc)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    # First encode message to uniform bits, without any context
    if unicode_enc:
        ba = bitarray.bitarray()
        ba.frombytes(message_str.encode('utf-8'))
        message = ba.tolist()
    else:
        message_ctx = enc.encode('<|endoftext|>')
        message_str += '<eos>'
        message = decode_arithmetic(model, enc, message_str, message_ctx, precision=40, topk=60000)

    # Next encode bits into cover text, using arbitrary context
    out, nll, kl, words_per_bit, Hq = encode_arithmetic(model, enc, message, context_tokens, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
    text = enc.decode(out)
    
    print("="*40 + " Original Message " + "="*40)
    print(message_str)
    # print(message)
    # print(len(message))

    print("="*40 + " Encoding " + "="*40)
    print(text)
    print('ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' % (math.exp(nll), kl, words_per_bit, 1/words_per_bit, Hq/0.69315))
    
    # Decode binary message from bits using the same arbitrary context
    message_rec = decode_arithmetic(model, enc, text, context_tokens, temp=temp, precision=precision, topk=topk)
    
    print("="*40 + " Recovered Message " + "="*40)
    # print(message_rec)
    # print("=" * 80)

    # Finally map message bits back to original text
    if unicode_enc:
        message_rec = [bool(item) for item in message_rec]
        ba = bitarray.bitarray(message_rec)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
    else:
        reconst = encode_arithmetic(model, enc, message_rec, message_ctx, precision=40, topk=60000)
        reconst = enc.decode(reconst[0])
    print(reconst)
    print()

if __name__ == '__main__':
    run_single("gpt2")
    run_single("gpt2-medium")
    # run_single("gpt2-large")
    # run_single("gpt2-xl")
    # run_single("Qwen/Qwen2.5-0.5B")
    # run_single("Qwen/Qwen2.5-1.5B")
    # run_single("meta-llama/Llama-2-7b-hf")
    # run_single("BAAI/Emu3-Gen")
