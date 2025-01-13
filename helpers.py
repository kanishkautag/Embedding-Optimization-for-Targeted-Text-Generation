import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, GPTJForCausalLM

# Function to load model, embeddings, and tokenizer
def load_all(model_name="gpt2", device='cpu', save_dir=''):
    if save_dir == '':
        cur_dir = os.listdir()
    else:
        cur_dir = os.listdir(save_dir)

    # Load or download tokenizer
    if model_name + '_tokenizer' in cur_dir:
        print('Loading tokenizer...')
        tokenizer = torch.load(os.path.join(save_dir, model_name + '_tokenizer'))
    else:
        print('Downloading tokenizer...')
        if 'gpt-j' in model_name:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
        torch.save(tokenizer, os.path.join(save_dir, model_name + '_tokenizer'))

    # Load or download model
    if model_name + '_model' in cur_dir:
        print('Loading model...')
        model = torch.load(os.path.join(save_dir, model_name + '_model')).to(device)
    else:
        print('Downloading model...')
        if 'gpt-j' in model_name:
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                    torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
        torch.save(model, os.path.join(save_dir, model_name + '_model'))

    model.eval()

    # Load or compute embeddings
    embeddings = model.transformer.wte.weight.detach()
    if model_name + '_embeddings' not in cur_dir:
        torch.save(embeddings, os.path.join(save_dir, model_name + '_embeddings'))

    return model.to(device), embeddings.to(device), tokenizer

# Function to find the closest tokens
def closest_tokens(emb, word_embeddings, tokenizer, n=1):
    dists = 1 - (emb.unsqueeze(0) @ word_embeddings.T).squeeze(0)
    sorted_dists, ix = torch.sort(dists)
    tokens = [tokenizer.decode(i) for i in ix[:n]]
    ixs = ix[:n]
    dists = sorted_dists[:n]
    return tokens, ixs, dists, word_embeddings[ixs]

# Function to get model logits and embeddings
def model_emb(model, inputs_embeds, word_embeddings, output_len):
    embs = inputs_embeds
    logits = []
    for _ in range(output_len):
        model_out = model(inputs_embeds=embs, return_dict=True)
        last_logits = model_out.logits[:, -1].unsqueeze(1)
        logits.append(last_logits)
        ix = torch.argmax(last_logits, dim=-1)
        output_embs = word_embeddings[ix]
        embs = torch.cat([embs, output_embs], dim=1)
    logits = torch.cat(logits, dim=1)
    return logits, embs

# Normalize a tensor to unit range
def normalise(x, min_max=[]):
    rnge = x.max() - x.min()
    if rnge > 0:
        x = (x - x.min()) / rnge
    if len(min_max) > 1:
        rnge = min_max[1] - min_max[0]
        x = x * rnge + min_max[0]
    return x
