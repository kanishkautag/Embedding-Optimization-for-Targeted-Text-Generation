import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, GPTJForCausalLM
from sklearn.mixture import GaussianMixture
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def load_all(model_name="gpt2", device='cpu', save_dir=''):
    """Load model, embeddings, and tokenizer"""
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
    embeddings = model.transformer.wte.weight.detach()
    
    if model_name + '_embeddings' not in cur_dir:
        torch.save(embeddings, os.path.join(save_dir, model_name + '_embeddings'))

    return model.to(device), embeddings.to(device), tokenizer

def model_emb(model, inputs_embeds, word_embeddings, output_len):
    """Get model logits and embeddings"""
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

def initialize_inputs(word_embeddings, input_len, batch_size, method='random'):
    """Initialize input embeddings for optimization"""
    if method == 'random':
        start_input = word_embeddings[torch.randperm(word_embeddings.shape[0])[:input_len * batch_size]].reshape(
            batch_size, input_len, -1)
    else:
        raise ValueError("Unsupported initialization method")
    return torch.nn.Parameter(start_input, requires_grad=True)

def knn_clustering(embeddings, query_embeddings, k=5):
    """KNN-based clustering"""
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    query_embeddings = query_embeddings / torch.norm(query_embeddings, dim=1, keepdim=True)
    similarities = torch.matmul(query_embeddings, embeddings.T)
    distances = 1 - similarities
    sorted_distances, sorted_indices = distances.topk(k, dim=-1, largest=False)
    return sorted_indices, sorted_distances

def gmm_clustering(embeddings, n_components=5, random_state=42):
    """GMM-based clustering"""
    embeddings_np = embeddings.cpu().detach().numpy() if torch.is_tensor(embeddings) else embeddings
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(embeddings_np)
    return labels, gmm

def create_word_cloud(words):
    """Create and display a word cloud"""
    text = " ".join(words)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Optimized Words")
    plt.show()

def optimize_inputs(
    model, tokenizer, word_embeddings, device, target_output,
    input_len=10, batch_size=20, epochs=100, lr=0.1,
    dist_reg=0.1, cluster_method='knn', cluster_kwargs=None,
    verbose=1
):
    """Optimize input embeddings using either KNN or GMM clustering"""
    print(f'Optimizing input of length {input_len} to maximize output logits for "{target_output}"')
    
    output_ix = tokenizer.encode(target_output, return_tensors='pt')[0].to(device)
    output_len = output_ix.shape[0]
    word_embeddings = word_embeddings / torch.norm(word_embeddings, dim=-1, keepdim=True)
    input_embeds = initialize_inputs(word_embeddings, input_len, batch_size)
    optimizer = torch.optim.Adam([input_embeds], lr=lr)
    
    cluster_kwargs = cluster_kwargs or {}
    all_optimized_words = []

    for epoch in range(epochs):
        logits, emb = model_emb(model, input_embeds, word_embeddings, output_len)
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[:, torch.arange(output_len), output_ix]
        prob_loss = -target_log_probs.mean()

        # Clustering-based regularization
        if cluster_method == 'knn':
            knn_indices, knn_distances = knn_clustering(
                word_embeddings,
                input_embeds.view(-1, word_embeddings.size(-1)),
                k=cluster_kwargs.get('k', 5)
            )
            dist_loss = knn_distances.mean() * dist_reg
            indices = knn_indices
        else:  # GMM
            flat_embeds = input_embeds.view(-1, word_embeddings.size(-1)).detach()
            labels, gmm = gmm_clustering(
                flat_embeds,
                n_components=cluster_kwargs.get('n_components', 5),
                random_state=cluster_kwargs.get('random_state', 42)
            )
            dist_loss = torch.tensor(gmm.score_samples(flat_embeds.cpu().numpy()).mean()) * dist_reg
            # Get closest words for visualization
            indices = torch.tensor(
                [np.argmin(np.linalg.norm(word_embeddings.cpu() - emb.cpu(), axis=1))
                 for emb in flat_embeds]
            ).unsqueeze(1)

        total_loss = prob_loss + dist_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Collect optimized words
        for idx in indices:
            tokens = [tokenizer.decode(i.item()) for i in idx]
            all_optimized_words.extend(tokens)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")

    return input_embeds, all_optimized_words