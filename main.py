import torch
import numpy as np
import random
from helpers import load_all, closest_tokens, model_emb


device="cpu"
# Initialize inputs
def initialize_inputs(word_embeddings, input_len, batch_size, method='random'):
    if method == 'random':
        start_input = word_embeddings[torch.randperm(word_embeddings.shape[0])[:input_len * batch_size]].reshape(
            batch_size, input_len, -1)
    else:
        raise ValueError("Unsupported initialization method")
    return torch.nn.Parameter(start_input.to(device), requires_grad=True)

# Optimize inputs
def optimize_inputs(
    model, tokenizer, word_embeddings, device, target_output, input_len=10,
    batch_size=20, epochs=100, lr=0.1, dist_reg=0.1, verbose=1
):
    print(f'Optimizing input of length {input_len} to maximize output logits for "{target_output}"')

    # Tokenize the target output
    output_ix = tokenizer.encode(target_output, return_tensors='pt')[0].to(device)
    output_len = output_ix.shape[0]

    # Normalize embeddings
    word_embeddings = word_embeddings / torch.sqrt(torch.sum(word_embeddings**2, dim=-1, keepdim=True))

    # Initialize inputs
    input_embeds = initialize_inputs(word_embeddings, input_len, batch_size)

    # Optimizer
    optimizer = torch.optim.Adam([input_embeds], lr=lr)

    for epoch in range(epochs):
        logits, emb = model_emb(model, input_embeds, word_embeddings, output_len)
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[:, torch.arange(output_len), output_ix]
        prob_loss = -target_log_probs.mean()

        # Token distance regularization
        token_distances = []
        for batch in input_embeds:
            distances = []
            for embedding in batch:
                _, _, dist, _ = closest_tokens(embedding, word_embeddings, tokenizer)
                distances.append(dist)
            token_distances.append(torch.stack(distances))
        dist_loss = torch.stack(token_distances).mean() * dist_reg

        total_loss = prob_loss + dist_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}, Prob Loss: {prob_loss.item():.4f}, Dist Loss: {dist_loss.item():.4f}")

    print("Optimization complete!")
    return input_embeds

# Main function
def main():
    # Configuration
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_output = " world"
    input_len = 10
    batch_size = 20
    epochs = 100
    lr = 0.1
    dist_reg = 0.1

    # Load model, tokenizer, and embeddings
    model, word_embeddings, tokenizer = load_all(model_name, device)

    # Optimize inputs
    optimized_inputs = optimize_inputs(
        model, tokenizer, word_embeddings, device, target_output,
        input_len, batch_size, epochs, lr, dist_reg, verbose=1
    )

    # Output results
    print("Optimized Inputs:")
    print(optimized_inputs)

if __name__ == "__main__":
    main()
