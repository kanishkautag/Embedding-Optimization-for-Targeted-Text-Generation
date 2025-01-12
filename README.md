# **Embedding Optimization for Targeted Text Generation**

This project explores **optimizing input embeddings** to steer language models (e.g., GPT-2) toward generating specific target outputs. By directly manipulating embeddings, we can influence the model's predictions while maintaining coherence and adherence to the vocabulary.

This work is heavily inspired by [Jessica Rumbelow's Backwards repository](https://github.com/jessicarumbelow/Backwards). While this project builds upon her concepts, we provide extended explanations, new features, and a strong focus on the mathematical foundations. Additionally, we visualize outputs using **word clouds** to demonstrate the generated token diversity and alignment with the target.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Mathematical Breakdown](#mathematical-breakdown)
4. [Key Functions](#key-functions)
5. [Word Cloud Visualizations](#word-cloud-visualizations)
6. [How to Use](#how-to-use)
7. [Credits](#credits)

---

## **1. Overview**

Large language models like GPT-2 are powerful tools for text generation. While prompts are typically used to guide these models, this project takes a different approach: directly optimizing **input embeddings** to achieve a target output.

For example, we can guide a model to generate "world" by modifying embeddings instead of tweaking prompts. This approach provides greater control over the output and reveals insights into how language models process embeddings.

---

## **2. Core Concepts**

### **Optimization**
- The goal is to optimize a sequence of input embeddings \( E_{\text{input}} \) to maximize the likelihood of generating a target output (e.g., "world").
- A loss function is minimized to ensure embeddings align with the target while maintaining semantic validity.

### **Constraints**
1. **Log-Probability Loss**:
   Encourages the model to generate tokens corresponding to the target output.


2. **Distance Regularization**:
   Keeps optimized embeddings close to valid token embeddings to ensure meaningful outputs.


3. **Diversity Penalty**:
   Penalizes repetitive outputs to encourage diverse generations.


4. **Total Loss**:
   Combines all components:

---

Certainly! Here's the mathematical breakdown in plain text for GitHub without special characters:

---

## **3. Mathematical Breakdown**

### **Cosine Similarity and Distance**
To find the closest tokens to a given embedding, we compute the cosine similarity.

The **cosine distance** is then:

    cosine_distance = 1 - cosine_similarity

### **Logits and Token Selection**
At each timestep, the transformer predicts logits L_t for all vocabulary tokens:

    L_t = f_model(E_1:t)

The most probable token is selected using:

    argmax(L_t) = argmax_{v in V} L_t

The corresponding embedding e_output is appended to the sequence:

    E_1:t+1 = concat(E_1:t, e_output)

---

### **Distance Regularization**
Ensures that optimized embeddings e_opt stay close to valid embeddings e_closest:

    L_dist = ||e_opt - e_closest||_2

### **Normalization**
Normalizes embeddings to a unit range:

    e_norm = (e - min(e)) / (max(e) - min(e))

---

This should be suitable for GitHub with plain text formatting!

## **4. Key Functions**

### **`closest_tokens`**
Finds the tokens closest to a given embedding using cosine similarity:
```python
def closest_tokens(emb, word_embeddings, tokenizer, n=1):
    dists = 1 - (emb.unsqueeze(0) @ word_embeddings.T).squeeze(0)
    sorted_dists, ix = torch.sort(dists)
    tokens = [tokenizer.decode(i) for i in ix[:n]]
    return tokens, ix[:n], sorted_dists[:n], word_embeddings[ix[:n]]
```

- **Input**: Embedding \( e_{\text{opt}} \), vocabulary embeddings.
- **Output**: Closest tokens, indices, and distances.

---

### **`model_emb`**
Generates tokens iteratively and computes the model’s logits:
```python
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
```

---

### **`normalise`**
Normalizes a tensor to a specified range:
```python
def normalise(x, min_max=[]):
    rnge = x.max() - x.min()
    if rnge > 0:
        x = (x - x.min()) / rnge
    if len(min_max) > 1:
        rnge = min_max[1] - min_max[0]
        x = x * rnge + min_max[0]
    return x
```

---

## **5. Word Cloud Visualizations**

Below are word clouds generated from the optimized embeddings for different target phrases. These illustrate the diversity and relevance of the tokens generated.

### **Example 1: Target Output - "world"**
![image](https://github.com/user-attachments/assets/ef558acb-1902-4874-8596-6bb10e0f98a0)


### **Example 2: Target Output - "engineer"**
![image](https://github.com/user-attachments/assets/1249df9f-8991-4638-ab11-410d44f34f0c)

### **Example 3: Target Output - "girl"**
![image](https://github.com/user-attachments/assets/e57b459d-5bfe-4322-9787-e700dfbd8387)

## **6. How to Use**

1. **Setup**:
   - Install dependencies (`transformers`, `torch`, etc.).
   - Clone the repository.
   - Prepare a pre-trained language model (e.g., GPT-2).

2. **Run Optimization**:
   - Define the target output (e.g., "world").
   - Optimize embeddings using the provided functions.

3. **Visualize Results**:
   - Use the decoded tokens to evaluate the optimized embeddings.
   - Generate word clouds using Python's `wordcloud` library to visualize token diversity.

---

## **7. Credits**

This project is **heavily inspired by [Jessica Rumbelow's Backwards repository](https://github.com/jessicarumbelow/Backwards)**, which introduced the concept of embedding optimization for text generation. This implementation builds on her work by adding:

