import torch
import random

from model.smolvla_policy import SmolVLALiberoPolicy
from sklearn.neighbors import NearestNeighbors



"""
Outputs a list. Contains 1 object, a module that contains the LM head, which has unembedding matrix
Necessary to project value vectors onto the token space
"""
def find_LMhead(policy):
    lm_head = []
    for name, module in policy.named_modules():
        print(name)
        if('lm_head' in name):
            lm_head.append((name, module))
            
    embed_matrix = lm_head[0][1].weight
    
    """
    print(embed_matrix)
    print(embed_matrix.shape)    #Produces a size of 49280, 960. 
    """
    return lm_head

"""
Finding the down projections
The down projection is crucial in the Llama2 architecture, which the SmolLM2 is made of. 
It acts as the parameter matrix that is described in the Haon et al. paper, projecting the hidden layer to the output layer
"""
def find_down_projs(policy):

    downs = []
    for name, module in policy.named_modules():
        if('mlp.down_proj' in name and 'text_model' in name):
            downs.append((name, module))

    #dimensionality is 960 for output, 2560 for hidden layer
    #optionally print a small 5x5 portion of the matrix
    """
    module1 = downs[1][1].weight
    print(module1[:5, :5])
    """
    return downs

"""
Value vector extraction. Layer-specific.
down_projects holds 32 modules/weight_matrices. 2560 value vectors per layer. 32 x 2560 x 960 = 78643200
"""
def extract_value_vectors(down_projs, layer):
    matrix = down_projs[layer][1].weight #gives the weight matrix of the layer
    value_vectors = matrix.T #transpose it.  now indexing gives the value vectors of dim 960

    return value_vectors

"""
Input: value vectors for 1 layer, unembedding matrix
Output: logits (2560, 49280) one probabiltiy distribution for every neuron
"""
def project_tkn_space(value_vectors, embed_mat):
    logits = value_vectors @ embed_mat.T 
    return logits


"""
Input: logits, tokenizer, k.
Output: list of top k tokens for each value vector
"""
def get_top_k_tokens(logits, tokenizer, k):

    #built in pytorch function that returns k largest elements along the token space 
    topk_vals, topk_indx = torch.topk(logits, k, dim=1)
    # both have dim (2560, k) but store the logit values and indices

    topk_tokens = []
    topk_scores = []

    for i in range(len(logits)):
        token_ids = topk_indx[i].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids) #tokenization

        topk_tokens.append(tokens)
        topk_scores.append(topk_vals[i].tolist())

    return topk_tokens, topk_scores

"""
Find neurons that activate tokens with this specific word
"""
def word_in_tokens(top_k_token_list, word):
    list = []
    for i in range(len(top_k_token_list)): 
        for token in top_k_token_list[i]:
            if (word in token):
                list.append((i, top_k_token_list[i]))
        for j in range(len(top_k_token_list[i])): #clean it up, the words
            if ('Ġ' in top_k_token_list[i][j]):
                top_k_token_list[i][j] = top_k_token_list[i][j].replace("Ġ","")

    for i in range(len(list)):
        print()
        print(list[i][0]) #print out the activation number
        print(list[i][1]) # print out the actual tokens


"""
Neuron grouping using KNN and semantic embeddings of neurons.
Input: list of top k tokens (2560, 30). logits (2560, 49280). unembedding matrix (49280, 960)
Output: list of tuples. neuron number, as well as semantic embedding (averaged vector of top k tokens)
"""
def create_semantic_embeddings(top_k_tokens_list, logits, embed_mat, tokenizer):
    avgs = []
    for num in range(len(top_k_tokens_list)): #2560
        
        for token in top_k_tokens_list[num]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            token_embed_vector = embed_mat[token_id]
            token_logit = logits[num][token_id]
            sum += token_logit * token_embed_vector
            avgs.append((num, token_logit * token_embed_vector))
    return avgs

def KNN(sem_embed, activation_ID):
    knn = NearestNeighbors(n_neighbors = 21, metric = "cosine")
    knn.fit(sem_embed)

    distances, indices = knn.kneighbors(sem_embed[activation_ID:activation_ID+1])

    return distances, indices

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLALiberoPolicy(model_name="HuggingFaceVLA/smolvla_libero",device=device)
    #this policy contains the SmolVLA policy, then SmolVLM2, SmolLM2, and the FFNs
    policy.eval()

    down_projs = find_down_projs(policy.policy)
    LM_head = find_LMhead(policy.policy) #still the module
    embed_matrix = LM_head[0][1].weight #convert the module to the tensor

    layer = 15 #N/2 = 16, indexing begins at 0
    value_vectors = extract_value_vectors(down_projs, layer)
    logits = project_tkn_space(value_vectors, embed_matrix) # these are the logits, the semantic embeddings in token space?

    """
    Get the top k tokens for each value vector within the defined layer
    """
    k = 30 #as used by Haon et al.
    top_k_tokens, top_k_scores = get_top_k_tokens(logits, policy.tokenizer, k)

    """
    Output the value vectors that have tokens with the word fast. Manually search for value vectors with semantic meaning.
    """
    word_in_tokens(top_k_tokens, "fast")

    semantic_embeddings = create_semantic_embeddings(top_k_tokens, logits, embed_matrix, policy.tokenizer)
    print(semantic_embeddings)



    

    #cluster to find the neurons for "Fast" or "Up". choose one

    #steering via forward hook. 

    #output will be in libero modeling.
    #we can compare success rates across tasks, or just focus on one task in particular.
    #for "up" we want to focus on gripper height (is there any way to actually get this value)
    #for "Fast" we want to focus on average speed (is there any way to get the velocity outputs)
    #we could also focus on time to reach goal


if __name__ == "__main__":
    main()