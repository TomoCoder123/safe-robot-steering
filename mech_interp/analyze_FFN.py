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
Output: list of tuples. activation number, as well as semantic embedding (averaged vector of top k tokens)
"""
def create_semantic_embeddings(logits, unembed_mat, k):
    topk_vals, topk_ids = torch.topk(logits, k=k, dim=1)    #2560 x 30
    token_vecs = unembed_mat[topk_ids] # (2560, 30, 960)

    #L2 normalization in order to create unit vectors for cosine similarity
    token_vecs = token_vecs / (token_vecs.norm())

    # computing weights of topk_vals using a softmax function
    weights = torch.softmax(topk_vals, dim=1)       #dimensions are 2560 x 30 still 
    weights = weights.unsqueeze(-1) #"Unsqueeze" to add another dimension to match that of token_vecs. "-1" denotes addition at the last dimension
    # Thus, we have (2560 x 30 x 1)

    semantic_embeds = (token_vecs * weights).sum(dim=1)    #This 3D multiplication allows us to generate 2560 x 960. We sum over the dimension w/ size 30.
    semantic_embeds = semantic_embeds / (semantic_embeds.norm())

    print(semantic_embeds.shape)

    return semantic_embeds

"""
K-nearest neighbors algorithm through sklearn. 
Finding the k nearest semantic embeddings for a given activation.
Clustering, in essence.
"""
def KNN(sem_embed, activation_ID, k):
    #sklearn utilizes CPU. However, sem_embed is a torch tensor on GPU.

    sem_embed = sem_embed.to(torch.float32).detach().cpu().numpy() #converting tensor to ndarray
    knn = NearestNeighbors(n_neighbors = k + 1, metric = "cosine") #10 + self = 11. Choosing 10 because with 20, the cluster size may be too big at layer 16.
    knn.fit(sem_embed)

    distances, indices = knn.kneighbors(sem_embed[activation_ID:activation_ID+1]) #sem_embed is 2560 x 960, thus the input is the row of 960 

    return distances, indices


def perform_downproj_steering(policy, layer, cluster, alpha):
    down_projs = find_down_projs(policy)
    steered_layer = down_projs[layer]


    return steered_layer

"""
def word_to_steer(word, policy, alpha):
    perform_steering
    """



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLALiberoPolicy(model_name="HuggingFaceVLA/smolvla_libero",device=device)
    #this policy contains the SmolVLA policy, then SmolVLM2, SmolLM2, and the FFNs
    policy.eval()

    down_projs = find_down_projs(policy.policy)
    LM_head = find_LMhead(policy.policy) #still the module
    embed_matrix = LM_head[0][1].weight #convert the module to the tensor

    """
    In this workflow, this is where we change the layer number.
    """
    layer = 15 #N/2 = 16, indexing begins at 0
    value_vectors = extract_value_vectors(down_projs, layer)
    logits = project_tkn_space(value_vectors, embed_matrix) # these are the logits, the semantic embeddings in token space?

    """
    Get the top k tokens for each value vector within the defined layer
    """
    k = 30 #as used by Haon et al.
    top_k_tokens, top_k_scores = get_top_k_tokens(logits, policy.tokenizer, k)

    print(embed_matrix.shape)

    """
    Output the value vectors that have tokens with the word fast. Manually search for value vectors with semantic meaning.
    """
    word_in_tokens(top_k_tokens, "fast")

    semantic_embeddings = create_semantic_embeddings(logits, embed_matrix, 30) #using k=30
    print(semantic_embeddings) #prints out 2560 unit vectors in R^960 space

    distances, indices = KNN(semantic_embeddings, 126, 20)
    print(distances) # These distances are unitless and based on cosine similarity
    print(indices)

    #constant for activation replacement
    alpha = 10 
    print(perform_downproj_steering(policy.policy, layer, indices, alpha)[1])
    

    #cluster to find the neurons for "Fast" or "Up". choose one

    #steering via forward hook. 

    #output will be in libero modeling.
    #we can compare success rates across tasks, or just focus on one task in particular.
    #for "up" we want to focus on gripper height (is there any way to actually get this value)
    #for "Fast" we want to focus on average speed (is there any way to get the velocity outputs)
    #we could also focus on time to reach goal


if __name__ == "__main__":
    main()