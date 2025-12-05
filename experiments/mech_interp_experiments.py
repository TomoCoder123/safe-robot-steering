import torch
import random
import numpy as np

from model.smolvla_policy import SmolVLALiberoPolicy
from sklearn.neighbors import NearestNeighbors

from env.env import make_libero_env, snapshot_obs, get_agentview_frame, get_wrist_frame
import imageio
import os
FPS = 60

TASK_SUITE_NAME = "libero_10" 
STEPS = 520 #lerobot libero default for object, 520 is default for long

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
    token_vecs = token_vecs / (token_vecs.norm(dim=-1, keepdim=True) + 1e-8)


    # computing weights of topk_vals using a softmax function
    weights = torch.softmax(topk_vals, dim=1)       #dimensions are 2560 x 30 still 
    weights = weights.unsqueeze(-1) #"Unsqueeze" to add another dimension to match that of token_vecs. "-1" denotes addition at the last dimension
    # Thus, we have (2560 x 30 x 1)

    semantic_embeds = (token_vecs * weights).sum(dim=1)    #This 3D multiplication allows us to generate 2560 x 960. We sum over the dimension w/ size 30.
    semantic_embeds = semantic_embeds / (semantic_embeds.norm(dim=-1, keepdim=True) + 1e-8)

    print(semantic_embeds.shape)

    return semantic_embeds

"""
K-nearest neighbors algorithm through sklearn. 
Finding the k nearest semantic embeddings for a given activation.
Clustering, in essence.
"""
def KNN(sem_embed, valvec_ID, k):
    #sklearn utilizes CPU. However, sem_embed is a torch tensor on GPU.

    sem_embed = sem_embed.to(torch.float32).detach().cpu().numpy() #converting tensor to ndarray
    knn = NearestNeighbors(n_neighbors = k + 1, metric = "cosine") #10 + self = 11. Choosing 10 because with 20, the cluster size may be too big at layer 16.
    knn.fit(sem_embed)

    distances, indices = knn.kneighbors(sem_embed[valvec_ID:valvec_ID+1]) #sem_embed is 2560 x 960, thus the input is the row of 960 

    return distances, indices


def perform_downproj_steering(policy, layer, cluster, alpha):
    down_projs = find_down_projs(policy)
    layer_module = down_projs[layer][1] #module as (name, tensor)

    def hook(module, inputs):
        (act,) = inputs # the activations 
        act = act.clone()
        act[..., cluster] = alpha
        return (act,)

    handle = layer_module.register_forward_pre_hook(hook)
    return handle

"""
returns average 3D displacement
"""
def rollout(task_id, policy):
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    assert torch.cuda.device_count() == 1, "More than one CUDA device available which will lead to runtime device mismatches"

    # If you want to train it (e.g., GRPO)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95)
    )

    env, language = make_libero_env(TASK_SUITE_NAME, task_id=task_id)
    print(f"Task description: {language}")
    obs = env.reset()
    snapshot_obs(obs, "before.png")
    
    agentview_path = "agentview.mp4"
    wristcam_path = "wristcam.mp4"

    agent_writer = imageio.get_writer(agentview_path, fps=FPS)
    wrist_writer = imageio.get_writer(wristcam_path, fps=FPS)

    print(f"Recording agentview -> {agentview_path}")
    print(f"Recording wristcam -> {wristcam_path}")
    
    # Write first frames
    agent_writer.append_data(get_agentview_frame(obs))
    wrist_writer.append_data(get_wrist_frame(obs))

    eef_positions = []
    eef_positions.append(np.array(obs["robot0_eef_pos"], dtype=float))
    
    # simple rollout. For GRPO we can rollout with no_grad but will need grads when recomputing new model log densities for chosen actions
    # print out first and last end-effector postitions
    for step in range(STEPS):
        with torch.no_grad():
            action = policy.get_action(obs, language)
            action = action.cpu().clone().detach().tolist()[0]
        obs, reward, done, info = env.step(action)

        eef_positions.append(np.array(obs["robot0_eef_pos"], dtype=float))
        
        print(f"Step {step} | Reward: {reward:.3f}")
            
        agent_writer.append_data(get_agentview_frame(obs))
        wrist_writer.append_data(get_wrist_frame(obs))
        if done:
            break

    snapshot_obs(obs, "after.png")
    env.close()

    eef_positions = np.stack(eef_positions, axis=0)   # shape: (T, 3)
    diffs = eef_positions[1:] - eef_positions[:-1]    # shape: (T-1, 3)
    step_displacements = np.linalg.norm(diffs, axis=1)  # (T-1,)

    avg_disp = float(step_displacements.mean())   # average displacement per step

    print(f"Average EEF displacement per step: {avg_disp:.6f}")

    return avg_disp*100 #avg_disp gives output in meters. We get cm output by multiplying by 100.


def rollout_repeat(n, policy):
    avg_displacements = []
    for i in range(n):
        avg_displacements.append(rollout(3, policy))
    return avg_displacements

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLALiberoPolicy(model_name="HuggingFaceVLA/smolvla_libero",device=device)
    #this policy contains the SmolVLA policy, then SmolVLM2, SmolLM2, and the FFNs
    policy.eval()

    down_projs = find_down_projs(policy.policy)
    LM_head = find_LMhead(policy.policy) #still the module
    embed_matrix = LM_head[0][1].weight #convert the module to the tensor

    layer = 15 #N/2 = 16, indexing begins at 0
    k = 30 #as used by Haon et al.
    valvec_ID = 126 #fast
    cluster_size = 20
    alpha = 2

    """
    In this workflow, this is where we change the layer number.
    """
    value_vectors = extract_value_vectors(down_projs, layer)
    logits = project_tkn_space(value_vectors, embed_matrix) # these are the logits, the semantic embeddings in token space?

    """
    Get the top k tokens for each value vector within the defined layer
    """
    top_k_tokens, top_k_scores = get_top_k_tokens(logits, policy.tokenizer, k)

    """
    Output the value vectors that have tokens with the word fast. Manually search for value vectors with semantic meaning.
    """
    word_in_tokens(top_k_tokens, "fast")
    word_in_tokens(top_k_tokens, "slow")
    
    semantic_embeddings = create_semantic_embeddings(logits, embed_matrix, k) #using k=30

    distances, indices = KNN(semantic_embeddings, valvec_ID, cluster_size)

    #constant for activation replacement
    handle = perform_downproj_steering(policy.policy, layer, indices, alpha)

    #handle.remove()

    avg_displacements = rollout_repeat(3, policy) # acts on task num 3, defined implicitlu

    print("average displacements:", avg_displacements)
    print("mean over rollouts:", sum(avg_displacements) / len(avg_displacements))


if __name__ == "__main__":
    main()

