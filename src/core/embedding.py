# Word2Vec with Numpy to learn the embedding (might switch to Pytorch)
import torch
from torchtext.data import get_tokenizer
#from profile import Faculty

def embed_bios(bios):
    '''
    Turn list of bio description from user profiles into numerical embeddings.
    Params: List of Strings (bios)
    Output: List of Vector<double>
    '''
    # tokenize
    tokenizer = get_tokenizer("basic_english")
    tokenized_bios = []
    for bio in bios:
        tokenized_bio = tokenizer(bio)
        tokenized_bios.append(tokenized_bio)

    # build the vocab from tokenized bios
    vocab = list(set(word for bio in tokenized_bios for word in bio))
    word_to_index = {}
    for word, index in enumerate(vocab):
        word_to_index[word] = index

    # initialize the word embedding layer (used @ tensor -> embeddings)
    word_embedding = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=40) # lower if overfitting

    # bio token -> indexed bios
    indexed_bios = []
    for bio in tokenized_bios:
        indexed_bios.append(vocab[word] for word in bio)

    # convert into tensors
    tensors = []
    for bio in indexed_bios:
        tensor = torch.tensor(bio)
        tensors.append(tensor)

    # embed bios (newly converted into tensors) into embedding vectors
    embedded_bios = []
    for tensor in tensors:
        emb_bio = word_embedding(tensor)
        embedded_bios.append(emb_bio)
