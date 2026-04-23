import pickle
with open('./data/genomic/genomic_encoding_states.pkl', 'rb') as f:
    enc = pickle.load(f)
print(enc.keys())