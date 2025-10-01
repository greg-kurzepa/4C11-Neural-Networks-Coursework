#%%

import torch
import torch.nn.utils.rnn as rnn_utils
# Define sequences
sequences = [  
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9],
    [10],
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9],
    [10]
]
sequences_tensor = [torch.tensor(seq) for seq in sequences] # Convert sequences to PyTorch tensors

# Padding
padded_sequences = rnn_utils.pad_sequence(sequences_tensor, batch_first=True)
print("Padded sequences:","\n",padded_sequences)

# Packing 
sequence_lengths = torch.tensor([len(seq) for seq in sequences]) # Calculating actual lengths of sequences
# Pack padded sequences
packed_sequences = rnn_utils.pack_padded_sequence(padded_sequences, sequence_lengths, batch_first=True, enforce_sorted=False)
print("\nPacked sequences:",packed_sequences)
# %%
