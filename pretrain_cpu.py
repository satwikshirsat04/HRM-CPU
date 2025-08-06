import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
from types import SimpleNamespace
from puzzle_dataset import PuzzleDataset
from models.losses import ACTLossHead
from utils.functions import load_model_class

# ==== CONFIG ====
from types import SimpleNamespace

config = SimpleNamespace(
    arch_name='hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1',
    loss_type='losses@ACTLossHead',
    dataset_path='data/arc-simple',

    # Training
    batch_size=2,
    global_batch_size=2,   # ✅ New (for single-replica CPU training)
    num_replicas=1,        # ✅ New (only 1 machine, 1 process)
    epochs=2,
    lr=1e-3,
    seed=42,
    test_set_mode=False,
    epochs_per_iter=1,

    # Model parameters
    seq_len=64,
    vocab_size=12,
    num_puzzle_identifiers=512,
    puzzle_emb_ndim=32,

    H_cycles=2,
    L_cycles=2,
    H_layers=1,
    L_layers=1,

    hidden_size=64,
    expansion=2.0,
    num_heads=4,
    pos_encodings="rope",

    halt_max_steps=6,
    halt_exploration_prob=0.1
)


torch.manual_seed(config.seed)

# ==== MODEL ====
model_cls = load_model_class(config.arch_name)
model = model_cls(vars(config))
loss_fn_cls = load_model_class(config.loss_type)
model_with_loss = loss_fn_cls(model, loss_type="softmax_cross_entropy")

device = torch.device("cpu")
model_with_loss.to(device)

# ==== DATA ====
train_dataset = PuzzleDataset(config, split='train')
train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

# ==== OPTIMIZER ====
optimizer = optim.Adam(model_with_loss.parameters(), lr=config.lr)

# ==== TRAIN LOOP ====
for epoch in range(config.epochs):
    model_with_loss.train()
    total_loss = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        carry = model_with_loss.initial_carry(batch=batch)

        _, loss, metrics, _, halted = model_with_loss(**batch, return_keys=["logits"], carry=carry)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")

# ==== SAVE ====
torch.save(model.state_dict(), "hrm_cpu_model.pth")
print("✅ Model training complete and saved.")
