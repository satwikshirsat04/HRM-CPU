import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
from puzzle_dataset import PuzzleDataset
from models.losses import ACTLossHead
from utils.functions import load_model_class
import matplotlib.pyplot as plt

# ==== CONFIG ====
config = SimpleNamespace(
    arch_name='hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1',
    loss_type='losses@ACTLossHead',

    dataset_path='data/arc-simple',
    test_set_mode=False,
    global_batch_size=2,
    rank=0,
    num_replicas=1,
    epochs_per_iter=1,

    batch_size=2,
    epochs=10,
    lr=1e-3,
    seed=42,

    # Model parameters - UPDATED TO MATCH YOUR DATASET
    seq_len=900,  # CHANGED FROM 64 TO 900 TO MATCH DATASET
    vocab_size=12,
    num_puzzle_identifiers=2,  # CHANGED FROM 512 TO 2 TO MATCH DATASET
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
device = torch.device("cpu")

print("Initializing model...")

# ==== LOAD MODEL ====
try:
    model_cls = load_model_class(config.arch_name)
    model = model_cls(vars(config))
    
    loss_fn_cls = load_model_class(config.loss_type)
    model_with_loss = loss_fn_cls(model, loss_type="softmax_cross_entropy")
    model_with_loss.to(device)
    
    print("✓ Model initialized successfully")
    print(f"  - Sequence length: {config.seq_len}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Puzzle identifiers: {config.num_puzzle_identifiers}")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    exit(1)

# ==== DATASET ====
try:
    train_dataset = PuzzleDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=None)
    
    print(f"[Debug] Dataset metadata: {train_dataset.metadata}")
    print("✓ Dataset loaded successfully")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    exit(1)

# ==== TEST LOADER ====
try:
    first_batch = next(iter(train_loader))
    print("✓ DataLoader returned the first batch successfully.")
    
    # Print batch info for debugging
    set_name, batch, eff_batch_size = first_batch
    print(f"[Debug] Batch info - Set: {set_name}, Effective batch size: {eff_batch_size}")
    print(f"[Debug] Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        print(f"[Debug] {k} shape: {v.shape}, dtype: {v.dtype}")
        
    # Verify dimensions match
    expected_seq_len = config.seq_len
    actual_seq_len = batch['inputs'].shape[1]
    if expected_seq_len != actual_seq_len:
        print(f"[Error] Sequence length mismatch: expected {expected_seq_len}, got {actual_seq_len}")
        print("Please update config.seq_len to match the dataset or recreate the dataset with smaller sequences")
        exit(1)
    else:
        print(f"✓ Sequence lengths match: {expected_seq_len}")
        
    # Store the batch for carry initialization
    first_batch_data = batch
        
except Exception as e:
    print(f"✗ DataLoader failed to return data: {e}")
    exit(1)

# ==== OPTIMIZER ====
optimizer = optim.Adam(model_with_loss.parameters(), lr=config.lr)

print("Starting training...")


# ==== TRAIN LOOP ====--------------------------------------------------------------
losses = []
for epoch in range(config.epochs):
    total_loss = 0.0
    total_samples = 0
    print(f"\n[Epoch {epoch + 1}] Starting training loop...")

    try:
        # Initialize carry for the new epoch using the first batch data
        batch_for_carry = {k: v.to(device) for k, v in first_batch_data.items()}
        carry = model_with_loss.model.initial_carry(batch=batch_for_carry)
        print("✓ Carry initialized successfully")
    except Exception as e:
        print(f"✗ Carry initialization failed: {e}")
        print(f"   Error details: {str(e)}")
        break

    batch_count = 0
    for step, (set_name, batch, eff_batch_size) in enumerate(train_loader):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Debug: print shapes before forward pass
            if step == 0:
                print(f"[Debug] Forward pass input shapes:")
                for k, v in batch.items():
                    print(f"  {k}: {v.shape}")

            # === Forward Pass ===
            carry, loss, metrics, detached_outputs, is_done = model_with_loss(
                carry=carry,
                batch=batch,
                return_keys=["logits", "q_halt_logits"]
            )

            if epoch == 0 and step == 0:  # Visualize only for first batch of first epoch
                predicted = torch.argmax(detached_outputs["logits"], dim=-1)
                print("[Predicted]:", predicted[0][:20].tolist())  # First 20 tokens
                print("[GroundTruth]:", batch["labels"][0][:20].tolist())

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[Warning] NaN loss detected at step {step}, skipping...")
                continue

            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * eff_batch_size
            total_samples += eff_batch_size
            batch_count += 1

            if batch_count % 5 == 0:  # Print progress every 5 batches
                print(f"  Step {batch_count}, Current loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"✗ Training step {step} failed: {e}")
            print(f"   Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
            # Continue with next batch instead of stopping
            continue

    if total_samples == 0:
        print("[Error] No data was processed in this epoch.")
        break

    avg_loss = total_loss / total_samples
    print(f"[Epoch {epoch + 1}] Processed {batch_count} batches, Avg Loss: {avg_loss:.4f}")
    losses.append(avg_loss)

# Save model----------------------------------------------------------------------------------------------
try:
    torch.save(model.state_dict(), "hrm_cpu_model.pth")
    print("✅ Model training complete and saved.")
except Exception as e:
    print(f"✗ Model saving failed: {e}")

# Plotting loss over epochs
plt.plot(losses, label="Training Loss", marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("training_loss.png")
plt.show()
