
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
from puzzle_dataset import PuzzleDataset
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from utils.functions import load_model_class

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
    epochs=20,  # Increased epochs for better training
    lr=1e-3,
    seed=42,

    # Model parameters
    seq_len=900,
    vocab_size=12,
    num_puzzle_identifiers=2,
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

def compute_metrics(logits, labels, verbose=False):
    """
    Compute accuracy and exact match metrics
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        verbose: If True, print detailed metrics
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    # Get predictions by taking argmax
    predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
    
    # Create mask for valid (non-ignored) tokens
    valid_mask = (labels != IGNORE_LABEL_ID)
    
    # Token-level accuracy
    correct_tokens = (predictions == labels) & valid_mask
    total_valid_tokens = valid_mask.sum().item()
    token_accuracy = correct_tokens.sum().item() / max(total_valid_tokens, 1)
    
    # Sequence-level exact match
    # A sequence is exactly correct if ALL its valid tokens are correct
    seq_correct = []
    for i in range(labels.shape[0]):
        valid_tokens_in_seq = valid_mask[i].sum().item()
        if valid_tokens_in_seq == 0:
            seq_correct.append(True)  # Empty sequences are considered correct
        else:
            correct_tokens_in_seq = correct_tokens[i].sum().item()
            seq_correct.append(correct_tokens_in_seq == valid_tokens_in_seq)
    
    exact_match_accuracy = sum(seq_correct) / len(seq_correct)
    
    # Additional metrics
    total_sequences = labels.shape[0]
    avg_sequence_length = total_valid_tokens / total_sequences
    
    metrics = {
        'token_accuracy': token_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'total_valid_tokens': total_valid_tokens,
        'total_sequences': total_sequences,
        'avg_sequence_length': avg_sequence_length,
        'correct_sequences': sum(seq_correct)
    }
    
    if verbose:
        print(f"Token Accuracy: {token_accuracy:.4f} ({correct_tokens.sum()}/{total_valid_tokens})")
        print(f"Exact Match Accuracy: {exact_match_accuracy:.4f} ({sum(seq_correct)}/{total_sequences})")
        print(f"Average Sequence Length: {avg_sequence_length:.2f}")
    
    return metrics

def visualize_prediction_sample(predictions, labels, sample_idx=0, max_tokens=50):
    """
    Visualize a sample prediction vs ground truth
    """
    pred_sample = predictions[sample_idx][:max_tokens].cpu().numpy()
    label_sample = labels[sample_idx][:max_tokens].cpu().numpy()
    
    # Filter out ignore tokens for visualization
    valid_indices = label_sample != IGNORE_LABEL_ID
    if valid_indices.any():
        valid_pred = pred_sample[valid_indices]
        valid_labels = label_sample[valid_indices]
        indices = np.arange(len(valid_pred))
        
        plt.figure(figsize=(12, 6))
        plt.plot(indices, valid_labels, 'o-', label='Ground Truth', linewidth=2, markersize=6)
        plt.plot(indices, valid_pred, 'x-', label='Prediction', linewidth=2, markersize=6)
        plt.xlabel('Token Index')
        plt.ylabel('Token ID')
        plt.title('First Sample: Prediction vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_vs_groundtruth.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[Predicted]: {valid_pred[:20].tolist()}")
        print(f"[GroundTruth]: {valid_labels[:20].tolist()}")
        print("🖼️ Saved image: prediction_vs_groundtruth.png")

def evaluate_on_test_set(model_with_loss, config):
    """
    Evaluate the model on test set
    """
    try:
        # Load test dataset
        test_dataset = PuzzleDataset(
            SimpleNamespace(**{**vars(config), 'test_set_mode': True}), 
            split='test'
        )
        test_loader = DataLoader(test_dataset, batch_size=None)
        
        model_with_loss.eval()
        all_metrics = []
        
        print("\n=== TEST SET EVALUATION ===")
        
        with torch.no_grad():
            for step, (set_name, batch, eff_batch_size) in enumerate(test_loader):
                batch = {k: v.to(torch.device("cpu")) for k, v in batch.items()}
                
                # Initialize carry for test
                carry = model_with_loss.model.initial_carry(batch=batch)
                
                # Forward pass
                carry, loss, metrics, detached_outputs, is_done = model_with_loss(
                    carry=carry,
                    batch=batch,
                    return_keys=["logits"]
                )
                
                # Compute our custom metrics
                custom_metrics = compute_metrics(detached_outputs["logits"], batch["labels"])
                all_metrics.append(custom_metrics)
                
                print(f"Test batch {step + 1}: Token Acc: {custom_metrics['token_accuracy']:.4f}, "
                      f"Exact Match: {custom_metrics['exact_match_accuracy']:.4f}")
        
        # Aggregate test metrics
        if all_metrics:
            avg_token_acc = np.mean([m['token_accuracy'] for m in all_metrics])
            avg_exact_match = np.mean([m['exact_match_accuracy'] for m in all_metrics])
            total_sequences = sum([m['total_sequences'] for m in all_metrics])
            total_correct_sequences = sum([m['correct_sequences'] for m in all_metrics])
            
            print(f"\n📊 FINAL TEST RESULTS:")
            print(f"Average Token Accuracy: {avg_token_acc:.4f}")
            print(f"Average Exact Match Accuracy: {avg_exact_match:.4f}")
            print(f"Total Correct Sequences: {total_correct_sequences}/{total_sequences}")
    
    except Exception as e:
        print(f"Test evaluation failed: {e}")

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

# ==== TRACKING METRICS ====
training_history = {
    'epoch': [],
    'loss': [],
    'token_accuracy': [],
    'exact_match_accuracy': [],
    'learning_rate': []
}

# ==== TRAIN LOOP ====
best_exact_match = 0.0

for epoch in range(config.epochs):
    total_loss = 0.0
    total_samples = 0
    epoch_metrics = []
    
    print(f"\n[Epoch {epoch + 1}/{config.epochs}] Starting training...")

    try:
        # Initialize carry for the new epoch using the first batch data
        batch_for_carry = {k: v.to(device) for k, v in first_batch_data.items()}
        carry = model_with_loss.model.initial_carry(batch=batch_for_carry)
        print("✓ Carry initialized successfully")
    except Exception as e:
        print(f"✗ Carry initialization failed: {e}")
        break

    model_with_loss.train()
    batch_count = 0
    
    for step, (set_name, batch, eff_batch_size) in enumerate(train_loader):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}

            # === Forward Pass ===
            carry, loss, metrics, detached_outputs, is_done = model_with_loss(
                carry=carry,
                batch=batch,
                return_keys=["logits", "q_halt_logits"]
            )

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[Warning] NaN loss detected at step {step}, skipping...")
                continue

            # === Compute Custom Metrics ===
            custom_metrics = compute_metrics(detached_outputs["logits"], batch["labels"])
            epoch_metrics.append(custom_metrics)

            # === Backward Pass ===
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * eff_batch_size
            total_samples += eff_batch_size
            batch_count += 1

            # Print progress
            if batch_count % 5 == 0 or batch_count == 1:
                print(f"  Batch {batch_count}: Loss: {loss.item():.4f}, "
                      f"Token Acc: {custom_metrics['token_accuracy']:.4f}, "
                      f"Exact Match: {custom_metrics['exact_match_accuracy']:.4f}")
                
        except Exception as e:
            print(f"✗ Training step {step} failed: {e}")
            continue

    if total_samples == 0:
        print("[Error] No data was processed in this epoch.")
        break

    # === EPOCH SUMMARY ===
    avg_loss = total_loss / total_samples
    avg_token_acc = np.mean([m['token_accuracy'] for m in epoch_metrics])
    avg_exact_match = np.mean([m['exact_match_accuracy'] for m in epoch_metrics])
    
    # Store metrics
    training_history['epoch'].append(epoch + 1)
    training_history['loss'].append(avg_loss)
    training_history['token_accuracy'].append(avg_token_acc)
    training_history['exact_match_accuracy'].append(avg_exact_match)
    training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    print(f"\n📊 [Epoch {epoch + 1}] Summary:")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   Token Accuracy: {avg_token_acc:.4f}")
    print(f"   Exact Match Accuracy: {avg_exact_match:.4f}")
    print(f"   Processed {batch_count} batches")
    
    # Save best model
    if avg_exact_match > best_exact_match:
        best_exact_match = avg_exact_match
        torch.save(model.state_dict(), "hrm_best_model.pth")
        print(f"   🏆 New best exact match: {best_exact_match:.4f} - Model saved!")
    
    # Visualize sample prediction (first epoch and every 5 epochs)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            model_with_loss.eval()
            # Get a fresh batch for visualization
            vis_batch = {k: v.to(device) for k, v in first_batch_data.items()}
            vis_carry = model_with_loss.model.initial_carry(batch=vis_batch)
            _, _, _, vis_outputs, _ = model_with_loss(
                carry=vis_carry, batch=vis_batch, return_keys=["logits"]
            )
            predictions = torch.argmax(vis_outputs["logits"], dim=-1)
            visualize_prediction_sample(predictions, vis_batch["labels"])

# === POST-TRAINING VISUALIZATION ===
print("\n📈 Creating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training Loss
axes[0, 0].plot(training_history['epoch'], training_history['loss'], 'o-', linewidth=2)
axes[0, 0].set_title('Training Loss over Epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Token Accuracy
axes[0, 1].plot(training_history['epoch'], training_history['token_accuracy'], 'o-', color='green', linewidth=2)
axes[0, 1].set_title('Token Accuracy over Epochs')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Token Accuracy')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1)

# Exact Match Accuracy
axes[1, 0].plot(training_history['epoch'], training_history['exact_match_accuracy'], 'o-', color='red', linewidth=2)
axes[1, 0].set_title('Exact Match Accuracy over Epochs')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Exact Match Accuracy')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(0, 1)

# Combined Metrics
axes[1, 1].plot(training_history['epoch'], training_history['token_accuracy'], 'o-', color='green', label='Token Accuracy', linewidth=2)
axes[1, 1].plot(training_history['epoch'], training_history['exact_match_accuracy'], 'o-', color='red', label='Exact Match', linewidth=2)
axes[1, 1].set_title('Combined Accuracy Metrics')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print("🖼️ Saved training metrics: training_metrics.png")

# === FINAL EVALUATION ON TEST SET ===
evaluate_on_test_set(model_with_loss, config)

# === SAVE FINAL MODEL ===
try:
    torch.save(model.state_dict(), "hrm_final_model.pth")
    
    # Save training history
    import json
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   📁 Final model saved: hrm_final_model.pth")
    print(f"   🏆 Best model saved: hrm_best_model.pth (Exact Match: {best_exact_match:.4f})")
    print(f"   📊 Training history saved: training_history.json")
    print(f"   📈 Metrics plots saved: training_metrics.png")
    
except Exception as e:
    print(f"✗ Model saving failed: {e}")