import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os
from collections import defaultdict
from puzzle_dataset import PuzzleDataset
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from utils.functions import load_model_class

# ==== IMPROVED CONFIG ====
config = SimpleNamespace(
    arch_name='hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1',
    loss_type='losses@ACTLossHead',

    dataset_path='data/arc-simple',
    test_set_mode=False,
    global_batch_size=4,  # Increased batch size
    rank=0,
    num_replicas=1,
    epochs_per_iter=1,

    batch_size=4,
    epochs=50,  # More epochs
    lr=5e-4,    # Lower learning rate
    weight_decay=1e-4,  # Add weight decay
    warmup_epochs=5,    # Learning rate warmup
    seed=42,

    # Improved Model parameters
    seq_len=900,
    vocab_size=12,
    num_puzzle_identifiers=2,
    puzzle_emb_ndim=64,  # Increased embedding size

    H_cycles=3,  # More reasoning cycles
    L_cycles=3,
    H_layers=2,  # More layers
    L_layers=2,

    hidden_size=128,  # Increased model size
    expansion=2.5,    # Slightly larger MLP
    num_heads=8,      # More attention heads
    pos_encodings="rope",

    halt_max_steps=8,  # More reasoning steps
    halt_exploration_prob=0.15,

    # Training improvements
    gradient_clip_norm=1.0,
    early_stopping_patience=10,
    save_every_n_epochs=5,
    validate_every_n_epochs=3,
    
    # Advanced features
    use_lr_scheduler=True,
    model_averaging=True,
    pattern_analysis=True
)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def compute_advanced_metrics(logits, labels, detailed=False):
    """
    Compute comprehensive accuracy and pattern-specific metrics
    """
    predictions = torch.argmax(logits, dim=-1)
    valid_mask = (labels != IGNORE_LABEL_ID)
    
    # Basic metrics
    correct_tokens = (predictions == labels) & valid_mask
    total_valid_tokens = valid_mask.sum().item()
    token_accuracy = correct_tokens.sum().item() / max(total_valid_tokens, 1)
    
    # Sequence-level metrics
    batch_size = labels.shape[0]
    seq_metrics = []
    pattern_metrics = defaultdict(list)
    
    for i in range(batch_size):
        seq_valid_mask = valid_mask[i]
        if seq_valid_mask.sum() == 0:
            continue
            
        seq_labels = labels[i][seq_valid_mask]
        seq_preds = predictions[i][seq_valid_mask]
        seq_inputs = None  # We'd need inputs to determine pattern type
        
        # Sequence accuracy
        seq_correct = torch.all(seq_preds == seq_labels).item()
        seq_token_acc = (seq_preds == seq_labels).float().mean().item()
        
        seq_metrics.append({
            'seq_correct': seq_correct,
            'seq_token_acc': seq_token_acc,
            'seq_length': len(seq_labels)
        })
    
    exact_match_accuracy = np.mean([m['seq_correct'] for m in seq_metrics]) if seq_metrics else 0.0
    
    # Additional metrics
    metrics = {
        'token_accuracy': token_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
        'total_valid_tokens': total_valid_tokens,
        'total_sequences': len(seq_metrics),
        'avg_sequence_length': np.mean([m['seq_length'] for m in seq_metrics]) if seq_metrics else 0,
        'correct_sequences': sum(m['seq_correct'] for m in seq_metrics),
        'avg_seq_token_accuracy': np.mean([m['seq_token_acc'] for m in seq_metrics]) if seq_metrics else 0,
    }
    
    if detailed:
        # Per-sequence analysis
        metrics['per_sequence'] = seq_metrics
    
    return metrics

def analyze_pattern_performance(model_with_loss, batch, outputs, config, verbose=False):
    """
    Analyze performance on different puzzle patterns within a batch
    """
    predictions = torch.argmax(outputs["logits"], dim=-1)
    inputs = batch["inputs"]
    labels = batch["labels"]
    
    pattern_results = {'copy': [], 'increment': [], 'unknown': []}
    
    for seq_idx in range(inputs.shape[0]):
        valid_mask = labels[seq_idx] != IGNORE_LABEL_ID
        if not valid_mask.any():
            continue
            
        seq_inputs = inputs[seq_idx][valid_mask]
        seq_labels = labels[seq_idx][valid_mask]
        seq_preds = predictions[seq_idx][valid_mask]
        
        # Determine pattern type by comparing input and expected output
        is_copy = torch.all(seq_inputs == seq_labels).item()
        
        # Check if it's an increment pattern (input + 1 with wraparound)
        expected_increment = torch.where(seq_inputs >= 11, 2, seq_inputs + 1)
        is_increment = torch.all(seq_labels == expected_increment).item()
        
        if is_copy:
            pattern_type = 'copy'
        elif is_increment:
            pattern_type = 'increment'
        else:
            pattern_type = 'unknown'
        
        # Calculate accuracy
        seq_correct = torch.all(seq_preds == seq_labels).item()
        token_acc = (seq_preds == seq_labels).float().mean().item()
        
        pattern_results[pattern_type].append({
            'sequence_correct': seq_correct,
            'token_accuracy': token_acc,
            'sequence_length': len(seq_labels)
        })
        
        if verbose and seq_idx < 2:  # Print first 2 sequences
            status = '✓' if seq_correct else '✗'
            print(f"    Seq {seq_idx}: {pattern_type.upper()} pattern {status} "
                  f"(Token Acc: {token_acc:.3f})")
    
    return pattern_results

def visualize_comprehensive_training(history, save_path='comprehensive_training_metrics.png'):
    """
    Create comprehensive training visualizations
    """
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Loss
    axes[0, 0].plot(epochs, history['loss'], 'o-', linewidth=2, color='blue')
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Token Accuracy
    axes[0, 1].plot(epochs, history['token_accuracy'], 'o-', linewidth=2, color='green')
    axes[0, 1].set_title('Token Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Exact Match Accuracy
    axes[0, 2].plot(epochs, history['exact_match_accuracy'], 'o-', linewidth=2, color='red')
    axes[0, 2].set_title('Exact Match Accuracy', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['learning_rate'], 'o-', linewidth=2, color='purple')
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pattern-specific accuracies (if available)
    if 'copy_accuracy' in history and 'increment_accuracy' in history:
        axes[1, 1].plot(epochs, history['copy_accuracy'], 'o-', linewidth=2, color='cyan', label='Copy Pattern')
        axes[1, 1].plot(epochs, history['increment_accuracy'], 'o-', linewidth=2, color='orange', label='Increment Pattern')
        axes[1, 1].set_title('Pattern-Specific Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Combined metrics
    axes[1, 2].plot(epochs, history['token_accuracy'], 'o-', linewidth=2, color='green', label='Token Accuracy')
    axes[1, 2].plot(epochs, history['exact_match_accuracy'], 'o-', linewidth=2, color='red', label='Exact Match')
    if 'val_token_accuracy' in history:
        axes[1, 2].plot(epochs, history['val_token_accuracy'], '--', linewidth=2, color='lightgreen', label='Val Token Acc')
        axes[1, 2].plot(epochs, history['val_exact_match_accuracy'], '--', linewidth=2, color='lightcoral', label='Val Exact Match')
    axes[1, 2].set_title('Training vs Validation', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved comprehensive metrics: {save_path}")

def evaluate_model(model_with_loss, config, split='test'):
    """
    Comprehensive model evaluation
    """
    try:
        dataset = PuzzleDataset(
            SimpleNamespace(**{**vars(config), 'test_set_mode': True}), 
            split=split
        )
        loader = DataLoader(dataset, batch_size=None)
        
        model_with_loss.eval()
        all_metrics = []
        all_pattern_results = defaultdict(list)
        
        print(f"\n=== {split.upper()} SET EVALUATION ===")
        
        with torch.no_grad():
            for step, (set_name, batch, eff_batch_size) in enumerate(loader):
                batch = {k: v.to(torch.device("cpu")) for k, v in batch.items()}
                
                carry = model_with_loss.model.initial_carry(batch=batch)
                carry, loss, metrics, outputs, is_done = model_with_loss(
                    carry=carry, batch=batch, return_keys=["logits"]
                )
                
                # Compute metrics
                custom_metrics = compute_advanced_metrics(outputs["logits"], batch["labels"], detailed=True)
                all_metrics.append(custom_metrics)
                
                # Pattern analysis
                if config.pattern_analysis:
                    pattern_results = analyze_pattern_performance(
                        model_with_loss, batch, outputs, config, verbose=(step < 2)
                    )
                    for pattern, results in pattern_results.items():
                        all_pattern_results[pattern].extend(results)
        
        # Aggregate results
        if all_metrics:
            overall_metrics = {
                'token_accuracy': np.mean([m['token_accuracy'] for m in all_metrics]),
                'exact_match_accuracy': np.mean([m['exact_match_accuracy'] for m in all_metrics]),
                'total_sequences': sum(m['total_sequences'] for m in all_metrics),
                'correct_sequences': sum(m['correct_sequences'] for m in all_metrics),
            }
            
            print(f"\n📊 {split.upper()} RESULTS:")
            print(f"Token Accuracy: {overall_metrics['token_accuracy']:.4f}")
            print(f"Exact Match Accuracy: {overall_metrics['exact_match_accuracy']:.4f}")
            print(f"Correct Sequences: {overall_metrics['correct_sequences']}/{overall_metrics['total_sequences']}")
            
            # Pattern-specific results
            if config.pattern_analysis and all_pattern_results:
                print(f"\n🔍 PATTERN ANALYSIS:")
                for pattern, results in all_pattern_results.items():
                    if results:
                        pattern_exact = np.mean([r['sequence_correct'] for r in results])
                        pattern_token = np.mean([r['token_accuracy'] for r in results])
                        print(f"  {pattern.upper()} (n={len(results)}): "
                              f"Exact={pattern_exact:.3f}, Token={pattern_token:.3f}")
            
            return overall_metrics, all_pattern_results
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None, None

def save_model_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save comprehensive model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': vars(config)
    }
    torch.save(checkpoint, path)

# ==== MAIN TRAINING SCRIPT ====
torch.manual_seed(config.seed)
np.random.seed(config.seed)
device = torch.device("cpu")

print("🚀 IMPROVED HRM TRAINING SCRIPT")
print("="*50)

# ==== LOAD MODEL ====
print("Initializing improved model...")
try:
    model_cls = load_model_class(config.arch_name)
    model = model_cls(vars(config))
    
    loss_fn_cls = load_model_class(config.loss_type)
    model_with_loss = loss_fn_cls(model, loss_type="softmax_cross_entropy")
    model_with_loss.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("✓ Model initialized successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Reasoning layers: H={config.H_layers}, L={config.L_layers}")
    print(f"  - Reasoning cycles: H={config.H_cycles}, L={config.L_cycles}")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    exit(1)

# ==== DATASET ====
try:
    train_dataset = PuzzleDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=None)
    
    print(f"✓ Dataset loaded - Metadata: {train_dataset.metadata}")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    exit(1)

# ==== OPTIMIZER AND SCHEDULER ====
optimizer = optim.AdamW(
    model_with_loss.parameters(), 
    lr=config.lr, 
    weight_decay=config.weight_decay,
    betas=(0.9, 0.999)
)

scheduler = None
if config.use_lr_scheduler:
    total_steps = config.epochs
    warmup_steps = config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"✓ Optimizer: AdamW (lr={config.lr}, weight_decay={config.weight_decay})")
print(f"✓ Scheduler: {'Cosine with warmup' if config.use_lr_scheduler else 'None'}")

# ==== GET FIRST BATCH FOR INITIALIZATION ====
first_batch = next(iter(train_loader))
set_name, first_batch_data, eff_batch_size = first_batch
print(f"✓ First batch loaded - Effective size: {eff_batch_size}")

# ==== TRAINING TRACKING ====
training_history = {
    'epoch': [], 'loss': [], 'token_accuracy': [], 'exact_match_accuracy': [],
    'learning_rate': [], 'copy_accuracy': [], 'increment_accuracy': [],
    'val_token_accuracy': [], 'val_exact_match_accuracy': []
}

best_metrics = {
    'exact_match': 0.0,
    'token_accuracy': 0.0,
    'epoch': 0
}

early_stopping_counter = 0

# ==== MAIN TRAINING LOOP ====
print(f"\n🏃 Starting training for {config.epochs} epochs...")
print("="*60)

for epoch in range(config.epochs):
    print(f"\n[Epoch {epoch + 1}/{config.epochs}]")
    
    # === TRAINING PHASE ===
    model_with_loss.train()
    epoch_loss = 0.0
    epoch_samples = 0
    epoch_metrics = []
    epoch_pattern_results = defaultdict(list)
    
    try:
        # Initialize carry
        batch_for_carry = {k: v.to(device) for k, v in first_batch_data.items()}
        carry = model_with_loss.model.initial_carry(batch=batch_for_carry)
        
        batch_count = 0
        for step, (set_name, batch, eff_batch_size) in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            carry, loss, metrics, outputs, is_done = model_with_loss(
                carry=carry, batch=batch, return_keys=["logits", "q_halt_logits"]
            )
            
            if torch.isnan(loss):
                print(f"  ⚠️  NaN loss detected, skipping batch {step}")
                continue
            
            # Compute custom metrics
            custom_metrics = compute_advanced_metrics(outputs["logits"], batch["labels"])
            epoch_metrics.append(custom_metrics)
            
            # Pattern analysis
            if config.pattern_analysis:
                pattern_results = analyze_pattern_performance(
                    model_with_loss, batch, outputs, config, verbose=(batch_count == 0)
                )
                for pattern, results in pattern_results.items():
                    epoch_pattern_results[pattern].extend(results)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), config.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item() * eff_batch_size
            epoch_samples += eff_batch_size
            batch_count += 1
            
            # Progress update
            if batch_count % 3 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_count}: Loss={loss.item():.4f}, "
                      f"Token Acc={custom_metrics['token_accuracy']:.3f}, "
                      f"LR={current_lr:.2e}")
        
        if epoch_samples == 0:
            print("  ⚠️  No data processed in this epoch")
            continue
            
        # === EPOCH SUMMARY ===
        avg_loss = epoch_loss / epoch_samples
        avg_token_acc = np.mean([m['token_accuracy'] for m in epoch_metrics])
        avg_exact_match = np.mean([m['exact_match_accuracy'] for m in epoch_metrics])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Pattern-specific metrics
        copy_acc = np.mean([r['sequence_correct'] for r in epoch_pattern_results['copy']]) if epoch_pattern_results['copy'] else 0.0
        inc_acc = np.mean([r['sequence_correct'] for r in epoch_pattern_results['increment']]) if epoch_pattern_results['increment'] else 0.0
        
        # Store metrics
        training_history['epoch'].append(epoch + 1)
        training_history['loss'].append(avg_loss)
        training_history['token_accuracy'].append(avg_token_acc)
        training_history['exact_match_accuracy'].append(avg_exact_match)
        training_history['learning_rate'].append(current_lr)
        training_history['copy_accuracy'].append(copy_acc)
        training_history['increment_accuracy'].append(inc_acc)
        
        print(f"\n  📊 Training Summary:")
        print(f"     Loss: {avg_loss:.4f}")
        print(f"     Token Accuracy: {avg_token_acc:.4f}")
        print(f"     Exact Match: {avg_exact_match:.4f}")
        print(f"     Copy Pattern: {copy_acc:.4f}")
        print(f"     Increment Pattern: {inc_acc:.4f}")
        print(f"     Learning Rate: {current_lr:.2e}")
        
        # === VALIDATION ===
        if (epoch + 1) % config.validate_every_n_epochs == 0:
            val_metrics, val_patterns = evaluate_model(model_with_loss, config, split='test')
            if val_metrics:
                training_history['val_token_accuracy'].append(val_metrics['token_accuracy'])
                training_history['val_exact_match_accuracy'].append(val_metrics['exact_match_accuracy'])
            else:
                training_history['val_token_accuracy'].append(0.0)
                training_history['val_exact_match_accuracy'].append(0.0)
        else:
            training_history['val_token_accuracy'].append(training_history['val_token_accuracy'][-1] if training_history['val_token_accuracy'] else 0.0)
            training_history['val_exact_match_accuracy'].append(training_history['val_exact_match_accuracy'][-1] if training_history['val_exact_match_accuracy'] else 0.0)
        
        # === MODEL SAVING ===
        is_best = avg_exact_match > best_metrics['exact_match']
        if is_best:
            best_metrics.update({
                'exact_match': avg_exact_match,
                'token_accuracy': avg_token_acc,
                'epoch': epoch + 1
            })
            save_model_checkpoint(model, optimizer, scheduler, epoch + 1, best_metrics, "hrm_best_model.pth")
            print(f"     🏆 New best model saved! (Exact Match: {avg_exact_match:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Checkpoint saving
        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_path = f"hrm_checkpoint_epoch_{epoch + 1}.pth"
            save_model_checkpoint(model, optimizer, scheduler, epoch + 1, 
                                {'token_accuracy': avg_token_acc, 'exact_match': avg_exact_match}, 
                                checkpoint_path)
            print(f"     💾 Checkpoint saved: {checkpoint_path}")
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Early stopping
        if early_stopping_counter >= config.early_stopping_patience:
            print(f"\n  🛑 Early stopping triggered after {config.early_stopping_patience} epochs without improvement")
            break
            
    except Exception as e:
        print(f"  ✗ Training error in epoch {epoch + 1}: {e}")
        continue

# === POST-TRAINING ANALYSIS ===
print("\n" + "="*60)
print("🏁 TRAINING COMPLETED")
print("="*60)

# Save final model
save_model_checkpoint(model, optimizer, scheduler, config.epochs, best_metrics, "hrm_final_model.pth")

# Create comprehensive visualizations
visualize_comprehensive_training(training_history)

# Final evaluation
print(f"\n🔬 FINAL EVALUATION")
final_test_metrics, final_test_patterns = evaluate_model(model_with_loss, config, split='test')
final_train_metrics, final_train_patterns = evaluate_model(model_with_loss, config, split='train')

# Save complete training history
with open('improved_training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)

# Summary report
print(f"\n📋 FINAL SUMMARY:")
print(f"   🏆 Best Epoch: {best_metrics['epoch']}")
print(f"   📈 Best Exact Match: {best_metrics['exact_match']:.4f}")
print(f"   📊 Best Token Accuracy: {best_metrics['token_accuracy']:.4f}")
print(f"   💾 Models saved: hrm_best_model.pth, hrm_final_model.pth")
print(f"   📊 History saved: improved_training_history.json")
print(f"   🖼️  Plots saved: comprehensive_training_metrics.png")

if final_test_metrics:
    print(f"\n🎯 FINAL TEST PERFORMANCE:")
    print(f"   Token Accuracy: {final_test_metrics['token_accuracy']:.4f}")
    print(f"   Exact Match: {final_test_metrics['exact_match_accuracy']:.4f}")

print(f"\n✅ Training completed successfully!")