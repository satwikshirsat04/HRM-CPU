import torch
import numpy as np
from puzzle_dataset import PuzzleDataset
from models.losses import ACTLossHead, IGNORE_LABEL_ID
from utils.functions import load_model_class
from types import SimpleNamespace

def analyze_pattern_performance(model_path="hrm_best_model.pth"):
    """
    Analyze performance on different puzzle patterns
    """
    config = SimpleNamespace(
        arch_name='hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1',
        loss_type='losses@ACTLossHead',
        dataset_path='data/arc-simple',
        test_set_mode=True,
        global_batch_size=2,
        rank=0,
        num_replicas=1,
        epochs_per_iter=1,
        seq_len=900,
        vocab_size=12,
        num_puzzle_identifiers=2,
        puzzle_emb_ndim=32,
        H_cycles=2, L_cycles=2, H_layers=1, L_layers=1,
        hidden_size=64, expansion=2.0, num_heads=4,
        pos_encodings="rope", halt_max_steps=6, halt_exploration_prob=0.1
    )
    
    # Load model
    model_cls = load_model_class(config.arch_name)
    model = model_cls(vars(config))
    loss_fn_cls = load_model_class(config.loss_type)
    model_with_loss = loss_fn_cls(model, loss_type="softmax_cross_entropy")
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"✓ Loaded model from {model_path}")
    except:
        print(f"⚠️ Could not load {model_path}, using random weights")
    
    model_with_loss.eval()
    
    # Test on both train and test sets
    for split in ['train', 'test']:
        print(f"\n=== ANALYZING {split.upper()} SET ===")
        
        dataset = PuzzleDataset(
            SimpleNamespace(**{**vars(config), 'test_set_mode': True}), 
            split=split
        )
        
        results = []
        
        with torch.no_grad():
            for batch_idx, (set_name, batch, eff_batch_size) in enumerate(dataset):
                carry = model_with_loss.model.initial_carry(batch=batch)
                
                _, loss, metrics, outputs, _ = model_with_loss(
                    carry=carry, batch=batch, return_keys=["logits"]
                )
                
                predictions = torch.argmax(outputs["logits"], dim=-1)
                
                # Analyze each sequence in the batch
                for seq_idx in range(batch["inputs"].shape[0]):
                    input_seq = batch["inputs"][seq_idx]
                    label_seq = batch["labels"][seq_idx]
                    pred_seq = predictions[seq_idx]
                    
                    # Get valid positions (non-ignored)
                    valid_mask = label_seq != IGNORE_LABEL_ID
                    if not valid_mask.any():
                        continue
                        
                    valid_inputs = input_seq[valid_mask]
                    valid_labels = label_seq[valid_mask]
                    valid_preds = pred_seq[valid_mask]
                    
                    # Determine pattern type by comparing input and output
                    is_copy_pattern = torch.all(valid_inputs == valid_labels)
                    
                    # Calculate accuracy
                    seq_correct = torch.all(valid_preds == valid_labels)
                    token_acc = (valid_preds == valid_labels).float().mean()
                    
                    results.append({
                        'batch_idx': batch_idx,
                        'seq_idx': seq_idx,
                        'pattern_type': 'copy' if is_copy_pattern else 'increment',
                        'sequence_correct': seq_correct.item(),
                        'token_accuracy': token_acc.item(),
                        'sequence_length': valid_mask.sum().item()
                    })
                    
                    # Print detailed analysis for first few examples
                    if batch_idx < 2:
                        pattern_type = 'COPY' if is_copy_pattern else 'INCREMENT'
                        status = '✓ CORRECT' if seq_correct else '✗ WRONG'
                        print(f"  Batch {batch_idx}, Seq {seq_idx}: {pattern_type} pattern - {status}")
                        print(f"    Token Accuracy: {token_acc:.3f}")
                        if not seq_correct:
                            print(f"    Input:  {valid_inputs[:10].tolist()}")
                            print(f"    Target: {valid_labels[:10].tolist()}")
                            print(f"    Pred:   {valid_preds[:10].tolist()}")
        
        # Aggregate results by pattern type
        copy_results = [r for r in results if r['pattern_type'] == 'copy']
        increment_results = [r for r in results if r['pattern_type'] == 'increment']
        
        print(f"\n📊 PATTERN ANALYSIS for {split.upper()}:")
        
        if copy_results:
            copy_exact = np.mean([r['sequence_correct'] for r in copy_results])
            copy_token = np.mean([r['token_accuracy'] for r in copy_results])
            print(f"  COPY Pattern (n={len(copy_results)}):")
            print(f"    Exact Match: {copy_exact:.3f}")
            print(f"    Token Accuracy: {copy_token:.3f}")
        
        if increment_results:
            inc_exact = np.mean([r['sequence_correct'] for r in increment_results])
            inc_token = np.mean([r['token_accuracy'] for r in increment_results])
            print(f"  INCREMENT Pattern (n={len(increment_results)}):")
            print(f"    Exact Match: {inc_exact:.3f}")
            print(f"    Token Accuracy: {inc_token:.3f}")
        
        overall_exact = np.mean([r['sequence_correct'] for r in results])
        overall_token = np.mean([r['token_accuracy'] for r in results])
        print(f"  OVERALL:")
        print(f"    Exact Match: {overall_exact:.3f}")
        print(f"    Token Accuracy: {overall_token:.3f}")

if __name__ == "__main__":
    analyze_pattern_performance()