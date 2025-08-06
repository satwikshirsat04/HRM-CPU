import os
import json
import numpy as np
from pathlib import Path

from common import PuzzleDatasetMetadata

# Simple test data for CPU training
def create_simple_arc_dataset():
    """Create a minimal ARC-like dataset for testing"""
    
    output_dir = "data/arc-simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simple 5x5 puzzles
    ARCMaxGridSize = 30
    seq_len = ARCMaxGridSize * ARCMaxGridSize
    
    # Generate some simple patterns
    puzzles = []
    
    # Pattern 1: Copy input to output
    for i in range(5):
        inp = np.random.randint(2, 12, size=(5, 5))  # Random colors (2-11, since 0=pad, 1=eos)
        out = inp.copy()
        
        # Pad to 30x30
        inp_padded = np.zeros((ARCMaxGridSize, ARCMaxGridSize), dtype=np.uint8)
        out_padded = np.zeros((ARCMaxGridSize, ARCMaxGridSize), dtype=np.uint8)
        
        inp_padded[:5, :5] = inp
        out_padded[:5, :5] = out
        
        # Add EOS markers
        inp_padded[5, :5] = 1  # EOS row
        inp_padded[:5, 5] = 1  # EOS col
        out_padded[5, :5] = 1  # EOS row  
        out_padded[:5, 5] = 1  # EOS col
        
        puzzles.append((inp_padded.flatten(), out_padded.flatten()))
    
    # Pattern 2: Increment all colors by 1 (with wrap around)
    for i in range(5):
        inp = np.random.randint(2, 11, size=(5, 5))  # Colors 2-10
        out = inp.copy()
        out = np.where(out >= 11, 2, out + 1)  # Wrap around
        
        # Pad to 30x30
        inp_padded = np.zeros((ARCMaxGridSize, ARCMaxGridSize), dtype=np.uint8)
        out_padded = np.zeros((ARCMaxGridSize, ARCMaxGridSize), dtype=np.uint8)
        
        inp_padded[:5, :5] = inp
        out_padded[:5, :5] = out
        
        # Add EOS markers
        inp_padded[5, :5] = 1  # EOS row
        inp_padded[:5, 5] = 1  # EOS col
        out_padded[5, :5] = 1  # EOS row
        out_padded[:5, 5] = 1  # EOS col
        
        puzzles.append((inp_padded.flatten(), out_padded.flatten()))
    
    # Split into train and test
    train_puzzles = puzzles[:8]
    test_puzzles = puzzles[8:]
    
    # Create datasets
    for split_name, puzzle_list in [("train", train_puzzles), ("test", test_puzzles)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        inputs = []
        labels = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]
        
        example_id = 0
        puzzle_id = 0
        
        for inp, out in puzzle_list:
            inputs.append(inp)
            labels.append(out)
            puzzle_identifiers.append(1)  # All puzzles have identifier 1
            example_id += 1
            puzzle_id += 1
            puzzle_indices.append(example_id)
            
        group_indices.append(puzzle_id)
        
        # Save arrays
        np.save(os.path.join(split_dir, "all__inputs.npy"), np.stack(inputs))
        np.save(os.path.join(split_dir, "all__labels.npy"), np.stack(labels))
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), np.array(puzzle_identifiers, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), np.array(puzzle_indices, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__group_indices.npy"), np.array(group_indices, dtype=np.int32))
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=12,  # PAD + EOS + "0" ... "9"
            
            pad_id=0,
            ignore_label_id=0,
            
            blank_identifier_id=0,
            num_puzzle_identifiers=2,  # 0 (blank) + 1 (puzzle)
            
            total_groups=len(group_indices) - 1,
            mean_puzzle_examples=1.0,
            sets=["all"]
        )
        
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
    
    # Save identifiers mapping
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "simple_pattern"], f)
    
    print(f"Created simple ARC dataset in {output_dir}")
    print(f"Train puzzles: {len(train_puzzles)}")
    print(f"Test puzzles: {len(test_puzzles)}")

if __name__ == "__main__":
    create_simple_arc_dataset()