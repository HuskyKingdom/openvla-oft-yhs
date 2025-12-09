"""
Diagnose RLDS dataset structure to understand data format
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import sys

tf.config.set_visible_devices([], 'GPU')

def diagnose_rlds(data_dir, suite_name):
    """Diagnose RLDS dataset structure"""
    dataset_path = data_dir + "/" + suite_name + "/1.0.0/"
    
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        builder = tfds.builder_from_directory(dataset_path)
        dataset = builder.as_dataset(split='train')
        
        print(f"\n✓ Successfully loaded dataset")
        print(f"Dataset info: {builder.info}")
        
        # Get first episode
        for episode_idx, episode in enumerate(dataset.take(1)):
            print(f"\n{'='*60}")
            print(f"EPISODE {episode_idx} STRUCTURE")
            print(f"{'='*60}")
            
            print(f"\nEpisode keys: {list(episode.keys())}")
            
            for key in episode.keys():
                if key == 'steps':
                    print(f"\n  '{key}': Dataset (contains steps)")
                    
                    # Examine first step
                    for step_idx, step in enumerate(episode['steps'].take(1)):
                        print(f"\n  First step keys: {list(step.keys())}")
                        
                        for step_key in step.keys():
                            if step_key == 'observation':
                                print(f"\n    '{step_key}' keys: {list(step[step_key].keys())}")
                                
                                # Print observation details
                                for obs_key in step['observation'].keys():
                                    obs_val = step['observation'][obs_key]
                                    print(f"      '{obs_key}': shape={obs_val.shape}, dtype={obs_val.dtype}")
                                    
                                    # Try to decode if it looks like text
                                    if obs_val.dtype == tf.string:
                                        try:
                                            decoded = obs_val.numpy().decode('utf-8')
                                            print(f"        → Value: '{decoded}'")
                                        except:
                                            print(f"        → (cannot decode)")
                            
                            elif step_key == 'action':
                                print(f"    '{step_key}': shape={step[step_key].shape}, dtype={step[step_key].dtype}")
                            else:
                                val = step[step_key]
                                print(f"    '{step_key}': {type(val)}")
                                if hasattr(val, 'shape'):
                                    print(f"      shape={val.shape}, dtype={val.dtype}")
                                if hasattr(val, 'numpy') and val.dtype == tf.string:
                                    try:
                                        decoded = val.numpy().decode('utf-8')
                                        print(f"      → Value: '{decoded}'")
                                    except:
                                        pass
                else:
                    val = episode[key]
                    print(f"\n  '{key}': {type(val)}")
                    if hasattr(val, 'shape'):
                        print(f"    shape={val.shape}, dtype={val.dtype}")
                    if hasattr(val, 'numpy'):
                        if val.dtype == tf.string:
                            try:
                                decoded = val.numpy().decode('utf-8')
                                print(f"    → Value: '{decoded}'")
                            except:
                                print(f"    → (cannot decode)")
                        elif val.shape == ():  # scalar
                            print(f"    → Value: {val.numpy()}")
            
            print(f"\n{'='*60}")
            print("Diagnosis complete!")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_rlds.py <data_dir> <suite_name>")
        print("Example: python diagnose_rlds.py /path/to/modified_libero_rlds libero_spatial_no_noops")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    suite_name = sys.argv[2]
    
    diagnose_rlds(data_dir, suite_name)

