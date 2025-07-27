#!/usr/bin/env python3
"""
Automatic Checkpoint Renamer
Automatically renames existing checkpoint files to use the NEW consistent seed generation
No prompts - fully automatic with smart conflict resolution
"""

import os
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict
import shutil
from datetime import datetime

class AutomaticCheckpointRenamer:
    """Automatically renames existing checkpoints to use new consistent seed format"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backup_dir = Path(f"{checkpoint_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    def generate_new_consistent_seed(self, grid_size, n_agents, pattern, trial_id):
        """
        Generate seed using the NEW method from the fixed experiment file
        This MUST match the generate_consistent_seed() function in complete_distributed_experiment.py
        """
        seed_string = f"{grid_size[0]}x{grid_size[1]}_{n_agents}agents_{pattern}_trial{trial_id}"
        hash_object = hashlib.sha256(seed_string.encode())
        hash_hex = hash_object.hexdigest()
        seed = int(hash_hex[:8], 16) % (2**31 - 1)
        return seed
    
    def parse_checkpoint_filename(self, filename):
        """Parse checkpoint filename to extract components"""
        try:
            # Example: grid10x10_agents2_evasive_trial0_interval0_seed1353.pkl
            parts = filename.replace('.pkl', '').split('_')
            
            # Extract grid size
            grid_part = [p for p in parts if p.startswith('grid')][0]
            grid_str = grid_part.replace('grid', '')
            rows, cols = map(int, grid_str.split('x'))
            grid_size = (rows, cols)
            
            # Extract agents
            agent_part = [p for p in parts if p.startswith('agents')][0]
            n_agents = int(agent_part.replace('agents', ''))
            
            # Extract pattern (find part that's not grid, agents, trial, interval, or seed)
            pattern = None
            for i, part in enumerate(parts):
                if (not part.startswith(('grid', 'agents', 'trial', 'interval', 'seed')) and 
                    part not in ['agents', 'trial', 'interval', 'seed']):
                    pattern = part
                    break
            
            # Extract trial
            trial_part = [p for p in parts if p.startswith('trial')][0]
            trial_id = int(trial_part.replace('trial', ''))
            
            # Extract interval
            interval_part = [p for p in parts if p.startswith('interval')][0]
            interval_str = interval_part.replace('interval', '')
            if interval_str == 'inf':
                merge_interval = float('inf')
            else:
                merge_interval = int(interval_str)
            
            # Extract old seed
            seed_part = [p for p in parts if p.startswith('seed')][0]
            old_seed = int(seed_part.replace('seed', ''))
            
            return {
                'grid_size': grid_size,
                'n_agents': n_agents,
                'pattern': pattern,
                'trial_id': trial_id,
                'merge_interval': merge_interval,
                'old_seed': old_seed,
                'valid': True
            }
        except Exception as e:
            print(f"Warning: Could not parse {filename}: {e}")
            return {'valid': False}
    
    def create_backup(self):
        """Create backup of existing checkpoints"""
        print(f"Creating backup in {self.backup_dir}...")
        self.backup_dir.mkdir(exist_ok=True)
        
        checkpoint_files = list(self.checkpoint_dir.glob("*.pkl"))
        error_files = list(self.checkpoint_dir.glob("*_ERROR.txt"))
        all_files = checkpoint_files + error_files
        
        for file_path in all_files:
            shutil.copy2(file_path, self.backup_dir / file_path.name)
        
        print(f"✅ Backed up {len(all_files)} files")
        return len(all_files)
    
    def automatic_rename_all(self):
        """Automatically rename all files with smart conflict resolution"""
        
        print("🔍 Analyzing checkpoint files...")
        checkpoint_files = list(self.checkpoint_dir.glob("*.pkl"))
        print(f"Found {len(checkpoint_files)} checkpoint files")
        
        if not checkpoint_files:
            print("No checkpoint files found.")
            return
        
        # Group files by logical task to handle duplicates
        task_groups = defaultdict(list)
        unparseable_files = []
        
        for file_path in checkpoint_files:
            parsed = self.parse_checkpoint_filename(file_path.name)
            
            if not parsed['valid']:
                unparseable_files.append(file_path)
                continue
            
            # Create task key (without seed)
            task_key = (
                parsed['grid_size'], parsed['n_agents'], 
                parsed['pattern'], parsed['trial_id'], 
                parsed['merge_interval']
            )
            
            task_groups[task_key].append({
                'file_path': file_path,
                'parsed': parsed,
                'mtime': file_path.stat().st_mtime
            })
        
        print(f"📊 Analysis complete:")
        print(f"   - Logical tasks found: {len(task_groups)}")
        print(f"   - Duplicate tasks: {len([k for k, v in task_groups.items() if len(v) > 1])}")
        print(f"   - Unparseable files: {len(unparseable_files)}")
        
        # Create backup
        backup_count = self.create_backup()
        
        # Process each task group
        renamed_count = 0
        removed_duplicates = 0
        error_count = 0
        
        print(f"\n🔄 Processing {len(task_groups)} logical tasks...")
        
        for task_key, file_infos in task_groups.items():
            grid_size, n_agents, pattern, trial_id, merge_interval = task_key
            
            # Calculate the new consistent seed
            new_seed = self.generate_new_consistent_seed(grid_size, n_agents, pattern, trial_id)
            
            # Create new filename
            interval_str = 'inf' if merge_interval == float('inf') else str(merge_interval)
            new_filename = (f"grid{grid_size[0]}x{grid_size[1]}_"
                          f"agents{n_agents}_{pattern}_"
                          f"trial{trial_id}_interval{interval_str}_"
                          f"seed{new_seed}.pkl")
            
            if len(file_infos) > 1:
                # Multiple files for same task - keep the most recent
                file_infos.sort(key=lambda x: x['mtime'], reverse=True)
                keep_file = file_infos[0]
                
                # Remove duplicates
                for file_info in file_infos[1:]:
                    try:
                        file_info['file_path'].unlink()
                        removed_duplicates += 1
                        print(f"🗑️  Removed duplicate: {file_info['file_path'].name}")
                    except Exception as e:
                        print(f"❌ Error removing {file_info['file_path'].name}: {e}")
                        error_count += 1
            else:
                keep_file = file_infos[0]
            
            # Rename the kept file to use new seed
            source_path = keep_file['file_path']
            target_path = source_path.parent / new_filename
            
            try:
                if source_path.name != new_filename:
                    # Check if target already exists
                    if target_path.exists():
                        # Target exists - keep the newer one
                        source_mtime = source_path.stat().st_mtime
                        target_mtime = target_path.stat().st_mtime
                        
                        if source_mtime > target_mtime:
                            # Source is newer - replace target
                            target_path.unlink()
                            source_path.rename(target_path)
                            print(f"🔄 Replaced: {source_path.name} → {new_filename}")
                            renamed_count += 1
                        else:
                            # Target is newer - remove source
                            source_path.unlink()
                            print(f"🗑️  Removed older: {source_path.name} (target exists and is newer)")
                            removed_duplicates += 1
                    else:
                        # Simple rename
                        source_path.rename(target_path)
                        print(f"✏️  Renamed: {source_path.name} → {new_filename}")
                        renamed_count += 1
                else:
                    print(f"✅ Already correct: {source_path.name}")
                    
            except Exception as e:
                print(f"❌ Error processing {source_path.name}: {e}")
                error_count += 1
        
        # Handle unparseable files
        if unparseable_files:
            print(f"\n⚠️  Found {len(unparseable_files)} unparseable files:")
            for file_path in unparseable_files:
                print(f"   - {file_path.name}")
            print("   These files were left unchanged.")
        
        # Summary
        print(f"\n🎉 AUTOMATIC RENAMING COMPLETE!")
        print("="*60)
        print(f"✅ Files renamed: {renamed_count}")
        print(f"🗑️  Duplicates removed: {removed_duplicates}")
        print(f"❌ Errors: {error_count}")
        print(f"📁 Backup created: {self.backup_dir}")
        print("="*60)
        
        if renamed_count > 0 or removed_duplicates > 0:
            print("✨ Your checkpoint files now use consistent seed format!")
            print("🚀 Run your experiment again - it will recognize completed tasks!")
        else:
            print("ℹ️  All files were already in the correct format.")
        
        return {
            'renamed': renamed_count,
            'removed_duplicates': removed_duplicates,
            'errors': error_count,
            'backup_dir': str(self.backup_dir)
        }


def main():
    """Main function - fully automatic"""
    print("AUTOMATIC CHECKPOINT RENAMER")
    print("="*80)
    print("Automatically renaming checkpoint files to use consistent seeds...")
    print("This preserves your completed work and eliminates duplicates.")
    print("="*80)
    
    renamer = AutomaticCheckpointRenamer()
    
    if not renamer.checkpoint_dir.exists():
        print(f"❌ Error: Checkpoint directory {renamer.checkpoint_dir} not found")
        return
    
    # Run automatic renaming
    results = renamer.automatic_rename_all()
    
    print(f"\n📋 FINAL SUMMARY:")
    print(f"   Renamed: {results['renamed']} files")
    print(f"   Removed duplicates: {results['removed_duplicates']} files")
    print(f"   Errors: {results['errors']} files")
    print(f"   Backup: {results['backup_dir']}")
    
    if results['errors'] == 0:
        print(f"\n🎯 SUCCESS! Ready to resume your experiment.")
    else:
        print(f"\n⚠️  Completed with {results['errors']} errors. Check the output above.")


if __name__ == "__main__":
    main()