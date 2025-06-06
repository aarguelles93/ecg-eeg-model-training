import os
import mne
import numpy as np
import pandas as pd

BDF_DIR = "data/bdf"
OUTPUT_CSV = "data/eeg_dataset_32.csv"
SEGMENT_LENGTH = 188
STEP = 188  # No overlap - each segment is independent
DOWNSAMPLED_FREQ = 128
EEG_LABEL = 1
N_TRIALS_DEAP = 40
TRIAL_DURATION = 63.0
TARGET_CHANNELS = 32  # We want exactly 32 EEG channels

# Standard DEAP 32 EEG channel names (in order)
DEAP_32_CHANNELS = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3',
    'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz',
    'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
    'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
]

# Create output directory if needed
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

print(f"\n🚀 Starting robust EEG extraction from DEAP BDF files in: {BDF_DIR}")
print(f"🎯 Target: {TARGET_CHANNELS} EEG channels, STEP={STEP} (no overlap)\n")

# Track file processing stats
csv_initialized = False
total_segments = 0
processed_files = 0
skipped_files = 0
channel_counts = {}

def get_eeg_channels(raw):
    """Extract exactly 32 EEG channels, handling inconsistencies"""
    all_channels = raw.ch_names
    
    # Method 1: Try to pick standard EEG channels
    try:
        raw_copy = raw.copy()
        raw_copy.pick_types(eeg=True, exclude=[])
        eeg_channels = raw_copy.ch_names
        
        if len(eeg_channels) >= TARGET_CHANNELS:
            # Take first 32 EEG channels
            selected_channels = eeg_channels[:TARGET_CHANNELS]
            return selected_channels, f"Selected first {TARGET_CHANNELS} from {len(eeg_channels)} EEG channels"
        else:
            return None, f"Only {len(eeg_channels)} EEG channels found, need {TARGET_CHANNELS}"
            
    except Exception as e:
        pass
    
    # Method 2: Try to match known DEAP channel names
    try:
        available_deap_channels = [ch for ch in DEAP_32_CHANNELS if ch in all_channels]
        if len(available_deap_channels) >= TARGET_CHANNELS:
            return available_deap_channels[:TARGET_CHANNELS], f"Matched {len(available_deap_channels)} DEAP channels"
    except:
        pass
    
    # Method 3: Pick channels that look like EEG (exclude obvious non-EEG)
    exclude_patterns = ['status', 'trigger', 'sti', 'ecg', 'eog', 'emg', 'gsr', 'resp', 'temp', 'misc']
    potential_eeg = []
    
    for ch in all_channels:
        if not any(pattern in ch.lower() for pattern in exclude_patterns):
            potential_eeg.append(ch)
    
    if len(potential_eeg) >= TARGET_CHANNELS:
        return potential_eeg[:TARGET_CHANNELS], f"Selected {TARGET_CHANNELS} from {len(potential_eeg)} potential EEG channels"
    
    # Method 4: Fallback - just take first 32 channels (excluding obvious non-EEG)
    if len(all_channels) >= TARGET_CHANNELS:
        filtered_channels = [ch for ch in all_channels if not any(pattern in ch.lower() for pattern in exclude_patterns)]
        if len(filtered_channels) >= TARGET_CHANNELS:
            return filtered_channels[:TARGET_CHANNELS], f"Fallback: first {TARGET_CHANNELS} channels after filtering"
    
    return None, f"Cannot find {TARGET_CHANNELS} suitable channels from {len(all_channels)} total channels"

# Process each file individually
for filename in sorted(os.listdir(BDF_DIR)):
    if filename.endswith(".bdf"):
        filepath = os.path.join(BDF_DIR, filename)
        print(f"📄 Processing {filename}...")
        
        try:
            # Load BDF file
            raw = mne.io.read_raw_bdf(filepath, preload=True, verbose='ERROR')
            
            # Debug info
            original_channels = len(raw.ch_names)
            original_freq = raw.info['sfreq']
            duration = raw.times[-1]
            
            print(f"   📋 Original: {original_channels} channels, {original_freq} Hz, {duration:.1f}s")
            
            # Track channel counts for analysis
            channel_counts[filename] = original_channels
            
            # Get exactly 32 EEG channels
            selected_channels, selection_msg = get_eeg_channels(raw)
            
            if selected_channels is None:
                print(f"   ❌ {selection_msg}")
                skipped_files += 1
                continue
            
            print(f"   ✅ {selection_msg}")
            
            # Pick selected channels and resample
            raw.pick_channels(selected_channels)
            raw.resample(DOWNSAMPLED_FREQ)
            
            final_channels = len(raw.ch_names)
            print(f"   🧠 Final: {final_channels} channels, {DOWNSAMPLED_FREQ} Hz")
            
            # Verify we have exactly what we expect
            if final_channels != TARGET_CHANNELS:
                print(f"   ❌ Expected {TARGET_CHANNELS} channels, got {final_channels}")
                skipped_files += 1
                continue
            
            # Create artificial events for DEAP structure
            samples_per_trial = int(TRIAL_DURATION * DOWNSAMPLED_FREQ)
            total_samples = len(raw.times)
            max_possible_trials = min(N_TRIALS_DEAP, total_samples // samples_per_trial)
            
            if max_possible_trials == 0:
                print(f"   ❌ File too short for trials ({total_samples} samples, need {samples_per_trial})")
                skipped_files += 1
                continue
            
            print(f"   🎯 Extracting {max_possible_trials} trials ({samples_per_trial} samples each)")
            
            # Generator for memory-efficient segment creation
            def generate_segments():
                segments_generated = 0
                for trial_idx in range(max_possible_trials):
                    start_sample = trial_idx * samples_per_trial
                    end_sample = start_sample + samples_per_trial
                    
                    try:
                        # Extract trial data
                        trial_data = raw.get_data(start=start_sample, stop=end_sample)
                        
                        # Verify trial shape
                        if trial_data.shape != (TARGET_CHANNELS, samples_per_trial):
                            print(f"   ⚠️ Trial {trial_idx} shape mismatch: {trial_data.shape}")
                            continue
                        
                        # Create non-overlapping segments
                        for seg_start in range(0, trial_data.shape[1] - SEGMENT_LENGTH + 1, STEP):
                            segment = trial_data[:, seg_start:seg_start + SEGMENT_LENGTH]
                            
                            # Verify segment shape
                            if segment.shape != (TARGET_CHANNELS, SEGMENT_LENGTH):
                                continue
                                
                            segment_flat = segment.flatten()
                            segments_generated += 1
                            yield np.append(segment_flat, EEG_LABEL)
                            
                    except Exception as e:
                        print(f"   ⚠️ Error in trial {trial_idx}: {e}")
                        continue
                
                return segments_generated
            
            # Process segments in chunks
            segments_this_file = 0
            chunk_size = 1000
            chunk_data = []
            
            for segment in generate_segments():
                chunk_data.append(segment)
                segments_this_file += 1
                
                # Write chunk when full
                if len(chunk_data) >= chunk_size:
                    if not csv_initialized:
                        # Initialize CSV with proper headers
                        columns = [f"ch{i+1}_t{j+1}" for i in range(TARGET_CHANNELS) for j in range(SEGMENT_LENGTH)] + ["label"]
                        df_chunk = pd.DataFrame(chunk_data, columns=columns)
                        df_chunk.to_csv(OUTPUT_CSV, index=False, mode='w')
                        csv_initialized = True
                        print(f"   📝 CSV initialized with {len(columns)} columns")
                    else:
                        df_chunk = pd.DataFrame(chunk_data)
                        df_chunk.to_csv(OUTPUT_CSV, index=False, mode='a', header=False)
                    
                    chunk_data = []
            
            # Write remaining data
            if chunk_data:
                if not csv_initialized:
                    columns = [f"ch{i+1}_t{j+1}" for i in range(TARGET_CHANNELS) for j in range(SEGMENT_LENGTH)] + ["label"]
                    df_chunk = pd.DataFrame(chunk_data, columns=columns)
                    df_chunk.to_csv(OUTPUT_CSV, index=False, mode='w')
                    csv_initialized = True
                    print(f"   📝 CSV initialized with {len(columns)} columns")
                else:
                    df_chunk = pd.DataFrame(chunk_data)
                    df_chunk.to_csv(OUTPUT_CSV, index=False, mode='a', header=False)
            
            total_segments += segments_this_file
            processed_files += 1
            print(f"   ✅ {segments_this_file} segments written to CSV")
            
            # Memory cleanup
            del raw
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            skipped_files += 1
            continue
        
        print()

# Final summary
print(f"🎉 Processing complete!")
print(f"📊 Files processed: {processed_files}")
print(f"📊 Files skipped: {skipped_files}")
print(f"📊 Total segments: {total_segments}")
print(f"💾 Dataset saved to: {OUTPUT_CSV}")

# Channel count analysis
if channel_counts:
    unique_counts = set(channel_counts.values())
    print(f"\n📋 Channel count analysis:")
    for count in sorted(unique_counts):
        files_with_count = [f for f, c in channel_counts.items() if c == count]
        print(f"   {count} channels: {len(files_with_count)} files")

# Final verification
if total_segments > 0 and os.path.exists(OUTPUT_CSV):
    try:
        df_check = pd.read_csv(OUTPUT_CSV, nrows=1)
        expected_features = TARGET_CHANNELS * SEGMENT_LENGTH
        actual_features = len(df_check.columns) - 1  # -1 for label
        
        print(f"\n✅ Final verification:")
        print(f"   Expected features per segment: {expected_features}")
        print(f"   Actual features per segment: {actual_features}")
        print(f"   Match: {'✅' if expected_features == actual_features else '❌'}")
        
        if expected_features == actual_features:
            print(f"   Segment shape: {TARGET_CHANNELS} channels × {SEGMENT_LENGTH} timepoints")
    except Exception as e:
        print(f"⚠️ Could not verify final CSV: {e}")

print("\n✨ Robust processing completed!")