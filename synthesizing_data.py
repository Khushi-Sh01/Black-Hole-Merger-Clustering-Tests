import numpy as np         
import pandas as pd        
import os                   
np.random.seed(42)
N_PER_CHANNEL = 400   # We are making 400 mergers per population channel
                      # Total = 3 channels × 400 = 1200 mergers
# CHANNEL 1: Field Binaries
channel1 = pd.DataFrame({
    'channel'    : ['field_binary'] * N_PER_CHANNEL,    # Primary mass: the heavier black hole (solar masses)
    'm1'         : np.random.uniform(5, 40, N_PER_CHANNEL),    # Secondary mass: the lighter black hole
    'm2'         : np.random.uniform(5, 30, N_PER_CHANNEL),    # Spin: how fast the BH is spinning (0 = no spin, 1 = max spin)
    'spin1'      : np.random.uniform(0.0, 0.5, N_PER_CHANNEL),
    'spin2'      : np.random.uniform(0.0, 0.5, N_PER_CHANNEL),    # Metallicity: fraction of heavy elements (solar = ~0.02)
    'metallicity': np.random.uniform(0.005, 0.02, N_PER_CHANNEL),  # Redshift: how far back in time (0 = today, 20 = very early universe)
    'redshift'   : np.random.uniform(0, 3, N_PER_CHANNEL),
})

# CHANNEL 2: Globular Cluster Mergers
channel2 = pd.DataFrame({
    'channel'    : ['globular_cluster'] * N_PER_CHANNEL,
    'm1'         : np.random.uniform(20, 80, N_PER_CHANNEL),
    'm2'         : np.random.uniform(15, 60, N_PER_CHANNEL),
    'spin1'      : np.random.uniform(0.0, 1.0, N_PER_CHANNEL),
    'spin2'      : np.random.uniform(0.0, 1.0, N_PER_CHANNEL),
    'metallicity': np.random.uniform(0.0001, 0.003, N_PER_CHANNEL),
    'redshift'   : np.random.uniform(0, 6, N_PER_CHANNEL),
})

# CHANNEL 3: Population III (PopIII) Star Remnants
channel3 = pd.DataFrame({
    'channel'    : ['pop3_remnant'] * N_PER_CHANNEL,
    'm1'         : np.random.uniform(50, 150, N_PER_CHANNEL),
    'm2'         : np.random.uniform(40, 120, N_PER_CHANNEL),
    'spin1'      : np.random.uniform(0.5, 1.0, N_PER_CHANNEL),  # Fast spinning
    'spin2'      : np.random.uniform(0.5, 1.0, N_PER_CHANNEL),
    'metallicity': np.random.uniform(0.0, 0.0001, N_PER_CHANNEL),  # Almost zero
    'redshift'   : np.random.uniform(6, 20, N_PER_CHANNEL),
})
# COMBINE ALL THREE CHANNELS: Stack all three DataFrames into one big table and shuffle
data = pd.concat([channel1, channel2, channel3], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# Adding derived features: # Mass ratio: q = m2/m1 (always ≤ 1)
data['mass_ratio'] = data['m2'] / data['m1']
data['total_mass'] = data['m1'] + data['m2']
os.makedirs('data', exist_ok=True) 
data.to_csv('data/bh_mergers.csv', index=False)
print(f"   Total mergers: {len(data)}")print(f"   Columns: {list(data.columns)}")print(f"\nFirst 5 rows:")
print(data.head())print(f"\nChannel breakdown:")print(data['channel'].value_counts())
