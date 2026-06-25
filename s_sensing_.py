import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
import matplotlib.pyplot as plt
import random
import gc

# ==============================================================================
# 1. CONFIGURATION (MODIFY HERE FOR FULL SIMULATION)
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- QUICK TEST SETTINGS (Runs in ~10-15 mins, uses minimal RAM) ---
# SNR_RANGE = np.arange(-20, -14, 4)  # Test with just 2 SNR points
# N_MC = 2000                          # Test with 500 trials
# SENSING_SAMPLES = 256               # Test with fewer samples
# SEQUENCE_LENGTH = 50                # Test with shorter sequences
# BATCH_SIZE = 64                     # Batch size for training
# EPOCHS = 20                         # Test with 5 epochs
# DRL_EPISODES = 200                  # Test with 50 DRL episodes
# N_EVAL_EPISODES = 20                # Test DQN evaluation episodes

# --- FULL SIMULATION SETTINGS (Uncomment these for the final 3-6 hr run) ---
SNR_RANGE = np.arange(-20, 12, 2)   # -20 to +10 dB, step 2 (16 points)
N_MC = 10000                        # 10,000 Monte Carlo trials (Table 3.1)
SENSING_SAMPLES = 2048              # 2048 samples (Table 3.1)
SEQUENCE_LENGTH = 100               # Sequence length for LSTM
BATCH_SIZE = 256                    # Larger batch size for faster GPU training
EPOCHS = 100                        # 100 epochs for supervised learning (Table 3.1)
DRL_EPISODES = 2000                 # 2000 episodes for DRL (Table 3.1)
N_EVAL_EPISODES = 1000              # 1000 episodes for DQN evaluation

# --- Fixed Parameters (Table 3.1 & 3.4) ---
PU_PRIOR = 0.30                       # P(H1)
TRANS_PROB_IDLE_TO_BUSY = 0.1
TRANS_PROB_BUSY_TO_IDLE = 0.2
TARGET_PD = 0.90                      # Regulatory target
TARGET_PF = 0.01                      # Regulatory target
REPLAY_BUFFER_SIZE = 50000            # DRL Replay buffer
GAMMA = 0.99                          # DRL Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# ==============================================================================
# 2. SIGNAL & CHANNEL GENERATION
# ==============================================================================
def generate_received_signal(snr_db, n_samples, pu_state):
    """Generates received signal with Rayleigh fading and AWGN."""
    h = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    s = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    
    snr_linear = 10 ** (snr_db / 10)
    
    if pu_state == 1:  # H1: PU Present
        signal_power = np.mean(np.abs(h * s)**2)
        noise_power = signal_power / snr_linear
        y = h * s + np.sqrt(noise_power) * noise
    else:  # H0: PU Absent
        y = np.sqrt(1/snr_linear) * noise
        
    return y.real.astype(np.float32), y.imag.astype(np.float32)

# ==============================================================================
# 3. MEMORY-EFFICIENT DATASET GENERATION
# ==============================================================================
def generate_lstm_dataset(snr_db, n_mc, sensing_samples, seq_length):
    """Generates dataset using pre-allocated arrays to prevent RAM crashes."""
    step = max(1, seq_length // 10)
    n_seqs_per_trial = (sensing_samples - seq_length) // step + 1
    total_seqs = n_mc * n_seqs_per_trial
    
    # Pre-allocate float32 arrays (cuts RAM usage by 50% compared to float64)
    X_np = np.empty((total_seqs, seq_length, 2), dtype=np.float32)
    Y_np = np.empty((total_seqs, 1), dtype=np.float32)
    
    idx = 0
    for _ in range(n_mc):
        pu_state = 1.0 if np.random.rand() < PU_PRIOR else 0.0
        i_sig, q_sig = generate_received_signal(snr_db, sensing_samples, int(pu_state))
        
        for i in range(0, sensing_samples - seq_length + 1, step):
            X_np[idx, :, 0] = i_sig[i:i+seq_length]
            X_np[idx, :, 1] = q_sig[i:i+seq_length]
            Y_np[idx, 0] = pu_state
            idx += 1
            
    return torch.from_numpy(X_np), torch.from_numpy(Y_np)

# ==============================================================================
# 4. LSTM SPECTRUM SENSING MODEL
# ==============================================================================
class LSTMSensor(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1):
        super(LSTMSensor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return self.sigmoid(out)

def find_optimal_threshold(predictions, labels, target_pf=TARGET_PF):
    """Finds threshold using vectorized sorting to guarantee Pf <= target_pf."""
    h0_mask = (labels == 0).squeeze()
    h1_mask = (labels == 1).squeeze()
    
    h0_predictions = predictions.squeeze()[h0_mask]
    h1_predictions = predictions.squeeze()[h1_mask]
    
    sorted_h0, _ = torch.sort(h0_predictions)
    idx = int(len(sorted_h0) * (1.0 - target_pf))
    if idx >= len(sorted_h0): idx = len(sorted_h0) - 1
        
    best_threshold = sorted_h0[idx].item()
    return best_threshold

def train_lstm_sensor(snr_db):
    model = LSTMSensor(hidden_size=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print(f"    Generating dataset for SNR {snr_db} dB...")
    X_tensor, Y_tensor = generate_lstm_dataset(snr_db, N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"    Early stopping at epoch {epoch+1}")
                break
                
    del X_tensor, Y_tensor, dataset, dataloader
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return model

def evaluate_lstm_sensor(model, snr_db):
    model.eval()
    print(f"    Generating evaluation dataset for SNR {snr_db} dB...")
    X_tensor, Y_tensor = generate_lstm_dataset(snr_db, N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            all_preds.append(model(batch_X.to(DEVICE)).cpu())
            all_labels.append(batch_Y)
            
    predictions = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    del X_tensor, Y_tensor, dataset, dataloader, all_preds, all_labels
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    
    optimal_threshold = find_optimal_threshold(predictions, labels, TARGET_PF)
    decisions = (predictions > optimal_threshold).float()
    
    h1_mask, h0_mask = (labels == 1), (labels == 0)
    Pd = (decisions[h1_mask] == 1).sum().item() / h1_mask.sum().item() if h1_mask.sum() > 0 else 0
    Pf = (decisions[h0_mask] == 1).sum().item() / h0_mask.sum().item() if h0_mask.sum() > 0 else 0
    
    n_h1, n_h0 = h1_mask.sum().item(), h0_mask.sum().item()
    ci_pd = 1.96 * np.sqrt(Pd * (1-Pd) / n_h1) if n_h1 > 0 else 0
    ci_pf = 1.96 * np.sqrt(Pf * (1-Pf) / n_h0) if n_h0 > 0 else 0
    
    del predictions, labels, decisions
    gc.collect()
    return Pd, Pf, ci_pd, ci_pf, optimal_threshold

# ==============================================================================
# 5. DQN DYNAMIC SPECTRUM ACCESS
# ==============================================================================
class CRNEnvironment:
    def __init__(self, n_channels=4):
        self.n_channels = n_channels
        self.pu_states = np.zeros(n_channels, dtype=int)
        
    def reset(self):
        self.pu_states = np.random.choice([0, 1], size=self.n_channels, p=[1-PU_PRIOR, PU_PRIOR])
        return self.pu_states.copy()
        
    def step(self, action):
        reward, interference = 0, 0
        if action < self.n_channels:
            if self.pu_states[action] == 0: reward = 10
            else: reward = -50; interference = 1
                
        for i in range(self.n_channels):
            if self.pu_states[i] == 0:
                self.pu_states[i] = 1 if np.random.rand() < TRANS_PROB_IDLE_TO_BUSY else 0
            else:
                self.pu_states[i] = 0 if np.random.rand() < TRANS_PROB_BUSY_TO_IDLE else 1
        return self.pu_states.copy(), reward, interference

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn_agent():
    n_channels, action_size, state_size = 4, 5, 4
    agent = DQNAgent(state_size, action_size).to(DEVICE)
    target_agent = DQNAgent(state_size, action_size).to(DEVICE)
    target_agent.load_state_dict(agent.state_dict())
    
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    memory = deque(maxlen=REPLAY_BUFFER_SIZE)
    epsilon = EPSILON_START
    env = CRNEnvironment(n_channels)
    rewards_history = []
    
    for episode in range(DRL_EPISODES):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        total_reward = 0
        
        for t in range(50):
            if np.random.rand() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad(): action = agent(state_tensor).argmax().item()
                    
            next_state, reward, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
            memory.append((state_tensor, action, torch.FloatTensor([reward]).to(DEVICE), next_state_tensor))
            state_tensor = next_state_tensor
            total_reward += reward
            
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_s, b_a, b_r, b_ns = zip(*batch)
                b_s, b_ns = torch.cat(b_s).to(DEVICE), torch.cat(b_ns).to(DEVICE)
                b_r, b_a = torch.cat(b_r).to(DEVICE), torch.LongTensor(b_a).to(DEVICE)
                
                current_q = agent(b_s).gather(1, b_a.unsqueeze(1))
                next_q = target_agent(b_ns).max(1)[0].detach()
                loss = F.mse_loss(current_q.squeeze(), b_r + (GAMMA * next_q))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if epsilon > EPSILON_END: epsilon *= EPSILON_DECAY
        if episode % 10 == 0: target_agent.load_state_dict(agent.state_dict())
        rewards_history.append(total_reward)
        
    return agent, env, rewards_history

def evaluate_dqn_throughput(agent, snr_db):
    env = CRNEnvironment(4)
    capacity = 8 * np.log2(1 + 10**(snr_db/10)) # 8MHz TVWS channel
    total_throughput = 0
    
    for _ in range(N_EVAL_EPISODES):
        state = env.reset()
        ep_throughput = 0
        for _ in range(50):
            with torch.no_grad():
                action = agent(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax().item()
            next_state, _, interference = env.step(action)
            if action < 4 and interference == 0: ep_throughput += capacity
            state = next_state
        total_throughput += ep_throughput
    return total_throughput / N_EVAL_EPISODES

# ==============================================================================
# 6. MAIN EXECUTION & PLOTTING
# ==============================================================================
def run_simulation():
    print(f"\nStarting Simulation | MC Trials: {N_MC} | Targets: Pd≥{TARGET_PD}, Pf≤{TARGET_PF}")
    print("="*70)
    
    # --- LSTM Evaluation ---
    Pd_list, Pf_list, Pd_ci, Pf_ci, thresholds = [], [], [], [], []
    
    for snr in SNR_RANGE:
        print(f"\n[Phase II-A] SNR: {snr} dB")
        model = train_lstm_sensor(snr)
        Pd, Pf, ci_pd, ci_pf, thresh = evaluate_lstm_sensor(model, snr)
        
        # Append to lists
        Pd_list.append(Pd)
        Pf_list.append(Pf)
        Pd_ci.append(ci_pd)
        Pf_ci.append(ci_pf)
        thresholds.append(thresh)
        
        status = "✓ PASS" if (Pd >= TARGET_PD and Pf <= TARGET_PF) else "✗ FAIL"
        print(f"  -> Pd: {Pd:.4f} (±{ci_pd:.4f}), Pf: {Pf:.4f} (±{ci_pf:.4f}) | Thresh: {thresh:.3f} | {status}")

    # --- DQN Evaluation ---
    print(f"\n[Phase II-B] Training DQN for {DRL_EPISODES} episodes...")
    agent, env, rewards = train_dqn_agent()
    
    print("\nEvaluating DQN Throughput...")
    throughputs = []
    for snr in SNR_RANGE:
        tp = evaluate_dqn_throughput(agent, snr)
        throughputs.append(tp)
        print(f"  -> SNR {snr:3d} dB: {tp:.2f} Mbps")

    # ==========================================
    # PLOTTING SECTION (MUST BE INSIDE THE FUNCTION)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC Curve
    ax1 = axes[0, 0]
    ax1.plot(Pf_list, Pd_list, 'b-o', linewidth=2, markersize=8, label='LSTM Sensor')
    ax1.axhline(y=TARGET_PD, color='r', linestyle='--', alpha=0.7, label=f'Target Pd ≥ {TARGET_PD}')
    ax1.axvline(x=TARGET_PF, color='g', linestyle='--', alpha=0.7, label=f'Target Pf ≤ {TARGET_PF}')
    ax1.set_xlabel('False Alarm Probability ($P_f$)', fontsize=11)
    ax1.set_ylabel('Detection Probability ($P_d$)', fontsize=11)
    ax1.set_title('LSTM Spectrum Sensing ROC', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 0.05])
    ax1.set_ylim([0.85, 1.0])
    
    # 2. DQN Throughput
    ax2 = axes[0, 1]
    ax2.plot(SNR_RANGE, throughputs, 'r-s', linewidth=2, markersize=8, label='DQN Access')
    ax2.set_xlabel('SNR (dB)', fontsize=11)
    ax2.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax2.set_title('DQN Dynamic Spectrum Access', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Pd & Pf vs SNR (DEBUG VERSION)
    ax3 = axes[1, 0]

    # Debug: Print what's actually in the lists
    print(f"\nDEBUG - Pd_list: {Pd_list}")
    print(f"DEBUG - Pf_list: {Pf_list}")
    print(f"DEBUG - SNR_RANGE: {SNR_RANGE}")

    # Plot with explicit markers
    ax3.plot(SNR_RANGE, Pd_list, 'b-o', linewidth=2, markersize=10, label='$P_d$ (Detection)', zorder=5)
    ax3.plot(SNR_RANGE, Pf_list, 'r-s', linewidth=2, markersize=10, label='$P_f$ (False Alarm)', zorder=5)

    # Add target lines
    ax3.axhline(y=TARGET_PD, color='blue', linestyle=':', alpha=0.5, linewidth=2, label=f'Target Pd={TARGET_PD}')
    ax3.axhline(y=TARGET_PF, color='red', linestyle=':', alpha=0.5, linewidth=2, label=f'Target Pf={TARGET_PF}')

    ax3.set_xlabel('SNR (dB)', fontsize=11)
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_title('Detection & False Alarm vs SNR', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='center right', fontsize=10)
    ax3.set_ylim([0, 1.0])
    
    # 4. DQN Training
    ax4 = axes[1, 1]
    window = max(1, len(rewards)//20)
    avg_rewards = [np.mean(rewards[i:i+window]) for i in range(0, len(rewards), window)]
    ax4.plot(avg_rewards, 'g-', linewidth=2)
    ax4.set_xlabel('Episode (averaged)', fontsize=11)
    ax4.set_ylabel('Reward', fontsize=11)
    ax4.set_title('DQN Training Convergence', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_crn_results_fixed.png', dpi=300)
    print("\n" + "="*70)
    print("Simulation complete! Results saved to 'ml_crn_results_fixed.png'")
    plt.show()

# Call the function to run everything
if __name__ == "__main__":
    run_simulation()
