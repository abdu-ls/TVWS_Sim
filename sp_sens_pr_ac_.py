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
# 1. CONFIGURATION
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- QUICK TEST SETTINGS (Runs in ~10-15 mins) ---
SNR_RANGE = np.array([-5, 0, 5, 10])  
N_MC = 2000                         
SENSING_SAMPLES = 1024              # More data points
SEQUENCE_LENGTH = 256               # CRITICAL: More samples = lower noise variance
BATCH_SIZE = 64                     
EPOCHS = 20                         
DRL_EPISODES = 200                  
N_EVAL_EPISODES = 20                
PREDICTION_WINDOW = 10 

# --- FULL SIMULATION SETTINGS (Uncomment for final 3-6 hr run) ---
# SNR_RANGE = np.arange(-20, 12, 2)   
# N_MC = 5000                        
# SENSING_SAMPLES = 1024              
# SEQUENCE_LENGTH = 100               
# BATCH_SIZE = 128                    
# EPOCHS = 50                        
# DRL_EPISODES = 500                 
# N_EVAL_EPISODES = 500              
# PREDICTION_WINDOW = 10              

# --- Fixed Parameters ---
PU_PRIOR = 0.30                       
TRANS_PROB_IDLE_TO_BUSY = 0.1
TRANS_PROB_BUSY_TO_IDLE = 0.2
TARGET_PD = 0.90                      
TARGET_PF = 0.01                      
REPLAY_BUFFER_SIZE = 50000            
GAMMA = 0.99                          
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# ==============================================================================
# 2. SIGNAL & CHANNEL GENERATION
# ==============================================================================
def generate_received_signal(snr_db, n_samples, pu_state):
    """Generates received signal with Rayleigh fading and AWGN (Standard Model)."""
    # Fixed noise variance = 1 (Complex Gaussian)
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    snr_linear = 10 ** (snr_db / 10)
    
    if pu_state == 1:  # H1: PU Present
        # Channel and Signal (both unit variance)
        h = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
        s = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
        # Standard model: y = sqrt(SNR)*h*s + noise
        y = np.sqrt(snr_linear) * h * s + noise
    else:  # H0: PU Absent
        y = noise
        
    return y.real.astype(np.float32), y.imag.astype(np.float32)

# ==============================================================================
# 3. DATASET GENERATION
# ==============================================================================
def generate_sensing_dataset(snr_db, n_mc, sensing_samples, seq_length):
    """Generates I/Q dataset for Spectrum Sensing (ED and DNN)."""
    step = max(1, seq_length // 10)
    n_seqs_per_trial = (sensing_samples - seq_length) // step + 1
    total_seqs = n_mc * n_seqs_per_trial
    
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

def generate_prediction_dataset(pu_states_seq, past_steps):
    """Generates sequence dataset for LSTM Occupancy Prediction."""
    X, Y = [], []
    for i in range(len(pu_states_seq) - past_steps):
        X.append(pu_states_seq[i : i+past_steps])
        Y.append(pu_states_seq[i+past_steps])
    return np.array(X).reshape(-1, past_steps, 1).astype(np.float32), \
           np.array(Y).reshape(-1, 1).astype(np.float32)

# ==============================================================================
# 4. PHASE 1: SPECTRUM SENSING (ED vs DNN)
# ==============================================================================
def energy_detection_sensing(X_np, Y_np, target_pf):
    """Baseline: Energy Detection."""
    # Calculate energy: mean(I^2 + Q^2)
    energy = np.mean(X_np**2, axis=(1, 2)) 
    h0_mask = Y_np.squeeze() == 0
    h1_mask = Y_np.squeeze() == 1
    
    # Find threshold for target Pf
    h0_energy = energy[h0_mask]
    threshold = np.percentile(h0_energy, (1 - target_pf) * 100)
    
    decisions = (energy > threshold).astype(int)
    
    Pd = np.sum(decisions[h1_mask] == 1) / np.sum(h1_mask) if np.sum(h1_mask) > 0 else 0
    Pf = np.sum(decisions[h0_mask] == 1) / np.sum(h0_mask) if np.sum(h0_mask) > 0 else 0
    
    return Pd, Pf, threshold

class DNNSensor(nn.Module):
    """Proposed: Deep Neural Network for Sensing."""
    def __init__(self, input_size):
        super(DNNSensor, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

def train_dnn_sensor(snr_db):
    model = DNNSensor(SEQUENCE_LENGTH).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    X_tensor, Y_tensor = generate_sensing_dataset(snr_db, N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            optimizer.step()
            
    del X_tensor, Y_tensor, dataset, dataloader
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return model

def evaluate_dnn_sensor(model, snr_db):
    model.eval()
    X_tensor, Y_tensor = generate_sensing_dataset(snr_db, N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            all_preds.append(model(batch_X.to(DEVICE)).cpu())
            all_labels.append(batch_Y)
            
    predictions = torch.cat(all_preds).squeeze()
    labels = torch.cat(all_labels).squeeze()
    
    h0_mask = labels == 0
    h1_mask = labels == 1
    
    # Find threshold for target Pf
    h0_preds = predictions[h0_mask]
    sorted_h0, _ = torch.sort(h0_preds)
    idx = int(len(sorted_h0) * (1.0 - TARGET_PF))
    threshold = sorted_h0[min(idx, len(sorted_h0)-1)].item()
    
    decisions = (predictions > threshold).float()
    Pd = (decisions[h1_mask] == 1).sum().item() / h1_mask.sum().item() if h1_mask.sum() > 0 else 0
    Pf = (decisions[h0_mask] == 1).sum().item() / h0_mask.sum().item() if h0_mask.sum() > 0 else 0
    
    del X_tensor, Y_tensor, dataset, dataloader, predictions, labels
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    
    return Pd, Pf, threshold

# ==============================================================================
# 5. PHASE 2: OCCUPANCY PREDICTION (LSTM)
# ==============================================================================
class LSTMPredictor(nn.Module):
    """Proposed: LSTM for predicting future PU state."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

def train_lstm_predictor():
    """Trains LSTM on a long sequence of simulated PU states."""
    # Generate a long sequence of PU states based on Markov chain
    seq_len = 10000
    states = np.zeros(seq_len, dtype=int)
    states[0] = 1 if np.random.rand() < PU_PRIOR else 0
    for i in range(1, seq_len):
        if states[i-1] == 0:
            states[i] = 1 if np.random.rand() < TRANS_PROB_IDLE_TO_BUSY else 0
        else:
            states[i] = 0 if np.random.rand() < TRANS_PROB_BUSY_TO_IDLE else 1
            
    X_np, Y_np = generate_prediction_dataset(states, PREDICTION_WINDOW)
    X_tensor = torch.from_numpy(X_np)
    Y_tensor = torch.from_numpy(Y_np)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMPredictor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            preds = model(batch_X.to(DEVICE)).cpu()
            preds = (preds > 0.5).float()
            correct += (preds == batch_Y).sum().item()
            total += batch_Y.size(0)
            
    accuracy = correct / total
    del X_tensor, Y_tensor, dataset, dataloader
    gc.collect()
    return accuracy

# ==============================================================================
# 6. PHASE 3: DYNAMIC SPECTRUM ACCESS (DQN vs DDQN)
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
            if self.pu_states[action] == 0: 
                reward = 10
            else: 
                reward = -50
                interference = 1
                
        # Update PU states (Markov)
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

    def learn(self, batch, target_net, is_ddqn=False):
        b_s, b_a, b_r, b_ns = batch
        b_s, b_ns = b_s.to(DEVICE), b_ns.to(DEVICE)
        b_r, b_a = b_r.to(DEVICE), b_a.to(DEVICE)

        current_q = self(b_s).gather(1, b_a.unsqueeze(1))

        if is_ddqn:
            # DDQN: Online net selects action, Target net evaluates it
            next_action = self(b_ns).argmax(1).unsqueeze(1)
            next_q = target_net(b_ns).gather(1, next_action).squeeze()
        else:
            # Standard DQN: Target net does both
            next_q = target_net(b_ns).max(1)[0]

        target_q = b_r + (GAMMA * next_q.detach())
        loss = F.mse_loss(current_q.squeeze(), target_q)
        return loss

def train_access_agent(is_ddqn=False):
    n_channels, action_size, state_size = 4, 4, 4
    
    # Create two completely separate networks
    agent = DQNAgent(state_size, action_size).to(DEVICE)
    target_agent = DQNAgent(state_size, action_size).to(DEVICE)
    
    # Copy initial weights BEFORE any submodule registration
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
                with torch.no_grad(): 
                    action = agent(state_tensor).argmax().item()
                    
            next_state, reward, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
            memory.append((state_tensor, action, torch.FloatTensor([reward]).to(DEVICE), next_state_tensor))
            state_tensor = next_state_tensor
            total_reward += reward
            
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_s, b_a, b_r, b_ns = zip(*batch)
                batch_data = (torch.cat(b_s), torch.LongTensor(b_a), torch.cat(b_r), torch.cat(b_ns))
                
                # Pass target_agent explicitly to avoid the state_dict error
                loss = agent.learn(batch_data, target_agent, is_ddqn=is_ddqn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if epsilon > EPSILON_END: epsilon *= EPSILON_DECAY
        
        # Update target network every 10 episodes
        if episode % 10 == 0: 
            target_agent.load_state_dict(agent.state_dict())
            
        rewards_history.append(total_reward)
        
    return agent, env, rewards_history

def evaluate_access_throughput(agent, snr_db):
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
            if action < 4 and interference == 0: 
                ep_throughput += capacity
            state = next_state
        total_throughput += ep_throughput
        
    return total_throughput / N_EVAL_EPISODES

# ==============================================================================
# 7. MAIN EXECUTION & PLOTTING
# ==============================================================================
def run_simulation():
    print(f"\nStarting ML-CRN Simulation | MC Trials: {N_MC}")
    print("="*70)
    
    # --- PHASE 1: SPECTRUM SENSING ---
    print("\n[Phase 1] Evaluating Spectrum Sensing (ED vs DNN)...")
    ed_pd, ed_pf = [], []
    dnn_pd, dnn_pf = [], []
    
    for snr in SNR_RANGE:
        print(f"  SNR: {snr} dB")
        # Generate data once for this SNR
        X_np, Y_np = generate_sensing_dataset(snr, N_MC, SENSING_SAMPLES, SEQUENCE_LENGTH)
        X_np = X_np.numpy()
        Y_np = Y_np.numpy()
        
        # ED
        pd, pf, _ = energy_detection_sensing(X_np, Y_np, TARGET_PF)
        ed_pd.append(pd); ed_pf.append(pf)
        print(f"    ED -> Pd: {pd:.4f}, Pf: {pf:.4f}")
        
        # DNN
        model = train_dnn_sensor(snr)
        pd, pf, _ = evaluate_dnn_sensor(model, snr)
        dnn_pd.append(pd); dnn_pf.append(pf)
        print(f"    DNN -> Pd: {pd:.4f}, Pf: {pf:.4f}")
        del X_np, Y_np
        gc.collect()

    # --- PHASE 2: OCCUPANCY PREDICTION ---
    print("\n[Phase 2] Training LSTM Occupancy Predictor...")
    pred_accuracy = train_lstm_predictor()
    print(f"  LSTM Prediction Accuracy: {pred_accuracy:.4f}")

    # --- PHASE 3: DYNAMIC SPECTRUM ACCESS ---
    print("\n[Phase 3] Training Access Agents (DQN vs DDQN)...")
    print("  Training Standard DQN...")
    dqn_agent, _, dqn_rewards = train_access_agent(is_ddqn=False)
    
    print("  Training Double DQN (DDQN)...")
    ddqn_agent, _, ddqn_rewards = train_access_agent(is_ddqn=True)
    
    print("\nEvaluating Throughput...")
    dqn_tp, ddqn_tp = [], []
    for snr in SNR_RANGE:
        tp1 = evaluate_access_throughput(dqn_agent, snr)
        tp2 = evaluate_access_throughput(ddqn_agent, snr)
        dqn_tp.append(tp1)
        ddqn_tp.append(tp2)
        print(f"  SNR {snr:3d} dB -> DQN: {tp1:.2f} Mbps | DDQN: {tp2:.2f} Mbps")

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Sensing Pd vs SNR
    ax1 = axes[0, 0]
    ax1.plot(SNR_RANGE, ed_pd, 'b-o', label='Energy Detection (ED)')
    ax1.plot(SNR_RANGE, dnn_pd, 'r-s', label='DNN Sensor')
    ax1.axhline(y=TARGET_PD, color='k', linestyle='--', label=f'Target Pd={TARGET_PD}')
    ax1.set_title('Detection Probability (Pd) vs SNR')
    ax1.set_xlabel('SNR (dB)'); ax1.set_ylabel('Pd')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Sensing Pf vs SNR
    ax2 = axes[0, 1]
    ax2.plot(SNR_RANGE, ed_pf, 'b-o', label='Energy Detection (ED)')
    ax2.plot(SNR_RANGE, dnn_pf, 'r-s', label='DNN Sensor')
    ax2.axhline(y=TARGET_PF, color='k', linestyle='--', label=f'Target Pf={TARGET_PF}')
    ax2.set_title('False Alarm Probability (Pf) vs SNR')
    ax2.set_xlabel('SNR (dB)'); ax2.set_ylabel('Pf')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # 3. LSTM Prediction
    ax3 = axes[0, 2]
    ax3.bar(['LSTM Predictor'], [pred_accuracy], color='g', width=0.4)
    ax3.set_ylim([0, 1.1])
    ax3.set_title('LSTM Occupancy Prediction Accuracy')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Access Throughput
    ax4 = axes[1, 0]
    ax4.plot(SNR_RANGE, dqn_tp, 'b-o', label='Standard DQN')
    ax4.plot(SNR_RANGE, ddqn_tp, 'r-s', label='Double DQN (DDQN)')
    ax4.set_title('Throughput vs SNR')
    ax4.set_xlabel('SNR (dB)'); ax4.set_ylabel('Throughput (Mbps)')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    # 5. DQN Training Curve
    ax5 = axes[1, 1]
    window = max(1, len(dqn_rewards)//20)
    avg_dqn = [np.mean(dqn_rewards[i:i+window]) for i in range(0, len(dqn_rewards), window)]
    ax5.plot(avg_dqn, 'b-', label='DQN')
    ax5.set_title('DQN Training Convergence')
    ax5.set_xlabel('Episode (averaged)'); ax5.set_ylabel('Reward')
    ax5.legend(); ax5.grid(True, alpha=0.3)

    # 6. DDQN Training Curve
    ax6 = axes[1, 2]
    avg_ddqn = [np.mean(ddqn_rewards[i:i+window]) for i in range(0, len(ddqn_rewards), window)]
    ax6.plot(avg_ddqn, 'r-', label='DDQN')
    ax6.set_title('DDQN Training Convergence')
    ax6.set_xlabel('Episode (averaged)'); ax6.set_ylabel('Reward')
    ax6.legend(); ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ml_crn_full_results.png', dpi=300)
    print("\n" + "="*70)
    print("Simulation complete! Results saved to 'ml_crn_full_results.png'")
    plt.show()

if __name__ == "__main__":
    run_simulation()
