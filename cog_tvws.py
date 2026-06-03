import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
import time

# ================= CONFIGURATION =================
CFG = {
    "freq_mhz": 600,
    "bandwidth_mhz": 8,
    "tx_power_min_w": 0.5,
    "tx_power_max_w": 5.0,
    "noise_floor_dbm": -100,
    "rural_clutter_db": 12,
    "distance_m": 8000,
    "channels": 4,
    "steps_per_ep": 100,
    "num_episodes": 30,
    "cost": {
        "wtd_radio": 450, "solar": 750, "tower": 900, "install": 350,
        "maint_per_yr": 100, "batt_rep_3yr": 120, "batt_wh": 200
    }
}

# ================= PROPAGATION & SIGNAL HELPERS =================
def pathloss_db(d, freq=CFG["freq_mhz"], clutter=CFG["rural_clutter_db"]):
    return 32.45 + 20*np.log10(freq) + 20*np.log10(d/1000) + clutter

def gen_signal(primary=True, snr_db=10):
    noise = np.random.randn(128) * 10**(CFG["noise_floor_dbm"]/20)
    if primary:
        sig_amp = 10**((CFG["noise_floor_dbm"] + snr_db)/20)
        sig = sig_amp * np.sin(2*np.pi*0.05*np.arange(128))
        return (sig + noise).reshape(1, -1)
    return noise.reshape(1, -1)

# ================= ML SENSING MODEL (1D-CNN) =================
class SpectrumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 3), nn.ReLU(),
            nn.Flatten(), nn.Linear(16*29, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

sensing_model = SpectrumCNN()

# ================= ML PREDICTION (Lightweight LSTM) =================
class OccupancyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, CFG["channels"]), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

predict_model = OccupancyLSTM()

# ================= ENVIRONMENT =================
class TVWSEnv:
    def __init__(self):
        self.reset()
        self.capex = sum(CFG["cost"].values())
        self.opex_accum = 0.0
        self.data_gb = 0.0
        self.energy_wh = 0.0

    def reset(self):
        self.step = 0
        self.occ_hist = np.random.binomial(1, 0.3, size=(1, 20, 1))  # [batch, seq, channels]
        self.primary_state = np.random.binomial(1, 0.3, size=CFG["channels"])
        self.battery_pct = 100.0
        return self._get_state()

    def _get_state(self):
        snr = np.array([10 if not self.primary_state[c] else -5 for c in range(CFG["channels"])])
        return np.concatenate([snr, self.occ_hist[0, -5:].flatten(), [self.battery_pct]])

    def step(self, action_ch, power_w):
        self.step += 1
        # Update primary occupancy (Markov)
        for c in range(CFG["channels"]):
            if random.random() < 0.1: self.primary_state[c] = 1 - self.primary_state[c]

        # Interference & SNR
        pl = pathloss_db(CFG["distance_m"])
        rx_power_dbm = 10*np.log10(power_w*1000) - pl
        noise_dbm = CFG["noise_floor_dbm"]
        snr = rx_power_dbm - noise_dbm if not self.primary_state[action_ch] else -10

        # Throughput (Shannon approx)
        bw_hz = CFG["bandwidth_mhz"] * 1e6
        throughput_bps = bw_hz * np.log2(1 + 10**(max(snr, -10)/10)) if snr > 5 else 0
        throughput_gbps = throughput_bps / 1e9
        self.data_gb += throughput_gbps * (1/3600)  # per step = 1 sec

        # Energy
        self.energy_wh += power_w * (1/3600) + 0.02  # base radio overhead
        self.battery_pct = max(0, self.battery_pct - (power_w*0.5 + 0.05))

        # Cost tracking
        self.opex_accum += (power_w * 0.1)  # simplified OPEX rate

        # Occupancy prediction ground truth for next step
        next_occ = self.primary_state.copy()
        reward = throughput_bps/1e6 - 0.5*(snr < 5) - 0.2*(self.primary_state[action_ch])
        
        self.step = self.step
        return self._get_state(), reward, self.step >= CFG["steps_per_ep"], {}

# ================= AGENTS =================
class BaselineAgent:
    def select_action(self, state):
        # Energy detection thresholding + random channel
        snr = state[:CFG["channels"]]
        valid = [c for c in range(CFG["channels"]) if snr[c] > 5]
        ch = random.choice(valid) if valid else random.randint(0, CFG["channels"]-1)
        pwr = CFG["tx_power_max_w"] * 0.6  # fixed
        return ch, pwr

class DQNAgent:
    def __init__(self, state_dim=24, action_dim=CFG["channels"]*3):
        self.q_net = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, action_dim))
        self.target = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, action_dim))
        self.target.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), 1e-3)
        self.mem = deque(maxlen=5000)
        self.eps = 1.0
        self.gamma = 0.95
        self.batch = 32

    def act(self, state):
        if random.random() < self.eps:
            ch, pwr_idx = random.randint(0, CFG["channels"]-1), random.randint(0, 2)
        else:
            with torch.no_grad():
                q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
                act = q.argmax().item()
                ch, pwr_idx = act // 3, act % 3
        pwr = CFG["tx_power_min_w"] + pwr_idx*(CFG["tx_power_max_w"]-CFG["tx_power_min_w"])/2
        return ch, pwr

    def learn(self, batch):
        s, a, r, ns, d = zip(*batch)
        s, a, r, ns, d = torch.FloatTensor(s), torch.LongTensor(a), torch.FloatTensor(r), torch.FloatTensor(ns), torch.FloatTensor(d)
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad:
            tq = r + self.gamma * self.target(ns).max(1)[0] * (1-d)
        loss = nn.MSELoss()(q, tq)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eps = max(0.05, self.eps*0.995)

    def update_target(self): self.target.load_state_dict(self.q_net.state_dict())

# ================= RUN SIMULATION =================
def run_sim():
    env = TVWSEnv()
    bl, dq = BaselineAgent(), DQNAgent()
    bl_metrics, dq_metrics = [], []
    
    print("🔹 Training DQN...")
    for ep in range(CFG["num_episodes"]):
        state = env.reset()
        done = False
        while not done:
            ch, pwr = dq.act(state)
            nstate, r, done, _ = env.step(ch, pwr)
            dq.mem.append((state, ch*3+int((pwr-CFG["tx_power_min_w"])/((CFG["tx_power_max_w"]-CFG["tx_power_min_w"])/2)), r, nstate, done))
            state = nstate
            if len(dq.mem) >= dq.batch:
                dq.learn(random.sample(dq.mem, dq.batch))
            if ep % 5 == 0 and ep > 0: dq.update_target()
        if ep % 10 == 0: print(f"  Ep {ep}/{CFG['num_episodes']} | ε={dq.eps:.2f}")

    print("🔹 Evaluating Baseline vs DQN...")
    for agent, name in [(bl, "Baseline"), (dq, "DQN")]:
        env.__init__()  # reset cost trackers
        state = env.reset()
        done = False
        while not done:
            ch, pwr = agent.act(state)
            state, _, done, _ = env.step(ch, pwr)
        metrics = {"Agent": name, "Data_GB": env.data_gb, "Energy_Wh": env.energy_wh, 
                   "OPEX_$": env.opex_accum, "CAPEX_$": env.capex, 
                   "Cost_per_GB_$": (env.capex/5 + env.opex_accum)/max(env.data_gb, 1e-6)}
        (dq_metrics if name=="DQN" else bl_metrics).append(metrics)

    df = pd.DataFrame(dq_metrics + bl_metrics)
    print("\n📊 COST & PERFORMANCE SUMMARY:")
    print(df.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(8,4))
    plt.bar(df["Agent"], df["Cost_per_GB_$"])
    plt.ylabel("Cost per GB ($)")
    plt.title("TVWS Rural Deployment: Cost Comparison")
    plt.show()
    return df

if __name__ == "__main__":
    run_sim()
