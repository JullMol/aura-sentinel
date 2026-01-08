import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
from tqdm import tqdm
import joblib

df = pd.read_csv('../../data/dataset.xls')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
df_clean = df.drop('customerID', axis=1)

df_features = df_clean.copy()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df_features[col] = df_features[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

df_features['MultipleLines'] = df_features['MultipleLines'].map({'Yes': 2, 'No': 1, 'No phone service': 0})
df_features['InternetService'] = df_features['InternetService'].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})

service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df_features[col] = df_features[col].map({'Yes': 2, 'No': 1, 'No internet service': 0})

df_features['Contract'] = df_features['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

payment_dummies = pd.get_dummies(df_features['PaymentMethod'], prefix='Payment')
df_features = pd.concat([df_features, payment_dummies], axis=1)
df_features.drop('PaymentMethod', axis=1, inplace=True)

X = df_features.drop('Churn', axis=1)
y = df_features['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
joblib.dump(model, 'xgboost_baseline_model.pkl')

np.random.seed(42)
X_test_rl = X_test.copy()
X_test_rl['prev_interventions'] = np.random.poisson(lam=1.5, size=len(X_test_rl)).clip(0, 5)
X_test_rl['days_since_contact'] = np.random.randint(0, 91, size=len(X_test_rl))
X_test_rl['response_rate'] = np.random.beta(2, 5, size=len(X_test_rl))
X_test_rl['engagement_score'] = np.random.normal(50, 20, size=len(X_test_rl)).clip(0, 100)
X_test_rl['churn_probability'] = model.predict_proba(X_test)[:, 1]

ACTION_COSTS = {0: 0.0, 1: 0.01, 2: 0.02, 3: 0.10, 4: 0.20, 5: 0.35}


def get_state(customer_data):
    return np.array([
        customer_data['churn_probability'],
        customer_data['Contract'],
        customer_data['InternetService'],
        customer_data['tenure'] / 72.0,
        customer_data['MonthlyCharges'] / 120.0,
        customer_data['prev_interventions'] / 5.0,
        customer_data['days_since_contact'] / 90.0,
        customer_data['response_rate'],
        customer_data['engagement_score'] / 100.0,
    ])


def calculate_reward(action, customer_data, retained):
    monthly_charge = customer_data['MonthlyCharges']
    action_cost = ACTION_COSTS[action] * monthly_charge
    customer_ltv = monthly_charge * 36
    
    if retained:
        reward = customer_ltv - action_cost
    else:
        reward = -action_cost
    
    if action > 0:
        if customer_data['prev_interventions'] >= 3:
            reward -= monthly_charge * 0.1
        if customer_data['response_rate'] < 0.2:
            reward -= monthly_charge * 0.05
    return reward


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory()
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class CustomerRetentionEnv:
    def __init__(self, X_data, y_data, churn_model):
        self.X_data = X_data.reset_index(drop=True)
        self.y_data = y_data.reset_index(drop=True)
        self.churn_model = churn_model
        self.current_idx = 0
    
    def reset(self):
        self.current_idx = random.randint(0, len(self.X_data) - 1)
        customer_data = self.X_data.iloc[self.current_idx]
        return get_state(customer_data)
    
    def step(self, action):
        customer_data = self.X_data.iloc[self.current_idx]
        actual_churn = self.y_data.iloc[self.current_idx]
        intervention_success = self.simulate_intervention(action, customer_data, actual_churn)
        retained = not actual_churn or intervention_success
        reward = calculate_reward(action, customer_data, retained)
        next_state = get_state(customer_data)
        done = True
        return next_state, reward, done, retained
    
    def simulate_intervention(self, action, customer_data, actual_churn):
        if action == 0:
            return False
        action_success_prob = {1: 0.1, 2: 0.15, 3: 0.3, 4: 0.5, 5: 0.7}
        base_prob = action_success_prob[action]
        
        if customer_data['response_rate'] > 0.5:
            base_prob += 0.1
        if customer_data['engagement_score'] > 70:
            base_prob += 0.1
        if customer_data['prev_interventions'] >= 3:
            base_prob -= 0.15
        return random.random() < max(0, min(1, base_prob))


agent = DQNAgent(state_size=9, action_size=6)
env = CustomerRetentionEnv(X_test_rl, y_test, model)

for episode in tqdm(range(1000)):
    state = env.reset()
    action = agent.select_action(state, training=True)
    next_state, reward, done, retained = env.step(action)
    agent.store_experience(state, action, reward, next_state, done)
    agent.train_step()
    
    if episode % 10 == 0:
        agent.update_target_network()
    agent.decay_epsilon()

torch.save({
    'policy_net': agent.policy_net.state_dict(),
    'target_net': agent.target_net.state_dict(),
    'optimizer': agent.optimizer.state_dict(),
}, 'rl_agent_checkpoint.pth')
