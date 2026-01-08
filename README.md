# 🛡️ Aura-Sentinel

**AI-Powered Customer Churn Prediction & Retention Strategy Engine**

A production-ready, polyglot system that combines **Go** for high-performance data processing with **Python** for machine learning inference. This project demonstrates a real-world CRM AI pipeline that processes customer data, predicts churn probability, and recommends personalized retention actions.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AURA-SENTINEL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐         ┌──────────────────────────────┐    │
│   │  DATA INPUT  │         │       PYTHON BRAIN API       │    │
│   │  (CSV/Excel) │         │   ┌────────────────────────┐ │    │
│   │   7,043      │         │   │   XGBoost Classifier   │ │    │
│   │  customers   │────────▶│   │  (Churn Prediction)    │ │    │
│   └──────────────┘         │   └────────────────────────┘ │    │
│          │                 │   ┌────────────────────────┐ │    │
│          │                 │   │   DQN RL Agent         │ │    │
│   ┌──────▼──────┐          │   │  (Action Selection)    │ │    │
│   │  GO ENGINE  │──HTTP───▶│   └────────────────────────┘ │    │
│   │  (Batch     │          └──────────────────────────────┘    │
│   │  Processor) │                       │                      │
│   └──────┬──────┘                       │                      │
│          │                              ▼                      │
│   ┌──────▼──────────────────────────────────────────────┐     │
│   │              OUTPUT: retention_strategy.csv          │     │
│   │  CustomerID | ChurnProb | RiskLevel | Action         │     │
│   └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎯 Churn Prediction** | XGBoost model trained on 7,043 customers with 80%+ accuracy |
| **🤖 RL Action Recommendation** | Deep Q-Network agent selects optimal retention actions |
| **⚡ High-Performance Processing** | Go engine processes thousands of customers in seconds |
| **📊 Actionable Output** | CSV report with risk levels and recommended interventions |
| **🔗 Microservice Architecture** | Clean separation between data processing and AI inference |

---

## 🚀 Quick Start

### Prerequisites
- Go 1.21+
- Python 3.10+
- pip

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/aura-sentinel.git
cd aura-sentinel
```

### 2. Install Python Dependencies

```bash
cd apps/brain-rl
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install flask torch xgboost joblib numpy pandas scikit-learn tqdm
```

### 3. Generate AI Models (if not present)

```bash
python generate_models.py
```

### 4. Start the Brain API

```bash
python api.py
# 🧠 Loading AI Models...
#    ✅ XGBoost model loaded
#    ✅ RL Agent loaded
# 🚀 Aura-Sentinel Brain API Starting...
```

### 5. Run the Go Engine (new terminal)

```bash
cd apps/engine-go
go run .
```

---

## 📊 Sample Output

```
🚀 Aura-Sentinel Engine: Batch Processing Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Loaded 7043 customers from dataset

📊 Processing all customers...
   ⏳ Processed 500/7043 customers...
   ⏳ Processed 1000/7043 customers...
   ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 BATCH PROCESSING COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total Processed:    7043 customers
   Processing Time:    12.5s
   Avg Churn Risk:     26.5%

   📊 Risk Distribution:
      🔴 HIGH Risk:    1,862 (26.4%)
      🟡 MEDIUM Risk:  1,245 (17.7%)
      🟢 LOW Risk:     3,936 (55.9%)

✅ Results saved to: retention_strategy_results.csv
```

---

## 📁 Project Structure

```
aura-sentinel/
├── apps/
│   ├── brain-rl/                 # Python AI Brain
│   │   ├── api.py                # Flask REST API
│   │   ├── generate_models.py    # Model training script
│   │   ├── xgboost_baseline_model.pkl
│   │   └── rl_agent_checkpoint.pth
│   │
│   └── engine-go/                # Go Data Engine
│       ├── main.go               # Batch processor
│       └── data_reader.go        # CSV parser
│
├── data/
│   └── dataset.xls               # Customer dataset (7,043 records)
│
└── README.md
```

---

## 🧠 AI Models

### XGBoost Classifier
- **Purpose**: Predict churn probability (0-100%)
- **Features**: 22 customer attributes (tenure, charges, services, etc.)
- **Output**: Probability score

### DQN Reinforcement Learning Agent
- **Purpose**: Select optimal retention action
- **State**: 9-dimensional vector (churn prob, contract type, engagement, etc.)
- **Actions**: 6 possible interventions

| Action ID | Intervention | Cost |
|-----------|--------------|------|
| 0 | No Action | 0% |
| 1 | Send Email | 1% |
| 2 | Send SMS | 2% |
| 3 | Offer 10% Discount | 10% |
| 4 | Offer 20% Discount | 20% |
| 5 | Personal Call + 30% Discount | 35% |

---

## 🛠️ Technologies

| Layer | Technology | Purpose |
|-------|------------|---------|
| Data Processing | **Go** | High-performance CSV parsing & HTTP client |
| AI Inference | **Python** | XGBoost + PyTorch for ML models |
| API | **Flask** | RESTful microservice |
| ML Framework | **XGBoost, PyTorch** | Gradient boosting & Deep RL |

---

## 📈 Business Value

This system transforms raw customer data into **actionable business intelligence**:

1. **Identify At-Risk Customers** - Flag high churn probability accounts
2. **Optimize Interventions** - RL agent balances cost vs. effectiveness
3. **Scale Operations** - Process 7,000+ customers in seconds
4. **Data-Driven Decisions** - Export results for marketing team execution

---

## 📝 License

MIT License - feel free to use for portfolio, learning, or production.

---

## 🤝 Author

Built as a demonstration of **end-to-end AI system design** combining:
- Machine Learning (XGBoost, Deep Q-Learning)
- Polyglot Programming (Go + Python)
- Microservice Architecture
- Real Business Problem Solving
