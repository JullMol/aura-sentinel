# Aura-Sentinel 🛡️

<div align="center">

**Enterprise AI-Powered Customer Retention Platform**

*Reinforcement Learning + XGBoost + Real-time Analytics*

![Version](https://img.shields.io/badge/version-2.0-emerald)
![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![Wails](https://img.shields.io/badge/Wails-2.11-8B5CF6)

</div>

---

## 🎯 Overview

Aura-Sentinel is an enterprise-grade AI platform for customer churn prediction and retention optimization. It combines:

- **XGBoost** for accurate churn probability prediction
- **Deep Q-Network (DQN)** Reinforcement Learning for optimal action selection
- **Wails Desktop App** for native cross-platform experience
- **Real-time Matrix Stream** for live customer analysis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Wails Desktop App                        │
├─────────────────────────────────────────────────────────────┤
│  Frontend: React + TypeScript + Vite                        │
│  Backend:  Go (Wails bindings)                              │
│  AI Brain: Python (Flask + PyTorch + XGBoost)               │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Churn Prediction** | XGBoost model with 94% accuracy |
| 🤖 **RL Action Selection** | DQN agent optimizes retention actions |
| 🔮 **Oracle Mode** | What-if scenario analysis with cost modifiers |
| 📊 **Live Matrix Feed** | Real-time customer processing visualization |
| 🧪 **Training Lab** | Upload datasets and train custom models |
| 📋 **Reports** | Export to PDF/CSV with filtering |
| 🖥️ **Desktop App** | Native Windows/Mac/Linux via Wails |

## 🚀 Quick Start

### Prerequisites
- Go 1.21+
- Node.js 18+
- Python 3.10+
- Wails CLI (`go install github.com/wailsapp/wails/v2/cmd/wails@latest`)

### Run Development Mode

```bash
# 1. Start Python Brain API
cd apps/brain-rl
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
python api.py

# 2. Run Wails Desktop App (new terminal)
cd apps
wails dev
```

### Build for Production

```bash
cd apps
wails build
# Output: build/bin/Aura-Sentinel.exe
```

## 📁 Project Structure

```
aura-sentinel/
├── apps/
│   ├── main.go          # Wails entry point
│   ├── app.go           # Engine bindings & Python launcher
│   ├── frontend/        # React UI
│   ├── brain-rl/        # Python AI models
│   │   ├── api.py       # Flask API
│   │   ├── generate_models.py
│   │   └── *.pth, *.pkl # Trained models
│   ├── engine-go/       # Standalone engine (alternative)
│   └── dashboard-js/    # Web dashboard (alternative)
├── data/
│   └── dataset.xls      # Customer data
├── .gitignore
├── Makefile
└── README.md
```

## 🧠 AI Models

### XGBoost Baseline
- Predicts churn probability (0-1)
- Features: tenure, charges, contract type, services

### DQN Reinforcement Learning
- State: Customer features + churn probability
- Actions: No Action, Email, SMS, Discount 10/20%, Personal Call
- Reward: CLV retained - action cost

## 🔮 Oracle Mode

Adjust cost modifier to simulate business scenarios:

| Modifier | Effect |
|----------|--------|
| 0.5x | Discounts are cheaper → AI prefers discounts |
| 1.0x | Normal pricing |
| 3.0x | Discounts costly → AI prefers Email/SMS |

## 📸 Screenshots

*Coming soon - Run the app to see the modern dashboard!*

## 📝 License

MIT License - Free for personal and commercial use.

---

<div align="center">
  <b>Built with ❤️ using Go, Python, and React</b>
</div>
