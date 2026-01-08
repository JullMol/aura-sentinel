# 🛡️ Aura-Sentinel

<div align="center">

<img src="https://img.shields.io/badge/AI-Powered-10b981?style=for-the-badge&logo=openai&logoColor=white" alt="AI Powered">
<img src="https://img.shields.io/badge/Go-1.21+-00ADD8?style=for-the-badge&logo=go&logoColor=white" alt="Go">
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
<img src="https://img.shields.io/badge/Wails-2.11-8B5CF6?style=for-the-badge" alt="Wails">

---

### **Enterprise AI-Powered Customer Retention Platform**

*Combining XGBoost + Deep Q-Network Reinforcement Learning + Real-time Analytics*

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [AI Models](#-ai-models)

</div>

---

## 🎯 What is Aura-Sentinel?

Aura-Sentinel is an **enterprise-grade AI platform** for customer churn prediction and retention optimization. It helps businesses:

- 📉 **Predict** which customers are likely to churn
- 🎯 **Decide** the optimal retention action for each customer
- 💰 **Maximize** revenue saved while minimizing intervention costs
- 📊 **Visualize** real-time analysis in a modern dashboard

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 **Churn Prediction**
XGBoost model with **94% accuracy** predicting customer churn probability

### 🤖 **RL Action Selection**
Deep Q-Network agent optimizes retention actions (Email, SMS, Discounts, Personal Call)

### 🔮 **Oracle Mode**
What-if scenario analysis - adjust cost modifiers to see how AI decisions change

</td>
<td width="50%">

### 📊 **Live Matrix Feed**
Real-time customer processing with animated visualization

### 🧪 **Training Lab**
Upload custom datasets and train new models with one click

### 📋 **Reports & Export**
Filter by risk level, export to **PDF** and **CSV**

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AURA-SENTINEL DESKTOP APP                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│    │   FRONTEND   │     │   BACKEND    │     │   AI BRAIN   │       │
│    │              │     │              │     │              │       │
│    │  React 18    │◄───►│   Go 1.21    │◄───►│  Python 3.10 │       │
│    │  TypeScript  │     │   Wails 2.11 │     │  Flask API   │       │
│    │  Recharts    │     │   Bindings   │     │  PyTorch     │       │
│    │  Lucide      │     │              │     │  XGBoost     │       │
│    └──────────────┘     └──────────────┘     └──────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Install Command |
|------|---------|----------------|
| Go | 1.21+ | [Download](https://golang.org/dl/) |
| Node.js | 18+ | [Download](https://nodejs.org/) |
| Python | 3.10+ | [Download](https://python.org/) |
| Wails CLI | 2.11 | `go install github.com/wailsapp/wails/v2/cmd/wails@latest` |

### 1️⃣ Start Python Brain API

```bash
cd apps/brain-rl
python -m venv venv
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
python api.py
```

### 2️⃣ Run Wails Desktop App

```bash
cd apps
wails dev
```

### 3️⃣ Build for Production

```bash
cd apps
wails build
```

Output: `build/bin/Aura-Sentinel.exe`

---

## 📁 Project Structure

```
aura-sentinel/
├── 📂 apps/
│   ├── 📄 main.go           # Wails entry point
│   ├── 📄 app.go            # Engine bindings & API
│   ├── 📂 frontend/         # React TypeScript UI
│   │   ├── src/App.tsx      # Main dashboard component
│   │   └── src/App.css      # Premium dark theme
│   ├── 📂 brain-rl/         # Python AI models
│   │   ├── api.py           # Flask REST API
│   │   ├── *.pth            # PyTorch DQN weights
│   │   └── *.pkl            # XGBoost model
│   └── 📂 engine-go/        # Standalone batch processor
├── 📂 data/
│   └── dataset.xls          # Telco customer data
└── 📄 README.md
```

---

## 🧠 AI Models

### XGBoost Churn Predictor

| Metric | Value |
|--------|-------|
| Accuracy | **94%** |
| Features | 22 customer attributes |
| Output | Churn probability (0.0 - 1.0) |

### Deep Q-Network (DQN) Agent

| Component | Description |
|-----------|-------------|
| **State** | 9 features (churn prob, tenure, charges, contract, etc) |
| **Actions** | 6 retention actions with varying costs |
| **Reward** | Customer Lifetime Value saved - action cost |
| **Network** | 4-layer MLP (128→128→64→6) |

### Available Actions

| ID | Action | Cost |
|----|--------|------|
| 0 | No Action | 0% |
| 1 | Send Email | 1% |
| 2 | Send SMS | 2% |
| 3 | Offer 10% Discount | 10% |
| 4 | Offer 20% Discount | 20% |
| 5 | Personal Call + 30% Discount | 35% |

---

## 🔮 Oracle Mode

Adjust the cost modifier to simulate different business scenarios:

| Modifier | Effect |
|----------|--------|
| **0.5x** | Discounts are cheaper → AI prefers discounts |
| **1.0x** | Normal business pricing |
| **3.0x** | Discounts are costly → AI prefers Email/SMS |

This demonstrates how the RL agent adapts its strategy based on business constraints.

---

## �️ Dashboard Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Live matrix feed, Oracle control, charts |
| **Analytics** | Retention trends, AI performance metrics |
| **Training Lab** | Upload datasets, train new models |
| **Reports** | Filter & export customer data |

---

## 🛠️ Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Recharts, Lucide Icons
- **Desktop**: Wails 2.11 (Go + WebView2)
- **AI Backend**: Python, Flask, PyTorch, XGBoost, NumPy
- **Styling**: Custom CSS with glassmorphism, dark theme

---

## 📝 License

MIT License - Free for personal and commercial use.

---

<div align="center">

**Built with ❤️ using Go, Python, and React**

*A modern AI-powered desktop application for enterprise customer retention*

</div>
