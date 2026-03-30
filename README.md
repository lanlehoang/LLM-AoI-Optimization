# LLM-GUIDE-Routing

## Overview
LLM-GUIDE-Routing implements **LLM-GUIDE**, a novel framework that enhances Deep Reinforcement Learning (DRL) with Large Language Model (LLM)-generated heuristics.

**LLM-GUIDE (LLM-based Generative Uncertainty-aware Inductive heuristics for DQN Enhancement)** integrates:
- A baseline Deep Q-Network (DQN)
- An Evolution-of-Heuristics (EoH) refinement process

---

## Key Features
- Baseline DQN training pipeline
- LLM-powered heuristic generation
- Heuristic evolution (EoH)
- Modular evaluation framework

---

## Installation

### 1. Create a virtual environment

**Conda**
```bash
conda create -n aoi_venv python=3.10
conda activate aoi_venv
```

**venv**
```bash
python3.10 -m venv aoi_venv
source aoi_venv/bin/activate
```

---

### 2. Install PyTorch (manual step)

**CPU**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.x)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. Install dependencies
```bash
pip install -e .
```

---

## Configuration

### Setup environment variables
```bash
cp .env.example .env
```

Edit `.env` to configure:
- LLM provider (OpenAI, etc.)
- API keys
- Model settings

---

## Usage

### 1. Train baseline DQN
```bash
python -m src.train.baseline
```

---

### 2. Run LLM-GUIDE (EoH)
```bash
pytho -m src.train.run_eoh
```

---

### 3. Evaluate performance
```bash
python -m src.evaluate
```

---

## Method Summary

LLM-GUIDE enhances DQN by:
1. Generating heuristic policies using LLMs
2. Injecting heuristics into the action-selection process
3. Iteratively refining heuristics via Evolution-of-Heuristics (EoH)

This enables:
- Faster convergence
- Better exploration
- Improved robustness under uncertainty

---

## Notes
- Install PyTorch manually before dependencies
- Ensure `.env` is properly configured
- Most behavior is controlled via YAML configs

---

## Future Work
- Multi-agent extensions
- Better heuristic selection strategies
- Real-world deployment scenarios

---
