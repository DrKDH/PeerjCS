# ECIA: Emotional-Cognition Integration Architecture

This repository contains the official implementation of the paper:
**"Biologically-Inspired Emotional Processing for Adaptive Decision-Making in Non-Stationary Environments"**

## Overview

The **Emotional-Cognition Integration Architecture (ECIA)** is a biologically-inspired reinforcement learning framework designed to manage environmental uncertainty. Unlike traditional methods that utilize statistical thresholds, this study demonstrates that emotion-like mechanisms (External Limbic System) function as **adaptive heuristics** for rapid uncertainty management.

This codebase reproduces the **large-scale experimental replication (N=3,600 runs)** reported in the manuscript, comparing ECIA against state-of-the-art non-stationary baselines.

## Key Features

  * **Biologically-Inspired Architecture:** Implements 8 computational emotions (Plutchik's wheel), episodic memory retrieval, and dopamine-modulated learning rates.
  * **Non-Stationary Environments:**
      * `Environment A`: Sudden Strategy Reversal (Shock)
      * `Environment B`: Predictable Alternation (Cyclic)
      * `Environment C`: Stochastic Disruptions (High Uncertainty)
  * **Improved Baselines:**
      * Sliding Window UCB (SW-UCB)
      * Adaptive Thompson Sampling (Adaptive TS)
      * Context-Aware Epsilon-Greedy
  * **Computational Tractability:** Optimized for standard research workstations.

## System Requirements & Replication

All experiments in the paper were conducted using the following specifications:

  * **OS:** Windows 10 Pro
  * **CPU:** Intel Core i7-6700 CPU @ 3.40GHz
  * **RAM:** 16GB
  * **Python Version:** 3.8

The simulation relies on **12 specific Fibonacci seeds** (34, 55, ..., 6765) to ensure statistical robustness and reproducibility of the reported results.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/DrKDH/PeerjCS.git
    cd PeerjCS
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the full simulation suite and generate the performance metrics:

```bash
python ECIA_improved.py
```

  * **Execution Time:** Approximately 20-30 minutes for the full suite (3,600 runs).
  * **Outputs:** The script will generate CSV files containing mean rewards and recovery metrics, corresponding to the Tables and Figures in the manuscript.

## Main Findings Replicated

Running this code will reproduce the following key findings:

1.  **Functional Specialization:** ECIA statistically outperforms baselines in uncertain/stochastic environments (Env A & C).
2.  **Cost of Complexity:** Simpler baselines (like SW-UCB) outperform ECIA in strictly predictable environments (Env B).
3.  **Synergistic Integration:** Ablation studies demonstrate that removing components (Emotion, Memory, Dopamine) causes non-additive performance degradation.

## License

This project is open-source and available for academic and research purposes.
