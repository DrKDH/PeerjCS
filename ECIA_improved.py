import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy import stats
import pickle
from tqdm import tqdm
import warnings
from collections import deque
import random
warnings.filterwarnings('ignore')

# Create result directories
os.makedirs("content/Results", exist_ok=True)

class ExperimentalReplicationManager:
    """
    Manages large-scale experimental replication for statistical robustness.
    Uses Fibonacci sequence master seeds to ensure systematic variation
    across 3,600 independent trials per agent per environment.
    
    Note: This is NOT meta-analysis (which aggregates independent studies),
    but rather extensive experimental replication with Monte Carlo simulation.
    """

    def __init__(self):
        # Fibonacci sequence (for systematic seed variation)
        self.SEEDS = [34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
        self.N_RUNS_PER_SEED = 300
        self.RANDOMSHIFT_SEEDS = [34, 55, 89]
        self.RANDOMSHIFT_N_RUNS_PER_SEED = 300
        self.TOTAL_EXPERIMENTS = len(self.SEEDS) * self.N_RUNS_PER_SEED

        # Logging and tracking
        self.seed_logs = {}
        self.replication_results = {}

    def get_seeds_and_runs(self, env_name):
        """Get environment-specific seeds and runs"""
        return self.SEEDS, self.N_RUNS_PER_SEED

    def generate_experiment_seeds(self, master_seed, env_name=None):
        """Generate experiment seeds from a Fibonacci master seed"""
        np.random.seed(master_seed)
        experiment_seeds = []

        n_runs = self.N_RUNS_PER_SEED

        for i in range(n_runs):
            seed = np.random.randint(0, 1000000)
            experiment_seeds.append(seed)

        self.seed_logs[master_seed] = {
            'first_10_seeds': experiment_seeds[:10],
            'total_seeds': len(experiment_seeds),
            'seed_range': [min(experiment_seeds), max(experiment_seeds)]
        }

        return experiment_seeds

    def save_configuration(self, filepath="content/Results/config.txt"):
        """Save complete experimental replication configuration"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("Experimental Replication Configuration\n")
            f.write("=" * 50 + "\n\n")

            f.write("REGULAR ENVIRONMENTS (EnvA, EnvB, EnvC):\n")
            f.write(f"Master Seeds: {self.SEEDS}\n")
            f.write(f"Runs per Master Seed: {self.N_RUNS_PER_SEED}\n")
            f.write(f"Total Experiments: {len(self.SEEDS) * self.N_RUNS_PER_SEED}\n\n")

            f.write("CROSS-DATASET ENVIRONMENT (RandomShift):\n")
            f.write(f"Master Seeds: {self.RANDOMSHIFT_SEEDS}\n")
            f.write(f"Runs per Master Seed: {self.RANDOMSHIFT_N_RUNS_PER_SEED}\n")
            f.write(f"Total Experiments: {len(self.RANDOMSHIFT_SEEDS) * self.RANDOMSHIFT_N_RUNS_PER_SEED}\n\n")

            f.write("Seed Generation Details:\n")
            f.write("-" * 30 + "\n")
            for master_seed, log in self.seed_logs.items():
                f.write(f"Master Seed {master_seed}:\n")
                f.write(f"  First 10 experiment seeds: {log['first_10_seeds']}\n")
                f.write(f"  Seed range: {log['seed_range']}\n")
                f.write(f"  Total generated: {log['total_seeds']}\n\n")


# Global manager instance
MANAGER = ExperimentalReplicationManager()

# =============================================
# ENVIRONMENT CLASSES
# =============================================

class EnvironmentA:
    """Environment A with deterministic seed control"""

    def __init__(self, total_trials=200, sigma=0.15, random_state=None):
        self.total_trials = total_trials
        self.trial = 0
        self.n_actions = 5
        self.sigma = sigma
        self.context = None

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def reset(self):
        self.trial = 0
        self.update_context()
        return None

    def step(self, action):
        reward = self.get_reward(action)
        self.trial += 1
        self.update_context()
        return reward, self.context

    def get_reward(self, action):
        t = self.trial
        if t < 50:
            means = [0.8, 0.2, 0.2, 0.2, 0.2]
        elif t < 100:
            means = [0.8, 0.2, 0.3, 0.2, 0.2]
        elif t < 150:
            means = [0.2, 0.2, 0.3, 0.2, 0.9]
        else:
            means = [0.2, 0.4, 0.3, 0.2, 0.9]

        return self.rng.normal(means[action], self.sigma)

    def update_context(self):
        t = self.trial
        norm_t = t / self.total_trials
        if t < 50:
            phase_id = 0
        elif t < 100:
            phase_id = 1
        elif t < 150:
            phase_id = 2
        else:
            phase_id = 3
        self.context = np.array([norm_t, phase_id])


class EnvironmentB:
    """Environment B with deterministic seed control"""

    def __init__(self, total_trials=200, sigma=0.05, random_state=None):
        self.total_trials = total_trials
        self.trial = 0
        self.n_actions = 5
        self.sigma = sigma
        self.context = None

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def reset(self):
        self.trial = 0
        self.update_context()
        return None

    def step(self, action):
        reward = self.get_reward(action)
        self.trial += 1
        self.update_context()
        return reward, self.context

    def get_reward(self, action):
        t = self.trial
        base = [0.2] * 5
        phase = t // 40

        if phase % 2 == 0:
            base[0] = 0.95; base[4] = 0.01
        else:
            base[0] = 0.01; base[4] = 0.95

        for i in [1, 2, 3]:
            base[i] = 0.05

        reward = base[action] + self.rng.normal(0, self.sigma)
        return np.clip(reward, 0, 1)

    def update_context(self):
        t = self.trial
        norm_t = t / self.total_trials
        phase = t // 40
        phase_id = phase % 2
        self.context = np.array([norm_t, phase_id])


class EnvironmentC:
    """Environment C with controlled randomness while maintaining stochastic nature"""
    
    def __init__(self, total_trials=200, sigma=0.15,
                 disturbance_prob=0.05, disturbance_strength=0.7,
                 random_state=None):
        self.total_trials = total_trials
        self.n_actions = 5
        self.sigma = sigma
        self.trial = 0
        self.disturbance_prob = disturbance_prob
        self.disturbance_strength = disturbance_strength

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

        self._generate_random_configuration()

        self.active_disturbance = False
        self.disturbance_action = None
        self.disturbance_duration = 0
        self.context = None
        self.update_reward_structure()

    def _generate_random_configuration(self):
        """Generate random configuration using controlled randomness"""
        self.change_points = sorted(self.rng.choice(range(30, 181), size=3, replace=False).tolist())

        self.optimal_actions = []
        self.optimal_rewards = []
        used_actions = set()

        for i in range(4):
            if len(used_actions) < self.n_actions:
                unused_actions = [a for a in range(self.n_actions) if a not in used_actions]
                if unused_actions:
                    opt_action = self.rng.choice(unused_actions)
                else:
                    prev_action = self.optimal_actions[-1] if self.optimal_actions else -1
                    opt_action = (prev_action + 1) % self.n_actions
            else:
                prev_action = self.optimal_actions[-1] if self.optimal_actions else -1
                available_actions = [a for a in range(self.n_actions) if a != prev_action]
                opt_action = self.rng.choice(available_actions)

            used_actions.add(opt_action)
            self.optimal_actions.append(opt_action)
            opt_reward = self.rng.uniform(0.75, 0.9)
            self.optimal_rewards.append(opt_reward)

    def reset(self):
        self.trial = 0
        self.active_disturbance = False
        self.disturbance_action = None
        self.disturbance_duration = 0
        self.update_reward_structure()
        return None

    def step(self, action):
        reward = self.rng.normal(self.means[action], self.sigma)
        self.trial += 1
        self.update_reward_structure()
        return reward, self.context

    def update_reward_structure(self):
        t = self.trial
        norm_t = t / self.total_trials

        phase = 0
        for i, cp in enumerate(self.change_points):
            if t >= cp:
                phase = i + 1
            else:
                break

        optimal_action = self.optimal_actions[phase]
        optimal_reward = self.optimal_rewards[phase]

        means = [0.2] * self.n_actions
        means[optimal_action] = optimal_reward

        if self.active_disturbance:
            self.disturbance_duration -= 1
            if self.disturbance_duration > 0 and self.disturbance_action != optimal_action:
                means[self.disturbance_action] = self.disturbance_strength
            else:
                self.active_disturbance = False
                self.disturbance_action = None

        elif self.rng.random() < self.disturbance_prob:
            non_optimal_actions = [a for a in range(self.n_actions) if a != optimal_action]
            if non_optimal_actions:
                self.disturbance_action = self.rng.choice(non_optimal_actions)
                self.active_disturbance = True
                self.disturbance_duration = self.rng.randint(1, 3)
                means[self.disturbance_action] = self.disturbance_strength

        self.means = means
        self.context = np.array([norm_t, phase])

    def get_current_optimal_info(self):
        """Get current segment's optimal action and reward"""
        phase = 0
        for i, cp in enumerate(self.change_points):
            if self.trial >= cp:
                phase = i + 1
            else:
                break

        return {
            'phase': phase,
            'optimal_action': self.optimal_actions[phase],
            'optimal_reward': self.optimal_rewards[phase],
            'change_point': self.change_points[phase-1] if phase > 0 else 0
        }


class RandomShiftEnvironment:
    """Random shift environment for cross-dataset testing"""
    
    def __init__(self, total_trials=600, n_actions=5, random_state=None):
        self.total_trials = total_trials
        self.n_actions = n_actions
        self.trial = 0
        self.sigma = 0.12

        if random_state is not None:
            self.master_rng = np.random.RandomState(random_state)
        else:
            self.master_rng = np.random.RandomState()

        self.reward_history = deque(maxlen=50)
        self.action_sequence = deque(maxlen=10)
        self.extreme_event_counter = 0
        self.stored_patterns = []
        self.short_term_cycle = 0
        self.medium_term_cycle = 0
        self.long_term_phase = 0
        self.segments = self._generate_enhanced_segments()
        self.current_segment_idx = 0
        self.segment_start_trial = 0
        self.current_rewards = [0.2] * self.n_actions
        self.context = None
        self.base_noise_level = 0.12
        self._initialize_current_segment()

    def _generate_enhanced_segments(self):
        """Generate segments with rich complexity patterns"""
        segments = []
        current_trial = 0
        phase_types = ['learning', 'challenging', 'mastery']
        phase_length = self.total_trials // 3

        for phase_idx, phase_type in enumerate(phase_types):
            phase_start = phase_idx * phase_length
            phase_end = min((phase_idx + 1) * phase_length, self.total_trials)

            phase_trials = phase_start
            while phase_trials < phase_end:
                if phase_type == 'learning':
                    segment_length = self.master_rng.randint(80, 120)
                elif phase_type == 'challenging':
                    segment_length = self.master_rng.randint(60, 100)
                else:
                    segment_length = self.master_rng.randint(40, 80)

                segment_end = min(phase_trials + segment_length, phase_end)

                segment = {
                    'start': phase_trials,
                    'end': segment_end,
                    'length': segment_end - phase_trials,
                    'phase_type': phase_type,
                    'phase_idx': phase_idx,
                    'config': self._generate_segment_config(phase_type, phase_idx, segment_end - phase_trials)
                }

                segments.append(segment)
                phase_trials = segment_end

        return segments

    def _generate_segment_config(self, phase_type, phase_idx, segment_length):
        """Generate segment configuration"""
        config = {
            'phase_type': phase_type,
            'reward_structure': 'layered',
            'is_reused_pattern': False
        }

        if phase_idx > 0 and self.stored_patterns and self.master_rng.random() < 0.25:
            base_pattern = self.master_rng.choice(self.stored_patterns)

            if self.master_rng.random() < 0.6:
                config.update(base_pattern)
                config['is_reused_pattern'] = True
                config['reuse_type'] = 'identical'
            else:
                config.update(self._create_similar_pattern(base_pattern))
                config['is_reused_pattern'] = True
                config['reuse_type'] = 'similar'
        else:
            config.update(self._generate_new_pattern_config(phase_type, segment_length))

            if phase_idx == 0:
                pattern_to_store = config.copy()
                self.stored_patterns.append(pattern_to_store)

        return config

    def _generate_new_pattern_config(self, phase_type, segment_length):
        """Generate new pattern configuration"""
        actions = list(range(self.n_actions))
        self.master_rng.shuffle(actions)

        config = {}

        if phase_type == 'learning':
            config.update({
                'excellent_actions': actions[:1],
                'good_actions': actions[1:2],
                'medium_actions': actions[2:3],
                'poor_actions': actions[3:4],
                'bad_actions': actions[4:],
                'noise_multiplier': 0.8,
                'extreme_event_prob': 0.02,
                'sequence_bonus_prob': 0.1,
                'emotional_shock_point': segment_length // 2,
                'shock_magnitude': 0.4
            })
        elif phase_type == 'challenging':
            config.update({
                'excellent_actions': actions[:1],
                'good_actions': actions[1:3],
                'medium_actions': actions[3:4],
                'poor_actions': actions[4:],
                'bad_actions': [],
                'noise_multiplier': 1.0,
                'extreme_event_prob': 0.05,
                'sequence_bonus_prob': 0.15,
                'context_dependent_bonus': 0.15,
                'memory_window': 20
            })
        else:
            config.update({
                'excellent_actions': actions[:2],
                'good_actions': actions[2:3],
                'medium_actions': actions[3:],
                'poor_actions': [],
                'bad_actions': [],
                'noise_multiplier': 1.2,
                'extreme_event_prob': 0.08,
                'sequence_bonus_prob': 0.2,
                'expectation_violation_prob': 0.06,
                'long_term_memory_bonus': 0.1
            })

        return config

    def _create_similar_pattern(self, base_pattern):
        """Create similar but not identical pattern"""
        similar_pattern = base_pattern.copy()
        variation_type = self.master_rng.choice(['timing', 'magnitude', 'order'])

        if variation_type == 'timing':
            if 'emotional_shock_point' in similar_pattern:
                original_point = similar_pattern['emotional_shock_point']
                variation = int(original_point * 0.2)
                similar_pattern['emotional_shock_point'] = original_point + self.master_rng.randint(-variation, variation+1)

        elif variation_type == 'magnitude':
            if 'shock_magnitude' in similar_pattern:
                similar_pattern['shock_magnitude'] *= self.master_rng.uniform(0.8, 1.2)
            if 'context_dependent_bonus' in similar_pattern:
                similar_pattern['context_dependent_bonus'] *= self.master_rng.uniform(0.8, 1.2)

        elif variation_type == 'order':
            if 'good_actions' in similar_pattern and 'medium_actions' in similar_pattern:
                if len(similar_pattern['good_actions']) == 1 and len(similar_pattern['medium_actions']) == 1:
                    temp = similar_pattern['good_actions'][0]
                    similar_pattern['good_actions'][0] = similar_pattern['medium_actions'][0]
                    similar_pattern['medium_actions'][0] = temp

        similar_pattern['reuse_type'] = 'similar'
        return similar_pattern

    def _initialize_current_segment(self):
        """Initialize current segment"""
        if self.current_segment_idx >= len(self.segments):
            return

        segment = self.segments[self.current_segment_idx]
        self.segment_start_trial = segment['start']
        self._update_rewards()

    def _update_rewards(self):
        """Update reward structure"""
        segment = self.segments[self.current_segment_idx]
        config = segment['config']
        trial_in_segment = self.trial - self.segment_start_trial

        rewards = [0.2] * self.n_actions

        for action in config.get('excellent_actions', []):
            rewards[action] = self.master_rng.uniform(0.80, 0.90)
        for action in config.get('good_actions', []):
            rewards[action] = self.master_rng.uniform(0.75, 0.82)
        for action in config.get('medium_actions', []):
            rewards[action] = self.master_rng.uniform(0.45, 0.55)
        for action in config.get('poor_actions', []):
            rewards[action] = self.master_rng.uniform(0.15, 0.25)
        for action in config.get('bad_actions', []):
            rewards[action] = self.master_rng.uniform(0.05, 0.15)

        self._apply_ecia_features(rewards, config, trial_in_segment)
        self._apply_sequence_effects(rewards, config)
        self._apply_temporal_patterns(rewards, trial_in_segment)

        self.current_rewards = rewards

    def _apply_ecia_features(self, rewards, config, trial_in_segment):
        """Apply ECIA-specific features"""
        if 'emotional_shock_point' in config and trial_in_segment == config['emotional_shock_point']:
            excellent_actions = config.get('excellent_actions', [])
            poor_actions = config.get('poor_actions', [])

            if excellent_actions and poor_actions:
                excellent_action = excellent_actions[0]
                poor_action = poor_actions[0]
                rewards[excellent_action] = 0.1
                rewards[poor_action] = 0.85

        if 'context_dependent_bonus' in config and len(self.reward_history) >= 5:
            recent_avg = np.mean(list(self.reward_history)[-5:])
            context_bonus = config['context_dependent_bonus']

            if recent_avg > 0.6:
                for action in config.get('excellent_actions', []):
                    rewards[action] += context_bonus
            elif recent_avg < 0.4:
                for action in config.get('medium_actions', []):
                    rewards[action] += context_bonus * 0.5

        if 'long_term_memory_bonus' in config and len(self.reward_history) >= 30:
            distant_avg = np.mean(list(self.reward_history)[-30:-20])
            recent_avg = np.mean(list(self.reward_history)[-10:])

            if abs(distant_avg - recent_avg) < 0.1:
                memory_bonus = config['long_term_memory_bonus']
                for i in range(self.n_actions):
                    rewards[i] += memory_bonus

        if 'expectation_violation_prob' in config:
            if self.master_rng.random() < config['expectation_violation_prob']:
                surprise_action = self.master_rng.randint(0, self.n_actions)
                if self.master_rng.random() < 0.7:
                    rewards[surprise_action] = min(0.95, rewards[surprise_action] + 0.3)
                else:
                    rewards[surprise_action] = max(0.05, rewards[surprise_action] - 0.3)

    def _apply_sequence_effects(self, rewards, config):
        """Apply sequence effects"""
        if len(self.action_sequence) >= 3:
            last_actions = list(self.action_sequence)[-3:]

            if len(set(last_actions)) == 3:
                if self.master_rng.random() < config.get('sequence_bonus_prob', 0.1):
                    for i in range(self.n_actions):
                        rewards[i] += 0.1

            elif len(set(last_actions)) == 1:
                repeated_action = last_actions[0]
                rewards[repeated_action] *= 0.85

    def _apply_temporal_patterns(self, rewards, trial_in_segment):
        """Apply temporal patterns"""
        short_cycle_pos = (trial_in_segment % 12) / 12.0
        short_modifier = 0.05 * np.sin(2 * np.pi * short_cycle_pos)

        medium_cycle_pos = (trial_in_segment % 65) / 65.0
        medium_modifier = 0.03 * np.cos(2 * np.pi * medium_cycle_pos)

        for i in range(self.n_actions):
            rewards[i] += short_modifier + medium_modifier
            rewards[i] = max(0.05, min(0.95, rewards[i]))

    def _update_context(self):
        """Update context"""
        norm_t = self.trial / self.total_trials
        phase_id = self.current_segment_idx % 4
        self.context = np.array([norm_t, phase_id])

    def step(self, action):
        """Execute one step"""
        current_segment = self.segments[self.current_segment_idx]
        if self.trial >= current_segment['end'] and self.current_segment_idx < len(self.segments) - 1:
            self.current_segment_idx += 1
            self._initialize_current_segment()

        self._update_rewards()

        base_reward = self.current_rewards[action]

        segment = self.segments[self.current_segment_idx]
        noise_multiplier = segment['config'].get('noise_multiplier', 1.0)
        noise = self.master_rng.normal(0, self.sigma * noise_multiplier)
        final_reward = np.clip(base_reward + noise, 0, 1)

        self.reward_history.append(final_reward)
        self.action_sequence.append(action)

        self.trial += 1
        self._update_context()

        return final_reward, self.context

    def reset(self):
        """Reset environment"""
        self.trial = 0
        self.current_segment_idx = 0
        self.segment_start_trial = 0
        self.reward_history.clear()
        self.action_sequence.clear()
        self.extreme_event_counter = 0

        self._initialize_current_segment()
        self._update_context()
        return None

    def get_change_points(self):
        """Get change points"""
        return [segment['start'] for segment in self.segments[1:]]


# =============================================
# IMPROVED BASELINE AGENTS
# =============================================

class ContextAwareEpsilonGreedy:
    """
    Epsilon-Greedy with temporal awareness and change detection.
    Fair baseline that can adapt to non-stationary environments.
    """
    def __init__(self, n_actions=5, epsilon=0.1, alpha=0.1, 
                 window_size=50, change_threshold=0.25, random_state=None):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.window_size = window_size
        self.change_threshold = change_threshold
        
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_actions = deque(maxlen=window_size)
        self.reward_by_action = [deque(maxlen=20) for _ in range(n_actions)]
        
        self.last_change_time = 0
        self.time = 0
        self.cooldown = 0
        
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def detect_environment_change(self):
        """Detect significant change in reward distribution"""
        if len(self.recent_rewards) < 30 or self.cooldown > 0:
            return False
        
        recent_10 = list(self.recent_rewards)[-10:]
        older_10 = list(self.recent_rewards)[-30:-20]
        
        mean_diff = abs(np.mean(recent_10) - np.mean(older_10))
        volatility = np.std(recent_10) + np.std(older_10)
        
        if mean_diff > self.change_threshold and mean_diff > volatility * 0.5:
            return True
        
        return False

    def reset_on_change(self):
        """Partial reset when environment change detected"""
        self.q_values *= 0.5
        self.action_counts *= 0.5
        
        for deque_list in self.reward_by_action:
            deque_list.clear()
        
        self.cooldown = 20
        self.last_change_time = self.time

    def select_action(self):
        if self.rng.rand() < self.epsilon:
            return self.rng.choice(self.n_actions)
        return np.argmax(self.q_values)

    def update(self, action, reward, context=None):
        """Update with temporal awareness"""
        self.time += 1
        
        self.recent_rewards.append(reward)
        self.recent_actions.append(action)
        self.reward_by_action[action].append(reward)
        
        if self.detect_environment_change():
            self.reset_on_change()
        
        if self.cooldown > 0:
            self.cooldown -= 1
        
        self.action_counts[action] += 1
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def reset(self):
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.recent_rewards.clear()
        self.recent_actions.clear()
        for deque_list in self.reward_by_action:
            deque_list.clear()
        self.time = 0
        self.last_change_time = 0
        self.cooldown = 0


class SlidingWindowUCB:
    """
    UCB with sliding window for non-stationary environments.
    Standard approach in bandit literature for changing environments.
    Reference: Garivier & Moulines (2011)
    """
    def __init__(self, n_actions=5, c=2.0, window_size=100, 
                 min_samples=5, random_state=None):
        self.n_actions = n_actions
        self.c = c
        self.window_size = window_size
        self.min_samples = min_samples
        
        self.reward_history = [deque(maxlen=window_size) 
                               for _ in range(n_actions)]
        self.time = 0
        
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def select_action(self):
        self.time += 1
        
        for a in range(self.n_actions):
            if len(self.reward_history[a]) < self.min_samples:
                return a
        
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            recent_rewards = list(self.reward_history[a])
            mean_reward = np.mean(recent_rewards)
            n_a = len(recent_rewards)
            
            bonus = self.c * np.sqrt(np.log(self.time) / n_a)
            ucb_values[a] = mean_reward + bonus
        
        return np.argmax(ucb_values)

    def update(self, action, reward, context=None):
        """Update with sliding window"""
        self.reward_history[action].append(reward)

    def reset(self):
        for deque_list in self.reward_history:
            deque_list.clear()
        self.time = 0


class AdaptiveThompsonSampling:
    """
    Thompson Sampling with exponential recency weighting.
    Appropriate for non-stationary environments.
    Reference: Raj & Kalyani (2017)
    """
    def __init__(self, n_actions=5, discount=0.99, min_std=0.1, 
                 forget_threshold=100, random_state=None):
        self.n_actions = n_actions
        self.discount = discount
        self.min_std = min_std
        self.forget_threshold = forget_threshold
        
        self.priors = [(0.5, 1.0) for _ in range(n_actions)]
        
        self.sum_rewards = np.zeros(n_actions)
        self.sum_squared_rewards = np.zeros(n_actions)
        self.effective_counts = np.zeros(n_actions)
        
        self.time = 0
        
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def select_action(self):
        samples = []
        for a in range(self.n_actions):
            mu, sigma = self.priors[a]
            sample = self.rng.normal(mu, max(sigma, self.min_std))
            samples.append(sample)
        
        return np.argmax(samples)

    def update(self, action, reward, context=None):
        """Update with discount factor"""
        self.time += 1
        
        self.sum_rewards *= self.discount
        self.sum_squared_rewards *= self.discount
        self.effective_counts *= self.discount
        
        self.sum_rewards[action] += reward
        self.sum_squared_rewards[action] += reward ** 2
        self.effective_counts[action] += 1
        
        if self.effective_counts[action] > 0:
            n = self.effective_counts[action]
            mean = self.sum_rewards[action] / n
            
            mean_sq = self.sum_squared_rewards[action] / n
            variance = max(mean_sq - mean**2, self.min_std**2)
            std = np.sqrt(variance / n)
            
            self.priors[action] = (mean, std)
        
        if self.time % self.forget_threshold == 0:
            for a in range(self.n_actions):
                mu, sigma = self.priors[a]
                self.priors[a] = (mu, min(sigma * 1.1, 1.0))

    def reset(self):
        self.priors = [(0.5, 1.0) for _ in range(self.n_actions)]
        self.sum_rewards = np.zeros(self.n_actions)
        self.sum_squared_rewards = np.zeros(self.n_actions)
        self.effective_counts = np.zeros(self.n_actions)
        self.time = 0


# =============================================
# NAIVE BASELINE AGENTS (for comparison)
# =============================================

class EpsilonGreedyAgent:
    """Naive Epsilon-Greedy without adaptation"""
    def __init__(self, n_actions=5, epsilon=0.1, random_state=None):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def reset(self):
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)

    def select_action(self):
        if self.rng.rand() < self.epsilon:
            return self.rng.choice(self.n_actions)
        return np.argmax(self.q_values)

    def update(self, action, reward, context=None):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])


class ThompsonSamplingAgent:
    """Naive Thompson Sampling without discount"""
    def __init__(self, n_actions=5, random_state=None):
        self.n_actions = n_actions
        self.priors = [(0.0, 1.0) for _ in range(n_actions)]
        self.observations = [[] for _ in range(n_actions)]

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def reset(self):
        self.priors = [(0.0, 1.0) for _ in range(self.n_actions)]
        self.observations = [[] for _ in range(self.n_actions)]

    def select_action(self):
        samples = [self.rng.normal(mu, sigma) for mu, sigma in self.priors]
        return np.argmax(samples)

    def update(self, action, reward, context=None):
        self.observations[action].append(reward)
        data = self.observations[action]
        if len(data) > 1:
            mu = np.mean(data)
            sigma = np.std(data) if np.std(data) > 0 else 1.0
            self.priors[action] = (mu, sigma)


class UCBAgent:
    """Naive UCB without windowing"""
    def __init__(self, n_actions=5, c=0.5, random_state=None):
        self.n_actions = n_actions
        self.c = c
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.total_steps = 0

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

    def reset(self):
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.total_steps = 0

    def select_action(self):
        self.total_steps += 1
        ucb_values = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            if self.action_counts[a] == 0:
                return a
            bonus = self.c * np.sqrt(np.log(self.total_steps) / self.action_counts[a])
            ucb_values[a] = self.q_values[a] + bonus

        return np.argmax(ucb_values)

    def update(self, action, reward, context=None):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])


# =============================================
# ECIA AGENT (keeping original implementation)
# =============================================

class ECIA:
    """Full ECIA implementation with emotion, memory, and dopamine systems"""

    def __init__(self, n_actions=5, epsilon=0.03, eta=0.55, xi=0.001,
                 memory_threshold=0.015, memory_influence=0.3,
                 window_size=30, min_eta=0.095, memory_size=15,
                 alpha=0.22, memory_similarity_threshold=0.035,
                 top_k=3, emotion_decay=0.96, random_state=None):

        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()

        self.n_actions = n_actions
        self.epsilon = epsilon
        self.xi = xi
        self.alpha = alpha
        self.emotion_decay = emotion_decay
        self.window_size = window_size
        self.min_eta = min_eta
        self.top_k = top_k

        self.base_eta = eta
        self.eta = self.base_eta
        self.eta_adaptation_counter = 0

        self.base_memory_threshold = memory_threshold
        self.base_memory_influence = memory_influence
        self.memory_size = memory_size
        self.memory_similarity_threshold = memory_similarity_threshold
        self.memory_activation_level = 1.0
        self.memory_quality_threshold = 0.15
        self.memory_usage_history = deque(maxlen=20)
        self.memory_cooldown = 0

        self.memory_effectiveness_tracker = deque(maxlen=50)
        self.environment_stability_tracker = deque(maxlen=30)
        self.change_detection_window = deque(maxlen=15)
        self.stable_performance_counter = 0

        self.context_clusters = {}
        self.cluster_performance = {}
        self.use_context_clustering = True

        self.q_values = np.zeros(n_actions)
        self.emotion = np.zeros(8)
        self.action_counts = np.zeros(n_actions)
        self.time = 0
        self.prev_reward = 0.5
        self.context = None

        self.emotion_names = [
            "fear", "joy", "hope", "sadness",
            "curiosity", "anger", "pride", "shame"
        ]
        self.emotion_weight = np.array([
            -0.15, 0.4, 0.3, -0.2, 0.35, -0.25, 0.25, -0.3
        ])
        self.max_total_emotion_energy = 2.5
        self.emotion_momentum = np.zeros(8)

        self.episodic_memory = []

        self.performance_tracker = deque(maxlen=25)
        self.recent_context_changes = deque(maxlen=10)
        self.action_history = deque(maxlen=20)
        self.reward_history = deque(maxlen=20)
        self.learning_boost = 0.2
        self.successful_emotion_patterns = {}
        self.neurogenesis_cycle = 25
        self.emotion_learning_rates = np.array([
            0.15, 0.25, 0.20, 0.12, 0.30, 0.18, 0.22, 0.28
        ])
        self.emotion_action_history = deque(maxlen=12)
        self.context_change_threshold = 0.1
        self.habit_strength_factor = 0.025

    def assess_environment_stability(self):
        """Assess environment stability"""
        if len(self.recent_context_changes) < 5:
            return 0.5

        context_changes = list(self.recent_context_changes)
        avg_change = np.mean(context_changes)
        change_variance = np.var(context_changes)
        context_stability = 1.0 - min(avg_change + change_variance * 0.5, 1.0)

        if len(self.reward_history) >= 10:
            recent_rewards = list(self.reward_history)[-10:]
            reward_stability = 1.0 - min(np.std(recent_rewards), 1.0)
        else:
            reward_stability = 0.5

        if len(self.performance_tracker) >= 8:
            recent_performance = list(self.performance_tracker)[-8:]
            performance_trend = abs(np.polyfit(range(8), recent_performance, 1)[0])
            trend_stability = 1.0 - min(performance_trend * 5, 1.0)
        else:
            trend_stability = 0.5

        stability = 0.4 * context_stability + 0.4 * reward_stability + 0.2 * trend_stability
        self.environment_stability_tracker.append(stability)
        return np.mean(self.environment_stability_tracker) if self.environment_stability_tracker else 0.5

    def evaluate_memory_effectiveness(self):
        """Evaluate memory effectiveness"""
        if len(self.memory_usage_history) < 10:
            return 0.5

        memory_used_rewards = [r for used, r in self.memory_usage_history if used]
        memory_unused_rewards = [r for used, r in self.memory_usage_history if not used]

        if len(memory_used_rewards) > 3 and len(memory_unused_rewards) > 3:
            used_avg = np.mean(memory_used_rewards)
            unused_avg = np.mean(memory_unused_rewards)
            effectiveness = (used_avg - unused_avg + 1.0) / 2.0
        else:
            effectiveness = 0.5

        self.memory_effectiveness_tracker.append(effectiveness)
        return np.mean(self.memory_effectiveness_tracker) if self.memory_effectiveness_tracker else 0.5

    def adaptive_memory_control(self):
        """Adaptive memory control"""
        effectiveness = self.evaluate_memory_effectiveness()

        if effectiveness > 0.6:
            self.memory_activation_level = 0.8 + 0.2 * effectiveness
        elif effectiveness > 0.4:
            self.memory_activation_level = 0.4 + 0.4 * effectiveness
        else:
            self.memory_activation_level = 0.1 + 0.3 * effectiveness

        self.memory_activation_level = np.clip(self.memory_activation_level, 0.05, 1.0)

        self.current_memory_threshold = self.base_memory_threshold / max(0.1, self.memory_activation_level)
        self.current_memory_influence = self.base_memory_influence * self.memory_activation_level

    def identify_context_cluster(self, context):
        """Identify context cluster"""
        if context is None or len(context) < 2:
            return "default"

        norm_t = context[0] if len(context) > 0 else 0
        phase_id = int(context[1]) if len(context) > 1 else 0
        time_cluster = int(norm_t * 4)

        return f"phase_{phase_id}_time_{time_cluster}"

    def compute_memory_quality_score(self, action, reward, prediction_error):
        """Compute memory quality score"""
        error_score = min(abs(prediction_error), 0.5) / 0.5
        extreme_reward_score = abs(reward - 0.5) / 0.5
        emotion_intensity = np.linalg.norm(self.emotion) / np.sqrt(8)

        return 0.4 * error_score + 0.4 * extreme_reward_score + 0.2 * emotion_intensity

    def store_adaptive_memory(self, action, reward, prediction_error):
        """Store adaptive memory"""
        if self.memory_activation_level < 0.1 or self.context is None:
            return

        quality_score = self.compute_memory_quality_score(action, reward, prediction_error)

        if quality_score < self.memory_quality_threshold:
            return

        if abs(prediction_error) > self.current_memory_threshold:
            memory = {
                'action': action,
                'reward': reward,
                'context': self.context.copy(),
                'time': self.time,
                'prediction_error': abs(prediction_error),
                'quality_score': quality_score,
                'emotion_state': self.emotion.copy()
            }

            if self.use_context_clustering and self.memory_activation_level > 0.6:
                cluster_id = self.identify_context_cluster(self.context)
                memory['cluster_id'] = cluster_id

                if cluster_id not in self.context_clusters:
                    self.context_clusters[cluster_id] = []
                    self.cluster_performance[cluster_id] = deque(maxlen=20)

                self.context_clusters[cluster_id].append(memory)
                self.cluster_performance[cluster_id].append(reward)

                if len(self.context_clusters[cluster_id]) > self.memory_size // 6:
                    self.context_clusters[cluster_id].sort(
                        key=lambda x: x['quality_score'], reverse=True
                    )
                    self.context_clusters[cluster_id] = self.context_clusters[cluster_id][:self.memory_size // 6]

            self.episodic_memory.append(memory)
            if len(self.episodic_memory) > self.memory_size:
                self.episodic_memory.sort(key=lambda x: (
                    0.6 * x['quality_score'] + 0.4 * (x['time'] / max(1, self.time))
                ), reverse=True)
                self.episodic_memory = self.episodic_memory[:self.memory_size]

    def compute_adaptive_memory_bias(self):
        """Compute adaptive memory bias"""
        if self.context is None or self.memory_activation_level < 0.1:
            self.memory_usage_history.append((False, self.prev_reward))
            return np.zeros(self.n_actions)

        relevant_memories = []

        if self.use_context_clustering and self.memory_activation_level > 0.5 and self.context_clusters:
            cluster_id = self.identify_context_cluster(self.context)
            cluster_memories = self.context_clusters.get(cluster_id, [])

            if cluster_memories:
                relevant_memories.extend(cluster_memories[-3:])

            if len(relevant_memories) < 2:
                for cid, memories in self.context_clusters.items():
                    if len(memories) > 0 and cid in self.cluster_performance:
                        cluster_perf = np.mean(self.cluster_performance[cid])
                        if cluster_perf > 0.6:
                            relevant_memories.extend(memories[-2:])
                            if len(relevant_memories) >= 4:
                                break

        if len(relevant_memories) < 3:
            similarity_memories = []
            for memory in self.episodic_memory[-30:]:
                if memory.get('context') is not None:
                    context_sim = self.compute_similarity(self.context, memory['context'])
                    if context_sim > self.memory_similarity_threshold:
                        emotion_sim = self.compute_similarity(self.emotion, memory['emotion_state'])
                        combined_sim = 0.7 * context_sim + 0.3 * emotion_sim
                        similarity_memories.append((combined_sim, memory))

            if similarity_memories:
                similarity_memories.sort(key=lambda x: x[0], reverse=True)
                additional_memories = [mem for _, mem in similarity_memories[:max(1, 4-len(relevant_memories))]]
                relevant_memories.extend(additional_memories)

        if len(relevant_memories) < 2:
            high_quality_memories = [m for m in self.episodic_memory if m['quality_score'] > 0.7]
            if high_quality_memories:
                relevant_memories.extend(high_quality_memories[-2:])

        if not relevant_memories:
            self.memory_usage_history.append((False, self.prev_reward))
            return np.zeros(self.n_actions)

        bias = np.zeros(self.n_actions)
        total_weight = 0

        for memory in relevant_memories[-5:]:
            action = memory['action']
            reward = memory['reward']
            quality = memory['quality_score']
            weight = quality * self.current_memory_influence

            if reward > 0.6:
                bias[action] += reward * weight
            elif reward < 0.4:
                bias[action] -= (0.5 - reward) * weight * 0.5

            total_weight += weight

        if total_weight > 0:
            bias = bias / (1.0 + total_weight * 0.2)

        self.memory_usage_history.append((True, self.prev_reward))
        return bias

    def emotional_processing(self, reward):
        """Emotional processing"""
        if self.prev_reward is None:
            self.prev_reward = 0.5

        self.emotion = self.emotion_decay * self.emotion
        self.reward_history.append(reward)
        recent_rewards = list(self.reward_history)

        current_emotion_updates = np.zeros(8)
        intensity_factor = 0.7

        # Fear
        if reward < self.prev_reward - 0.15:
            fear_strength = min(0.6, 0.15 + 0.4 * abs(reward - self.prev_reward))
            current_emotion_updates[0] = fear_strength * intensity_factor

        # Joy
        if reward > 0.7:
            joy_strength = min(0.7, 0.2 + 0.5 * reward)
            current_emotion_updates[1] = joy_strength * intensity_factor

        # Hope
        if len(recent_rewards) >= 4:
            recent_trend = np.polyfit(range(4), recent_rewards[-4:], 1)[0]
            if recent_trend > 0.03:
                hope_strength = min(0.6, 0.15 + 0.6 * recent_trend * 10)
                current_emotion_updates[2] = hope_strength * intensity_factor

        # Sadness
        if len(recent_rewards) >= 6:
            avg_recent = np.mean(recent_rewards[-6:])
            if avg_recent < 0.4:
                sadness_strength = min(0.5, 0.1 + 0.4 * (0.4 - avg_recent) / 0.4)
                current_emotion_updates[3] = sadness_strength * intensity_factor

        # Curiosity
        if len(self.action_history) > 0:
            action_diversity = len(set(self.action_history)) / min(len(self.action_history), self.n_actions)
            recent_performance = np.mean(recent_rewards[-3:]) if len(recent_rewards) >= 3 else 0.5

            if recent_performance < 0.6 or action_diversity < 0.8:
                curiosity_strength = min(0.7, 0.2 + 0.3 * (1 - action_diversity) +
                                       0.2 * max(0, 0.6 - recent_performance))
                current_emotion_updates[4] = curiosity_strength * intensity_factor

        # Anger
        expected_improvement = 0.05 * self.time / 200
        expected_reward = 0.5 + expected_improvement
        if reward < expected_reward - 0.2:
            anger_strength = min(0.5, 0.1 + 0.4 * abs(reward - expected_reward))
            current_emotion_updates[5] = anger_strength * intensity_factor

        # Pride
        if len(recent_rewards) >= 5:
            success_rate = sum(r > 0.7 for r in recent_rewards[-5:]) / 5
            if success_rate > 0.6:
                pride_strength = min(0.6, 0.1 + 0.4 * success_rate)
                current_emotion_updates[6] = pride_strength * intensity_factor

        # Shame
        if len(recent_rewards) >= 4:
            failure_rate = sum(r < 0.3 for r in recent_rewards[-4:]) / 4
            if failure_rate > 0.5:
                shame_strength = min(0.4, 0.1 + 0.3 * failure_rate)
                current_emotion_updates[7] = shame_strength * intensity_factor

        self.resolve_emotion_conflicts(current_emotion_updates)

        for i in range(8):
            self.emotion_momentum[i] = 0.7 * self.emotion_momentum[i] + 0.3 * current_emotion_updates[i]
            emotion_change = 0.5 * current_emotion_updates[i] + 0.5 * self.emotion_momentum[i]
            self.emotion[i] = 0.7 * self.emotion[i] + 0.3 * emotion_change

        self.normalize_emotions()
        self.prev_reward = reward

    def resolve_emotion_conflicts(self, emotion_updates):
        """Resolve emotion conflicts"""
        conflicting_pairs = [(0, 1), (2, 3), (6, 7)]

        for idx1, idx2 in conflicting_pairs:
            if emotion_updates[idx1] > 0.5 and emotion_updates[idx2] > 0.5:
                avg_strength = (emotion_updates[idx1] + emotion_updates[idx2]) / 2
                emotion_updates[idx1] = avg_strength * 0.8
                emotion_updates[idx2] = avg_strength * 0.8

    def normalize_emotions(self):
        """Normalize emotions"""
        self.emotion = np.clip(self.emotion, 0.0, 1.0)
        total_emotion_energy = np.sum(self.emotion)

        if total_emotion_energy > self.max_total_emotion_energy:
            self.emotion = self.emotion * (self.max_total_emotion_energy / total_emotion_energy)

    def adaptive_eta_adjustment(self):
        """Adaptive eta adjustment"""
        self.eta_adaptation_counter += 1

        if self.eta_adaptation_counter >= 20 and len(self.performance_tracker) >= 10:
            recent_performance = np.mean(list(self.performance_tracker)[-10:])
            performance_variance = np.var(list(self.performance_tracker)[-10:])

            if recent_performance > 0.75:
                self.eta = self.base_eta * 0.6
            elif recent_performance > 0.6:
                self.eta = self.base_eta * 0.8
            elif recent_performance < 0.4:
                self.eta = self.base_eta * 1.3
            elif performance_variance > 0.05:
                self.eta = self.base_eta * 1.1
            else:
                self.eta = self.base_eta

            self.eta = np.clip(self.eta, self.base_eta * 0.3, self.base_eta * 1.5)
            self.eta_adaptation_counter = 0

    def select_top_emotions(self):
        """Select top emotions"""
        emotion_indices = np.argsort(self.emotion)[-self.top_k:]
        selective_emotions = np.zeros(8)
        selective_emotions[emotion_indices] = self.emotion[emotion_indices]
        return selective_emotions

    def compute_direct_emotion_influence(self):
        """Compute direct emotion influence"""
        influences = np.zeros(self.n_actions)
        selective_emotions = self.select_top_emotions()

        # Fear
        if selective_emotions[0] > 0.1:
            fear_level = selective_emotions[0]
            min_q = np.min(self.q_values)
            max_q = np.max(self.q_values)
            q_range = max_q - min_q + 0.001

            for action in range(self.n_actions):
                relative_badness = (max_q - self.q_values[action]) / q_range
                influences[action] -= fear_level * 0.4 * relative_badness

        # Joy
        if selective_emotions[1] > 0.1:
            joy_level = selective_emotions[1]
            min_q = np.min(self.q_values)
            max_q = np.max(self.q_values)
            q_range = max_q - min_q + 0.001

            for action in range(self.n_actions):
                relative_goodness = (self.q_values[action] - min_q) / q_range
                influences[action] += joy_level * 0.4 * relative_goodness

        # Curiosity
        if selective_emotions[4] > 0.1:
            curiosity_level = selective_emotions[4]
            min_count = np.min(self.action_counts)
            max_count = np.max(self.action_counts) + 1

            for action in range(self.n_actions):
                exploration_factor = (max_count - self.action_counts[action]) / max_count
                influences[action] += curiosity_level * 0.4 * exploration_factor

        return influences

    def hippocampal_neurogenesis(self):
        """Hippocampal neurogenesis"""
        if self.time % self.neurogenesis_cycle == 0:
            emotion_intensity = np.linalg.norm(self.emotion)
            base_boost = 0.2

            if emotion_intensity > 0.7:
                self.learning_boost = base_boost * 1.3
            elif self.emotion[4] > 0.6:
                self.learning_boost = base_boost * 1.2
            else:
                self.learning_boost = base_boost
        else:
            self.learning_boost = max(0, self.learning_boost - 0.01)

    def prefrontal_modulation(self):
        """Prefrontal modulation"""
        if len(self.episodic_memory) < 5:
            return 0.5

        recent_rewards = [mem['reward'] for mem in self.episodic_memory[-10:]]
        reward_stability = 1.0 - np.std(recent_rewards) if recent_rewards else 0.5

        context_change = np.mean(self.recent_context_changes) if self.recent_context_changes else 0
        emotion_volatility = np.std(self.emotion) if np.sum(self.emotion) > 0 else 0
        emotion_stability = 1.0 - min(emotion_volatility, 1.0)

        performance_trend = 0.5
        if len(self.performance_tracker) >= 5:
            recent_performance = list(self.performance_tracker)[-5:]
            performance_trend = np.mean(recent_performance)

        stability = (0.35 * reward_stability + 0.25 * emotion_stability -
                    0.15 * context_change + 0.25 * performance_trend)

        return np.clip(stability, 0.1, 0.9)

    def compute_uncertainty_bonus(self):
        """Compute uncertainty bonus"""
        uncertainty = np.zeros(self.n_actions)
        total_experiences = self.time + 1

        for a in range(self.n_actions):
            action_count = self.action_counts[a] + 1
            count_uncertainty = np.sqrt(np.log(total_experiences) / action_count)

            action_rewards = [mem['reward'] for mem in self.episodic_memory if mem['action'] == a]

            if len(action_rewards) > 1:
                reward_std = np.std(action_rewards)
            else:
                reward_std = 0.5

            emotion_factor = 1.0
            if self.emotion[4] > 0.6:
                emotion_factor = 1.3
            elif self.emotion[0] > 0.6:
                emotion_factor = 0.7

            uncertainty[a] = count_uncertainty * (1 + reward_std) * emotion_factor

        return uncertainty * 0.3

    def build_context(self, norm_t, phase_id):
        """Build context"""
        grid_patterns = []

        for scale in [1.0, 3.0, 6.0]:
            for offset in [0.0, 0.33, 0.67]:
                grid_patterns.append(np.sin(2*np.pi * (norm_t * scale + offset)))
                grid_patterns.append(np.cos(2*np.pi * (norm_t * scale + offset)))

        time_cells = [
            np.exp(-(norm_t - 0.25)**2 / 0.15),
            np.exp(-(norm_t - 0.5)**2 / 0.15),
            np.exp(-(norm_t - 0.75)**2 / 0.15)
        ]

        emotion_context = self.emotion * 1.5

        return np.concatenate([grid_patterns, time_cells, [phase_id], emotion_context])

    def update_context(self, norm_t, phase_id):
        """Update context"""
        self.prev_context = self.context.copy() if self.context is not None else None
        self.context = self.build_context(norm_t, phase_id)

        if self.prev_context is not None:
            min_len = min(len(self.context), len(self.prev_context))
            context_change = np.linalg.norm(self.context[:min_len] - self.prev_context[:min_len])
            self.recent_context_changes.append(context_change)

    def select_action(self):
        """Select action"""
        self.hippocampal_neurogenesis()
        self.adaptive_eta_adjustment()
        self.adaptive_memory_control()

        pfc_stability = self.prefrontal_modulation()

        curiosity_boost = self.emotion[4] * 0.2
        fear_penalty = self.emotion[0] * 0.25
        joy_exploitation = self.emotion[1] * 0.15

        adaptive_epsilon = self.epsilon * (1 - pfc_stability)
        adaptive_epsilon = np.clip(
            adaptive_epsilon + curiosity_boost - fear_penalty - joy_exploitation,
            0.01, 0.2
        )

        if self.rng.rand() < adaptive_epsilon:
            if self.emotion[4] > 0.6:
                action_probs = 1.0 / (self.action_counts + 0.1)
                action_probs = action_probs / np.sum(action_probs)
                return self.rng.choice(self.n_actions, p=action_probs)
            elif self.emotion[5] > 0.6:
                if len(self.action_history) > 0:
                    recent_action = self.action_history[-1]
                    available_actions = [a for a in range(self.n_actions) if a != recent_action]
                    if available_actions:
                        return self.rng.choice(available_actions)
            return self.rng.choice(self.n_actions)

        emotion_influences = self.compute_direct_emotion_influence()
        memory_bias = self.compute_adaptive_memory_bias()
        uncertainty_bonus = self.compute_uncertainty_bonus()

        final_values = (self.q_values +
                       self.eta * emotion_influences +
                       memory_bias +
                       uncertainty_bonus +
                       self.xi * self.rng.randn(self.n_actions))

        return np.argmax(final_values)

    def update_dopamine_learning(self, action, reward):
        """Dopamine-based learning"""
        prediction_error = reward - self.q_values[action]
        emotion_intensity = np.linalg.norm(self.emotion)

        base_alpha = self.alpha
        emotional_boost = 1.0 + emotion_intensity * 0.3

        if abs(prediction_error) > 0.3:
            adaptive_alpha = base_alpha * 1.3 * emotional_boost
        elif abs(prediction_error) > 0.15:
            adaptive_alpha = base_alpha * 1.1 * emotional_boost
        else:
            adaptive_alpha = base_alpha * emotional_boost

        adaptive_alpha += self.learning_boost

        if self.emotion[4] > 0.6:
            adaptive_alpha *= 1.2

        if self.emotion[0] > 0.7:
            adaptive_alpha *= 0.85

        self.q_values[action] += adaptive_alpha * prediction_error

        habit_strength = min(0.3, self.habit_strength_factor * self.action_counts[action])
        if reward > 0.65:
            self.q_values[action] += habit_strength * reward

        self.performance_tracker.append(reward)
        return prediction_error

    def update(self, action, reward, context=None):
        """Update agent"""
        if context is not None:
            self.context = context

        self.hippocampal_neurogenesis()
        prediction_error = self.update_dopamine_learning(action, reward)
        self.emotional_processing(reward)
        self.store_adaptive_memory(action, reward, prediction_error)

        self.action_history.append(action)
        self.action_counts[action] += 1
        self.time += 1

    def compute_similarity(self, c1, c2):
        """Compute similarity"""
        if not isinstance(c1, np.ndarray) or not isinstance(c2, np.ndarray):
            return 0.0

        min_len = min(len(c1), len(c2))
        if min_len == 0:
            return 0.0

        c1_part, c2_part = c1[:min_len], c2[:min_len]

        if np.linalg.norm(c1_part) == 0 or np.linalg.norm(c2_part) == 0:
            return 0.0

        return np.dot(c1_part, c2_part) / (np.linalg.norm(c1_part) * np.linalg.norm(c2_part))

    def reset(self):
        """Reset agent"""
        self.q_values = np.zeros(self.n_actions)
        self.emotion = np.zeros(8)
        self.emotion_momentum = np.zeros(8)
        self.action_counts = np.zeros(self.n_actions)
        self.time = 0
        self.prev_reward = 0.5
        self.context = None
        self.eta = self.base_eta
        self.learning_boost = 0.2
        self.successful_emotion_patterns = {}

        self.memory_effectiveness_tracker.clear()
        self.environment_stability_tracker.clear()
        self.memory_activation_level = 1.0
        self.memory_usage_history.clear()
        self.change_detection_window.clear()
        self.stable_performance_counter = 0
        self.memory_cooldown = 0
        self.context_clusters.clear()
        self.cluster_performance.clear()

        self.performance_tracker.clear()
        self.recent_context_changes.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.episodic_memory.clear()
        self.emotion_action_history.clear()
        self.eta_adaptation_counter = 0


# =============================================
# ABLATION STUDY CLASSES
# =============================================

class ECIA_NoEmotion(ECIA):
    """ECIA without emotion system"""
    def emotional_processing(self, reward):
        if self.prev_reward is None:
            self.prev_reward = 0.5
        self.emotion = np.zeros(8)
        self.prev_reward = reward


class ECIA_NoMemory(ECIA):
    """ECIA without memory system"""
    def store_adaptive_memory(self, action, reward, prediction_error):
        pass

    def compute_adaptive_memory_bias(self):
        return np.zeros(self.n_actions)


class ECIA_NoDopamine(ECIA):
    """ECIA without dopamine adaptation"""
    def update_dopamine_learning(self, action, reward):
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += 0.1 * prediction_error
        self.performance_tracker.append(reward)
        return prediction_error


class ECIA_NoDopamine_NoMemory(ECIA):
    """ECIA without dopamine and memory"""
    def update_dopamine_learning(self, action, reward):
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += 0.1 * prediction_error
        self.performance_tracker.append(reward)
        return prediction_error

    def store_adaptive_memory(self, action, reward, prediction_error):
        pass

    def compute_adaptive_memory_bias(self):
        return np.zeros(self.n_actions)


class ECIA_NoDopamine_NoEmotion(ECIA):
    """ECIA without dopamine and emotion"""
    def update_dopamine_learning(self, action, reward):
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += 0.1 * prediction_error
        self.performance_tracker.append(reward)
        return prediction_error

    def emotional_processing(self, reward):
        if self.prev_reward is None:
            self.prev_reward = 0.5
        self.emotion = np.zeros(8)
        self.prev_reward = reward


class ECIA_NoMemory_NoEmotion(ECIA):
    """ECIA without memory and emotion"""
    def store_adaptive_memory(self, action, reward, prediction_error):
        pass

    def compute_adaptive_memory_bias(self):
        return np.zeros(self.n_actions)

    def emotional_processing(self, reward):
        if self.prev_reward is None:
            self.prev_reward = 0.5
        self.emotion = np.zeros(8)
        self.prev_reward = reward


class ECIA_NoAll_Components(ECIA):
    """ECIA without all advanced components"""
    def update_dopamine_learning(self, action, reward):
        prediction_error = reward - self.q_values[action]
        self.q_values[action] += 0.1 * prediction_error
        self.performance_tracker.append(reward)
        return prediction_error

    def store_adaptive_memory(self, action, reward, prediction_error):
        pass

    def compute_adaptive_memory_bias(self):
        return np.zeros(self.n_actions)

    def emotional_processing(self, reward):
        if self.prev_reward is None:
            self.prev_reward = 0.5
        self.emotion = np.zeros(8)
        self.prev_reward = reward


# =============================================
# EXPERIMENT EXECUTION FUNCTIONS
# =============================================

def run_single_experiment(env_class, agent_class, agent_kwargs,
                         n_trials=200, env_name="Unknown",
                         experiment_seed=42):
    """Run a single experiment with specific seed"""

    try:
        env = env_class(random_state=experiment_seed)

        agent_kwargs_with_seed = agent_kwargs.copy()
        agent_kwargs_with_seed['random_state'] = experiment_seed + 1
        agent = agent_class(**agent_kwargs_with_seed)

        env.reset()
        agent.reset()

        rewards = []
        actions = []

        for t in range(n_trials):
            action = agent.select_action()
            reward, context = env.step(action)

            # ✅ All agents receive context (but may not use it)
            agent.update(action, reward, context=context)

            rewards.append(reward)
            actions.append(action)

        return {
            "rewards": np.array(rewards),
            "actions": np.array(actions),
            "mean_reward": np.mean(rewards),
            "success": True,
            "experiment_seed": experiment_seed,
            "env_instance": env
        }

    except Exception as e:
        print(f"    Experiment error: {e}")
        return {
            "rewards": np.zeros(n_trials),
            "actions": np.zeros(n_trials),
            "mean_reward": 0.0,
            "success": False,
            "experiment_seed": experiment_seed,
            "env_instance": None
        }


def run_master_seed_experiments(master_seed, env_class, agent_class,
                               agent_kwargs, n_trials=200, env_name="Environment"):
    """Run experiments for a single master seed"""

    seeds = MANAGER.SEEDS
    n_runs = MANAGER.N_RUNS_PER_SEED
    print(f"  Master Seed {master_seed}: Running {n_runs} experiments...")

    experiment_seeds = MANAGER.generate_experiment_seeds(master_seed, env_name)

    all_rewards = []
    all_actions = []
    all_env_instances = []
    success_count = 0

    for run_id, experiment_seed in enumerate(tqdm(experiment_seeds, desc=f"Seed-{master_seed}")):
        result = run_single_experiment(
            env_class, agent_class, agent_kwargs,
            n_trials, env_name, experiment_seed
        )

        if result["success"]:
            all_rewards.append(result["rewards"])
            all_actions.append(result["actions"])
            success_count += 1

            if "env_instance" in result:
                all_env_instances.append(result["env_instance"])

    master_seed_result = {
        "master_seed": master_seed,
        "rewards": np.array(all_rewards) if success_count > 0 else np.zeros((1, n_trials)),
        "actions": np.array(all_actions) if success_count > 0 else np.zeros((1, n_trials)),
        "mean_reward": np.mean([np.mean(r) for r in all_rewards]) if success_count > 0 else 0.0,
        "std_reward": np.std([np.mean(r) for r in all_rewards]) if success_count > 0 else 0.0,
        "success_rate": success_count / len(experiment_seeds),
        "env_instances": all_env_instances,
        "n_experiments": len(experiment_seeds)
    }

    print(f"  ✅ Master Seed {master_seed}: Success rate {master_seed_result['success_rate']:.1%}, "
          f"Mean reward {master_seed_result['mean_reward']:.4f} ± {master_seed_result['std_reward']:.4f}")

    return master_seed_result


# =============================================
# ANALYSIS HELPER FUNCTIONS
# =============================================

def get_environment_change_points(env_name, env_instance=None):
    """Get change points for each environment"""
    if env_name == "EnvA":
        return [100]
    elif env_name == "EnvB":
        return [40, 80, 120, 160]
    elif env_name == "EnvC":
        if env_instance and hasattr(env_instance, 'change_points'):
            return env_instance.change_points
        else:
            return []
    else:
        return []


def get_environment_optimal_rewards(env_name, env_instance=None):
    """Get optimal rewards for each segment"""
    if env_name == "EnvA":
        return [0.8, 0.9]
    elif env_name == "EnvB":
        return [0.95, 0.95, 0.95, 0.95, 0.95]
    elif env_name == "EnvC":
        if env_instance and hasattr(env_instance, 'optimal_rewards'):
            return env_instance.optimal_rewards
        else:
            return []
    else:
        return []


def calculate_noise_adjusted_parameters(env_name):
    """Calculate parameters adjusted for noise level"""
    noise_levels = {
        "EnvA": {"sigma": 0.15, "noise_level": "medium"},
        "EnvB": {"sigma": 0.05, "noise_level": "low"},
        "EnvC": {"sigma": 0.15, "noise_level": "medium"}
    }

    noise_info = noise_levels.get(env_name, {"noise_level": "medium"})

    if noise_info["noise_level"] == "low":
        threshold_ratio = 0.90
        stability_window = 3
        min_stability_trials = 2
    elif noise_info["noise_level"] == "medium":
        threshold_ratio = 0.85
        stability_window = 5
        min_stability_trials = 3
    else:
        threshold_ratio = 0.80
        stability_window = 7
        min_stability_trials = 4

    return {
        "threshold_ratio": threshold_ratio,
        "stability_window": stability_window,
        "min_stability_trials": min_stability_trials
    }


def detect_randomshift_changes_from_rewards(rewards, window_size=20):
    """Detect change points in RandomShift environment"""
    change_points = []

    for i in range(window_size, len(rewards) - window_size):
        before_window = rewards[i-window_size:i]
        after_window = rewards[i:i+window_size]

        mean_diff = abs(np.mean(after_window) - np.mean(before_window))

        if mean_diff > 0.10:
            change_points.append(i)

    filtered_points = []
    for point in change_points:
        if not filtered_points or point - filtered_points[-1] > 25:
            filtered_points.append(point)

    return filtered_points


def compute_unified_recovery_rate(rewards, env_name, env_instances=None, analysis_window=30):
    """Compute recovery rate after environment changes"""
    env_params = calculate_noise_adjusted_parameters(env_name)
    all_recovery_rates = []

    for run_idx in range(rewards.shape[0]):
        run_rewards = rewards[run_idx]

        if env_name == "RandomShift":
            change_points = detect_randomshift_changes_from_rewards(run_rewards)
            if not change_points:
                continue
            optimal_rewards = [0.8] * len(change_points)
        else:
            if env_name == "EnvC":
                if env_instances is not None and len(env_instances) > run_idx:
                    actual_env = env_instances[run_idx]
                    change_points = actual_env.change_points if hasattr(actual_env, 'change_points') else get_environment_change_points(env_name)
                    optimal_rewards = actual_env.optimal_rewards if hasattr(actual_env, 'optimal_rewards') else get_environment_optimal_rewards(env_name)
                else:
                    change_points = get_environment_change_points(env_name)
                    optimal_rewards = get_environment_optimal_rewards(env_name)
            else:
                change_points = get_environment_change_points(env_name)
                optimal_rewards = get_environment_optimal_rewards(env_name)

        run_recovery_rates = []

        for change_idx, change_point in enumerate(change_points):
            if change_point >= len(run_rewards) - analysis_window:
                continue

            post_start = change_point
            post_end = min(change_point + analysis_window, len(run_rewards))

            if post_end <= post_start:
                continue

            segment_optimal = optimal_rewards[min(change_idx, len(optimal_rewards) - 1)]
            post_change_performance = np.mean(run_rewards[post_start:post_end])
            recovery_rate = post_change_performance / segment_optimal

            run_recovery_rates.append(recovery_rate)

        if run_recovery_rates:
            avg_recovery_rate = np.mean(run_recovery_rates)
            all_recovery_rates.append(avg_recovery_rate)

    return np.array(all_recovery_rates)


def measure_unified_recovery_time(rewards, env_name, env_instances=None, analysis_window=50):
    """Measure recovery time after environment changes"""
    env_params = calculate_noise_adjusted_parameters(env_name)
    all_recovery_times = []

    for run_idx in range(rewards.shape[0]):
        run_rewards = rewards[run_idx]

        if env_name == "RandomShift":
            change_points = detect_randomshift_changes_from_rewards(run_rewards)
            if not change_points:
                continue
            optimal_rewards = [0.8] * len(change_points)
        else:
            if env_name == "EnvC":
                if env_instances is not None and len(env_instances) > run_idx:
                    actual_env = env_instances[run_idx]
                    change_points = actual_env.change_points if hasattr(actual_env, 'change_points') else get_environment_change_points(env_name)
                    optimal_rewards = actual_env.optimal_rewards if hasattr(actual_env, 'optimal_rewards') else get_environment_optimal_rewards(env_name)
                else:
                    change_points = get_environment_change_points(env_name)
                    optimal_rewards = get_environment_optimal_rewards(env_name)
            else:
                change_points = get_environment_change_points(env_name)
                optimal_rewards = get_environment_optimal_rewards(env_name)

        run_recovery_times = []

        for change_idx, change_point in enumerate(change_points):
            if change_point >= len(run_rewards) - 10:
                continue

            segment_optimal = optimal_rewards[min(change_idx, len(optimal_rewards) - 1)]
            threshold = segment_optimal * env_params["threshold_ratio"]

            post_change_start = change_point
            post_change_end = min(change_point + analysis_window, len(run_rewards))
            post_change_rewards = run_rewards[post_change_start:post_change_end]

            recovery_time = len(post_change_rewards)
            stability_window = env_params["stability_window"]
            min_trials = env_params["min_stability_trials"]

            for i in range(min_trials, len(post_change_rewards) - stability_window + 1):
                window = post_change_rewards[i:i + stability_window]
                if np.mean(window) >= threshold:
                    recovery_time = i + stability_window // 2
                    break

            run_recovery_times.append(recovery_time)

        if run_recovery_times:
            avg_recovery_time = np.mean(run_recovery_times)
            all_recovery_times.append(avg_recovery_time)

    return np.array(all_recovery_times)


def run_experimental_replication_for_agent(env_class, agent_class, agent_kwargs,
                                          n_trials=200, env_name="Environment"):
    """Run complete experimental replication for one agent"""

    seeds = MANAGER.SEEDS

    replication_results = {}

    for master_seed in seeds:
        master_seed_result = run_master_seed_experiments(
            master_seed, env_class, agent_class, agent_kwargs, n_trials, env_name
        )
        replication_results[f"seed_{master_seed}"] = master_seed_result

    # Calculate statistics
    all_mean_rewards = []
    all_std_rewards = []
    all_success_rates = []
    all_recovery_rates = []
    all_recovery_times = []

    for seed_result in replication_results.values():
        if seed_result['success_rate'] > 0:
            rewards = seed_result['rewards']
            env_instances = seed_result.get('env_instances', None)

            all_mean_rewards.append(seed_result['mean_reward'])
            all_std_rewards.append(seed_result['std_reward'])
            all_success_rates.append(seed_result['success_rate'])

            try:
                recovery_rates = compute_unified_recovery_rate(rewards, env_name, env_instances)
                if len(recovery_rates) > 0:
                    all_recovery_rates.append(np.mean(recovery_rates))

                recovery_times = measure_unified_recovery_time(rewards, env_name, env_instances)
                if len(recovery_times) > 0:
                    all_recovery_times.append(np.mean(recovery_times))

            except Exception as e:
                print(f"    Warning: Analysis failed for master seed {seed_result['master_seed']}: {e}")

    if all_mean_rewards:
        replication_statistics = {
            "rep_mean": np.mean(all_mean_rewards),
            "rep_std": np.std(all_mean_rewards),
            "rep_sem": np.std(all_mean_rewards) / np.sqrt(len(all_mean_rewards)),
            "rep_success_rate": np.mean(all_success_rates),
            "n_master_seeds": len(all_mean_rewards),
            "total_experiments": sum(seed_result['n_experiments'] for seed_result in replication_results.values()),

            "rep_recovery_rate_mean": np.mean(all_recovery_rates) if all_recovery_rates else 0.0,
            "rep_recovery_rate_std": np.std(all_recovery_rates) if all_recovery_rates else 0.0,
            "rep_recovery_time_mean": np.mean(all_recovery_times) if all_recovery_times else 0.0,
            "rep_recovery_time_std": np.std(all_recovery_times) if all_recovery_times else 0.0
        }

        if len(all_mean_rewards) > 1:
            alpha = 0.05
            t_critical = stats.t.ppf(1 - alpha/2, len(all_mean_rewards) - 1)

            margin_error = t_critical * replication_statistics['rep_sem']
            replication_statistics['ci_lower'] = replication_statistics['rep_mean'] - margin_error
            replication_statistics['ci_upper'] = replication_statistics['rep_mean'] + margin_error

            if all_recovery_rates:
                recovery_sem = np.std(all_recovery_rates) / np.sqrt(len(all_recovery_rates))
                recovery_margin = t_critical * recovery_sem
                replication_statistics['recovery_rate_ci_lower'] = replication_statistics['rep_recovery_rate_mean'] - recovery_margin
                replication_statistics['recovery_rate_ci_upper'] = replication_statistics['rep_recovery_rate_mean'] + recovery_margin

            if all_recovery_times:
                time_sem = np.std(all_recovery_times) / np.sqrt(len(all_recovery_times))
                time_margin = t_critical * time_sem
                replication_statistics['recovery_time_ci_lower'] = replication_statistics['rep_recovery_time_mean'] - time_margin
                replication_statistics['recovery_time_ci_upper'] = replication_statistics['rep_recovery_time_mean'] + time_margin
        else:
            replication_statistics.update({
                'ci_lower': replication_statistics['rep_mean'],
                'ci_upper': replication_statistics['rep_mean'],
                'recovery_rate_ci_lower': replication_statistics['rep_recovery_rate_mean'],
                'recovery_rate_ci_upper': replication_statistics['rep_recovery_rate_mean'],
                'recovery_time_ci_lower': replication_statistics['rep_recovery_time_mean'],
                'recovery_time_ci_upper': replication_statistics['rep_recovery_time_mean']
            })
    else:
        replication_statistics = {
            "rep_mean": 0.0, "rep_std": 0.0, "rep_sem": 0.0,
            "rep_success_rate": 0.0, "n_master_seeds": 0, "total_experiments": 0,
            "ci_lower": 0.0, "ci_upper": 0.0,
            "rep_recovery_rate_mean": 0.0, "rep_recovery_rate_std": 0.0,
            "rep_recovery_time_mean": 0.0, "rep_recovery_time_std": 0.0,
            "recovery_rate_ci_lower": 0.0, "recovery_rate_ci_upper": 0.0,
            "recovery_time_ci_lower": 0.0, "recovery_time_ci_upper": 0.0
        }

    complete_result = {
        "individual_seeds": replication_results,
        "replication_statistics": replication_statistics,
        "seeds": seeds,
        "env_name": env_name,
        "comprehensive_analysis_included": True
    }

    return complete_result


# =============================================
# MAIN EXECUTION FUNCTION
# =============================================

def run_complete_experimental_replication():
    """
    Run complete experimental replication study for all agents and environments.
    12 master seeds × 300 runs = 3,600 independent trials per agent per environment.
    """

    print("=" * 80)
    print("EXPERIMENTAL REPLICATION STUDY")
    print("12 Master Seeds × 300 Runs = 3,600 Total Experiments per Agent")
    print("With Improved Baselines for Fair Comparison")
    print("=" * 80)

    optimized_params = {
        "n_actions": 5,
        "epsilon": 0.03,
        "eta": 0.55,
        "xi": 0.001,
        "memory_threshold": 0.015,
        "memory_influence": 0.3,
        "memory_similarity_threshold": 0.035,
        "top_k": 3,
        "alpha": 0.22,
        "window_size": 30,
        "memory_size": 15,
        "emotion_decay": 0.96,
        "min_eta": 0.095,
    }

    agents = {
        # === IMPROVED BASELINES (Fair comparison) ===
        "CA_EpsilonGreedy": (
            ContextAwareEpsilonGreedy,
            {
                "n_actions": 5,
                "epsilon": 0.1,
                "alpha": 0.1,
                "window_size": 50,
                "change_threshold": 0.25
            }
        ),

        "SW_UCB": (
            SlidingWindowUCB,
            {
                "n_actions": 5,
                "c": 2.0,
                "window_size": 100,
                "min_samples": 5
            }
        ),

        "Adaptive_TS": (
            AdaptiveThompsonSampling,
            {
                "n_actions": 5,
                "discount": 0.99,
                "min_std": 0.1,
                "forget_threshold": 100
            }
        ),

        # === NAIVE BASELINES (For reference) ===
        "Naive_EpsilonGreedy": (
            EpsilonGreedyAgent,
            {"n_actions": 5, "epsilon": 0.1}
        ),

        "Naive_UCB": (
            UCBAgent,
            {"n_actions": 5, "c": 0.5}
        ),

        "Naive_TS": (
            ThompsonSamplingAgent,
            {"n_actions": 5}
        ),

        # === ECIA FULL ===
        "ECIA_Full": (ECIA, optimized_params),

        # === ABLATION STUDIES ===
        "ECIA_NoEmotion": (ECIA_NoEmotion, optimized_params),
        "ECIA_NoMemory": (ECIA_NoMemory, optimized_params),
        "ECIA_NoDopamine": (ECIA_NoDopamine, optimized_params),
        "ECIA_NoDop_NoMem": (ECIA_NoDopamine_NoMemory, optimized_params),
        "ECIA_NoDop_NoEmo": (ECIA_NoDopamine_NoEmotion, optimized_params),
        "ECIA_NoMem_NoEmo": (ECIA_NoMemory_NoEmotion, optimized_params),
        "ECIA_NoAll": (ECIA_NoAll_Components, optimized_params)
    }

    environments = {
        "EnvA": EnvironmentA,
        "EnvB": EnvironmentB,
        "EnvC": EnvironmentC,
    }

    complete_replication_results = {}

    for env_name, env_class in environments.items():
        print(f"\n📊 ENVIRONMENT {env_name} Experimental Replication")
        n_trials = 200

        print(f"⚙️ Configuration: {len(MANAGER.SEEDS)} seeds × {MANAGER.N_RUNS_PER_SEED} runs = {len(MANAGER.SEEDS) * MANAGER.N_RUNS_PER_SEED} experiments")
        print(f"Expected change points: {get_environment_change_points(env_name)}")
        print("-" * 60)

        complete_replication_results[env_name] = {}

        for agent_name, (agent_class, agent_kwargs) in agents.items():
            print(f"\n🤖 Agent: {agent_name}")

            replication_result = run_experimental_replication_for_agent(
                env_class, agent_class, agent_kwargs,
                n_trials=n_trials, env_name=env_name
            )

            complete_replication_results[env_name][agent_name] = replication_result

            rep_stats = replication_result['replication_statistics']
            print(f"   📈 Replication Results:")
            print(f"    Overall Performance:")
            print(f"      Mean ± Std: {rep_stats['rep_mean']:.4f} ± {rep_stats['rep_std']:.4f}")
            print(f"      95% CI: [{rep_stats['ci_lower']:.4f}, {rep_stats['ci_upper']:.4f}]")
            print(f"    Recovery Analysis:")
            print(f"      Recovery Rate: {rep_stats['rep_recovery_rate_mean']:.4f} ± {rep_stats['rep_recovery_rate_std']:.4f}")
            print(f"      Recovery Time: {rep_stats['rep_recovery_time_mean']:.2f} ± {rep_stats['rep_recovery_time_std']:.2f} trials")
            print(f"    Experiment Details:")
            print(f"      Total experiments: {rep_stats['total_experiments']}")
            print(f"      Master seeds used: {rep_stats['n_master_seeds']}")
            print(f"      Success rate: {rep_stats['rep_success_rate']:.1%}")

    # Save results
    os.makedirs("content/Results", exist_ok=True)
    
    with open("content/Results/complete_results_improved.pkl", "wb") as f:
        pickle.dump(complete_replication_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Results saved: content/Results/complete_results_improved.pkl")

    # Save configuration
    MANAGER.save_configuration()

    print(f"\n✅ Complete Experimental Replication finished!")
    print(f"📁 Results saved in: content/Results/")

    return complete_replication_results


def bonferroni_correction(p_values, alpha=0.05):
    """Simple Bonferroni correction"""
    p_values = np.array(p_values)
    n_tests = len(p_values)

    p_corrected = p_values * n_tests
    p_corrected = np.clip(p_corrected, 0, 1)

    rejected = p_corrected < alpha

    return rejected, p_corrected


def perform_statistical_comparison(complete_results):
    """Perform statistical comparison with Bonferroni correction"""

    print("\n📊 STATISTICAL COMPARISON (Bonferroni Correction)")
    print("=" * 70)

    comparison_results = {}

    for env_name, env_results in complete_results.items():
        print(f"\n🌍 Environment: {env_name}")

        agent_data = {}

        for agent_name, agent_result in env_results.items():
            rep_stats = agent_result['replication_statistics']
            if rep_stats['n_master_seeds'] > 0:
                seed_means = []
                for seed_key, seed_result in agent_result['individual_seeds'].items():
                    if seed_result['success_rate'] > 0:
                        seed_means.append(seed_result['mean_reward'])

                if len(seed_means) >= 3:
                    agent_data[agent_name] = seed_means

        if len(agent_data) < 2:
            print(f"   ⚠️ Insufficient data for comparisons")
            continue

        pairwise_comparisons = []
        p_values = []

        agent_names = list(agent_data.keys())
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                agent1, agent2 = agent_names[i], agent_names[j]
                data1, data2 = agent_data[agent1], agent_data[agent2]

                try:
                    stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1) +
                                         (len(data2) - 1) * np.var(data2)) /
                                        (len(data1) + len(data2) - 2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0

                    pairwise_comparisons.append({
                        'Agent_1': agent1,
                        'Agent_2': agent2,
                        'Mean_1': np.mean(data1),
                        'Mean_2': np.mean(data2),
                        'Mean_Diff': np.mean(data1) - np.mean(data2),
                        'P_Value_Raw': p_val,
                        'Cohens_D': cohens_d
                    })

                    p_values.append(p_val)

                except Exception as e:
                    print(f"    ⚠️ Test failed for {agent1} vs {agent2}: {e}")

        if p_values:
            rejected, p_corrected = bonferroni_correction(p_values, alpha=0.05)

            for i, result in enumerate(pairwise_comparisons):
                result['P_Value_Bonferroni'] = p_corrected[i]
                result['Significant_Bonferroni'] = rejected[i]
                result['Significant_Raw'] = result['P_Value_Raw'] < 0.05

            comparison_results[env_name] = pairwise_comparisons

            significant_after = sum(r['Significant_Bonferroni'] for r in pairwise_comparisons)
            significant_before = sum(r['Significant_Raw'] for r in pairwise_comparisons)

            print(f"   📊 Total comparisons: {len(pairwise_comparisons)}")
            print(f"   ✅ Significant before correction: {significant_before}")
            print(f"   ✅ Significant after Bonferroni: {significant_after}")

            if significant_after > 0:
                print(f"   🎯 Significant comparisons after correction:")
                for result in pairwise_comparisons:
                    if result['Significant_Bonferroni']:
                        direction = ">" if result['Mean_Diff'] > 0 else "<"
                        print(f"      {result['Agent_1']} {direction} {result['Agent_2']}: "
                              f"p_corrected={result['P_Value_Bonferroni']:.6f}, d={result['Cohens_D']:.3f}")

            # Save to CSV
            df = pd.DataFrame(pairwise_comparisons)
            csv_filename = f"content/Results/bonferroni_{env_name}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"   💾 Saved: {csv_filename}")

    return comparison_results


def save_enhanced_csv(complete_results):
    """Save enhanced CSV with all statistics"""
    
    print("\n💾 Saving enhanced CSV data...")
    
    os.makedirs("content/Results", exist_ok=True)
    
    for env_name, env_results in complete_results.items():
        agent_summary = []
        
        for agent_name, agent_result in env_results.items():
            rep_stats = agent_result['replication_statistics']
            
            if rep_stats['n_master_seeds'] > 0:
                agent_summary.append({
                    'Environment': env_name,
                    'Agent': agent_name,
                    'Agent_Type': 'IMPROVED_BASELINE' if 'CA_' in agent_name or 'SW_' in agent_name or 'Adaptive_' in agent_name
                                  else 'NAIVE_BASELINE' if 'Naive_' in agent_name
                                  else 'ECIA',
                    
                    'Mean': rep_stats['rep_mean'],
                    'Std': rep_stats['rep_std'],
                    'SEM': rep_stats['rep_sem'],
                    'CI_Lower_95': rep_stats.get('ci_lower', 0),
                    'CI_Upper_95': rep_stats.get('ci_upper', 0),
                    
                    'Recovery_Rate_Mean': rep_stats.get('rep_recovery_rate_mean', 0),
                    'Recovery_Rate_Std': rep_stats.get('rep_recovery_rate_std', 0),
                    'Recovery_Time_Mean': rep_stats.get('rep_recovery_time_mean', 0),
                    'Recovery_Time_Std': rep_stats.get('rep_recovery_time_std', 0),
                    
                    'N_Master_Seeds': rep_stats['n_master_seeds'],
                    'Total_Experiments': rep_stats['total_experiments'],
                    'Success_Rate': rep_stats['rep_success_rate']
                })
        
        df = pd.DataFrame(agent_summary)
        csv_file = f"content/Results/{env_name}_comprehensive_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"   ✅ Saved: {csv_file}")


# =============================================
# SIMPLIFIED MAIN MENU
# =============================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("ECIA EXPERIMENTAL REPLICATION SYSTEM")
    print("Improved Baselines for Fair Comparison")
    print("="*80)
    
    print("\n🎯 This will run:")
    print("   • 3 Improved Baselines (Context-Aware, Sliding Window, Adaptive)")
    print("   • 3 Naive Baselines (for comparison)")
    print("   • ECIA Full + 7 Ablation variants")
    print("   • 3,600 experiments per agent (12 seeds × 300 runs)")
    print("   • EnvA, EnvB, EnvC environments")
    print("\n⏱️ Estimated time: 5-6 hours")
    
    response = input("\n▶️ Start experimental replication? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\n🚀 Starting experimental replication...")
        
        # Run experiments
        results = run_complete_experimental_replication()
        
        # Perform statistical analysis
        print("\n📊 Performing statistical analysis...")
        comparisons = perform_statistical_comparison(results)
        
        # Save enhanced CSV
        print("\n💾 Saving comprehensive data...")
        save_enhanced_csv(results)
        
        print("\n" + "="*80)
        print("✅ EXPERIMENTAL REPLICATION COMPLETED!")
        print("="*80)
        print("\n📁 Output files:")
        print("   • complete_results_improved.pkl")
        print("   • bonferroni_EnvA.csv, bonferroni_EnvB.csv, bonferroni_EnvC.csv")
        print("   • EnvA_comprehensive_results.csv (and B, C)")
        print("   • config.txt")
        
        return results
    else:
        print("\n❌ Experimental replication cancelled.")
        return None


if __name__ == "__main__":
    main()
