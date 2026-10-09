import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std=False, adv_clip_max=None, reward_fn=None, temperature=1.0, energy_max=3.0, energy_min=-3.0):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        self.reward_fn = reward_fn
        self.adv_clip_max = adv_clip_max
        self.energy_max = energy_max
        self.energy_min = energy_min
        self.temperature = temperature

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False, hard_gating=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)
        energies = np.zeros_like(rewards) if exp else None
        energies_stds = np.zeros_like(rewards) if exp else None
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            prompt_adv = (prompt_rewards - mean) / std
            advantages[prompts == prompt] = prompt_adv
            if exp:
                k = len(prompt_rewards)
                assert k >= 2, "Group size must be at least 2 for softmax energy"
                prompt_adv_clipped = (
                    np.clip(prompt_adv, -self.adv_clip_max, self.adv_clip_max)
                    if self.adv_clip_max is not None else prompt_adv
                )
                prompt_adv_clipped = prompt_adv_clipped / self.temperature
                exp_prompt_adv = np.exp(prompt_adv_clipped - np.max(prompt_adv_clipped, axis=0, keepdims=True))
                probs = exp_prompt_adv / np.sum(exp_prompt_adv, axis=0, keepdims=True)
                raw_energies = probs - (1.0 / k)

                energy_stds = np.std(raw_energies, axis=0, keepdims=True)
                if hard_gating:
                    gating_energy_stds = np.maximum(energy_stds + 1e-4, 1e-2) 
                    energies_reranged = raw_energies / gating_energy_stds
                else:
                    energies_reranged = raw_energies * k

                energies_clipped = np.clip(energies_reranged, self.energy_min, self.energy_max)
                energies[prompts == prompt] = energies_clipped
                energies_stds[prompts == prompt] = energy_stds
        if exp:
            return advantages, energies, energies_stds
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
