# Simulation Results Schema

This file describes the columns in `data/agents/simulation_results.csv`, which records the trial-by-trial logs of the "Mining in Space" bandit simulations.

## Metadata
| Column | Description |
| :--- | :--- |
| `simulation_id` | Integer ID identifying a specific simulation run within a batch. |
| `agent_name` | The descriptive name of the agent configuration (e.g., `UCB (reg=100)`). |
| `trial` | The trial number within the simulation (1-indexed). |
| `reg` | The regularization parameter used for the simulation. |
| `k` | The exploration multiplier parameter used for the simulation. |

## Context & Environment
| Column | Description |
| :--- | :--- |
| `context_mercury` | Binary signal (-1 or 1) for the Mercury context feature. |
| `context_krypton` | Binary signal (-1 or 1) for the Krypton context feature. |
| `context_nobelium` | Binary signal (-1 or 1) for the Nobelium context feature. |
| `mapping_arm_i` | The planet label (A, B, C, or D) assigned to arm index `i` for this simulation. |

## Agent Decisions & Outcomes
| Column | Description |
| :--- | :--- |
| `choice_arm_index` | The index (0-3) of the arm selected by the agent. |
| `choice_planet_label` | The planet label (A-D) corresponding to the agent's choice. |
| `is_optimal` | 1 if the selected arm was the one with the highest expected reward, 0 otherwise. |
| `exploration` | 1 if the choice was an exploration step, 0 if exploitation. |
| `intent` | The strategic intent of the agent: `explore` or `exploit`. |
| `intent_reason` | A brief technical reason for the chosen intent. |
| `explanation` | A human-readable narrative explaining the choice based on intent and context. |
| `reward_received` | The integer reward obtained from the selected arm (rounded in-game). |

## Agent Internal State (Weights)
Each arm `i` (0-3) has a set of learned weights based on the linear regression model:
| Column | Description |
| :--- | :--- |
| `w_intercept_arm_i` | The learned intercept (bias) for arm `i`. |
| `w_mercury_arm_i` | The learned weight for the Mercury feature for arm `i`. |
| `w_krypton_arm_i` | The learned weight for the Krypton feature for arm `i`. |
| `w_nobelium_arm_i` | The learned weight for the Nobelium feature for arm `i`. |

## Agent Feature Uncertainty (Per-Feature)
Per arm `i` (0-3), the diagonal contribution to uncertainty for each feature in the current context, computed as `(x_i ** 2) * A_inv[i,i]` before the update for that trial:
| Column | Description |
| :--- | :--- |
| `feature_uncertainty_mercury_arm_i` | The Mercury feature's uncertainty contribution for arm `i`. |
| `feature_uncertainty_krypton_arm_i` | The Krypton feature's uncertainty contribution for arm `i`. |
| `feature_uncertainty_nobelium_arm_i` | The Nobelium feature's uncertainty contribution for arm `i`. |

## Agent Beliefs (Predictions)
The agent's estimate of the reward for each arm `i` (0-3) before taking an action:
| Column | Description |
| :--- | :--- |
| `agent_mu_arm_i` | The predicted mean reward for arm `i` given the current context. |
| `agent_sigma_arm_i` | The agent's uncertainty (standard deviation) for arm `i`'s prediction. |
