//! Reinforcement-learning hyperparameter search — the `AutoRl` API
//! (roadmap v0.8; PRD §14, §27 "noisy objectives").
//!
//! Per the PRD the *adapter* owns environment creation, rollouts and evaluation
//! episodes (§14); the core engine only sees a named metric. This adapter pairs a
//! small stochastic environment ([`GridWorld`], a slippery 1-D corridor) with a
//! tabular Q-learning agent ([`QLearner`]) and searches the agent's
//! hyperparameters — learning rate, discount, exploration — to maximize episode
//! return. Support for PPO/DQN/SAC would slot in as further adapters behind the
//! same metric seam, "not core enums" (§14).
//!
//! RL returns are *noisy* (slip makes the same policy score differently each
//! rollout), so each configuration is evaluated with **replicated robust
//! aggregation** ([`automl_core::robust`]) — the mode the PRD requires for RL and
//! GAN/diffusion objectives alike. The episode and step counts are the
//! adapter-owned budgets (§14 "budgets include environment steps and episodes").

use automl_core::error::{Error, Result};
use automl_core::metrics::{Direction, NamedMetrics};
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::prelude::{replicate, Aggregator, Distribution, SearchSpace, Study, TpeSampler};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A slippery 1-D corridor. The agent starts at cell `0` and reaches the goal at
/// cell `size - 1`. Actions are `0` (left) and `1` (right); with probability
/// `slip` the intended move reverses, making returns stochastic. Each step costs
/// `-1`; reaching the goal pays `+10` and ends the episode.
#[derive(Debug, Clone, Copy)]
pub struct GridWorld {
    /// Number of cells in the corridor.
    pub size: usize,
    /// Probability that a step goes the opposite way.
    pub slip: f64,
}

impl GridWorld {
    /// A new corridor of `size` cells with the given slip probability.
    pub fn new(size: usize, slip: f64) -> Self {
        GridWorld {
            size: size.max(2),
            slip: slip.clamp(0.0, 0.9),
        }
    }

    /// Step the environment: returns `(next_state, reward, done)`.
    pub fn step<R: Rng>(&self, state: usize, action: usize, rng: &mut R) -> (usize, f64, bool) {
        let go_right = if rng.gen_bool(self.slip) {
            action == 0
        } else {
            action == 1
        };
        let next = if go_right {
            (state + 1).min(self.size - 1)
        } else {
            state.saturating_sub(1)
        };
        if next == self.size - 1 {
            (next, 10.0, true)
        } else {
            (next, -1.0, false)
        }
    }
}

/// A tabular Q-learning agent over a [`GridWorld`]'s discrete states and two
/// actions.
pub struct QLearner {
    q: Vec<[f64; 2]>,
}

impl QLearner {
    /// A zero-initialized Q-table for a `size`-cell corridor.
    pub fn new(size: usize) -> Self {
        QLearner {
            q: vec![[0.0; 2]; size],
        }
    }

    fn greedy_action(&self, state: usize) -> usize {
        if self.q[state][1] >= self.q[state][0] {
            1
        } else {
            0
        }
    }

    /// Train for `episodes` episodes of at most `max_steps` steps with
    /// epsilon-greedy exploration.
    #[allow(clippy::too_many_arguments)]
    pub fn train<R: Rng>(
        &mut self,
        env: &GridWorld,
        episodes: usize,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        max_steps: usize,
        rng: &mut R,
    ) {
        for _ in 0..episodes {
            let mut state = 0;
            for _ in 0..max_steps {
                let action = if rng.gen_bool(epsilon) {
                    rng.gen_range(0..2)
                } else {
                    self.greedy_action(state)
                };
                let (next, reward, done) = env.step(state, action, rng);
                let best_next = self.q[next][0].max(self.q[next][1]);
                let target = reward + if done { 0.0 } else { gamma * best_next };
                self.q[state][action] += alpha * (target - self.q[state][action]);
                state = next;
                if done {
                    break;
                }
            }
        }
    }

    /// Mean return of the greedy policy over `episodes` rollouts.
    pub fn greedy_return<R: Rng>(
        &self,
        env: &GridWorld,
        episodes: usize,
        max_steps: usize,
        rng: &mut R,
    ) -> f64 {
        let mut total = 0.0;
        for _ in 0..episodes {
            let mut state = 0;
            for _ in 0..max_steps {
                let (next, reward, done) = env.step(state, self.greedy_action(state), rng);
                total += reward;
                state = next;
                if done {
                    break;
                }
            }
        }
        total / episodes as f64
    }
}

/// One-call reinforcement-learning hyperparameter search (PRD §14, §20).
///
/// Searches learning rate, discount and exploration for a tabular agent on a
/// [`GridWorld`], maximizing the robust (median) mean episode return.
pub struct AutoRl {
    env: GridWorld,
    train_episodes: usize,
    eval_episodes: usize,
    max_steps: usize,
    replicates: usize,
    trials: u64,
    seed: u64,
}

impl AutoRl {
    /// A new RL search over `env` with sensible episode/step budgets.
    pub fn new(env: GridWorld) -> Self {
        AutoRl {
            env,
            train_episodes: 200,
            eval_episodes: 20,
            max_steps: 50,
            replicates: 5,
            trials: 20,
            seed: 0,
        }
    }

    /// Training episodes per replicate (an adapter-owned budget).
    pub fn train_episodes(mut self, n: usize) -> Self {
        self.train_episodes = n.max(1);
        self
    }

    /// Evaluation episodes per replicate (an adapter-owned budget).
    pub fn eval_episodes(mut self, n: usize) -> Self {
        self.eval_episodes = n.max(1);
        self
    }

    /// Maximum steps per episode (the horizon budget).
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n.max(1);
        self
    }

    /// Number of noisy replicates aggregated per configuration.
    pub fn replicates(mut self, n: usize) -> Self {
        self.replicates = n.max(1);
        self
    }

    /// Number of hyperparameter configurations to try.
    pub fn trials(mut self, t: u64) -> Self {
        self.trials = t;
        self
    }

    /// Seed for the search and environment.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the search, returning the study (maximizing robust median return).
    pub fn fit(self) -> Result<Study> {
        if self.env.size < 2 {
            return Err(Error::Objective("grid world needs at least 2 cells".into()));
        }
        let space = SearchSpace::new()
            .add("alpha", Distribution::float(0.05, 0.9))
            .add("gamma", Distribution::float(0.5, 0.999))
            .add("epsilon", Distribution::float(0.0, 0.5));

        let mut study = Study::builder(space)
            .name("auto-rl")
            .maximize("return")
            .sampler(TpeSampler::new("return", Direction::Maximize, self.seed))
            .seed(self.seed)
            .build()?;

        let (env, train_ep, eval_ep, max_steps, reps) = (
            self.env,
            self.train_episodes,
            self.eval_episodes,
            self.max_steps,
            self.replicates,
        );

        let objective = move |p: &ParamSet, sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
            let (alpha, gamma, epsilon) =
                (p.float("alpha")?, p.float("gamma")?, p.float("epsilon")?);
            // Noisy objective: each replicate trains and evaluates under its own
            // seed, and the robust median tames the slip-induced variance.
            replicate(
                Aggregator::Median,
                reps,
                1,
                "return",
                sink,
                |seed, _inner| {
                    let mut rng = ChaCha8Rng::seed_from_u64(seed);
                    let mut agent = QLearner::new(env.size);
                    agent.train(&env, train_ep, alpha, gamma, epsilon, max_steps, &mut rng);
                    let ret = agent.greedy_return(&env, eval_ep, max_steps, &mut rng);
                    Ok(NamedMetrics::single("return", ret))
                },
            )
        };
        study.optimize_n(&objective, self.trials)?;
        Ok(study)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gridworld_pays_out_at_the_goal() {
        let env = GridWorld::new(3, 0.0); // deterministic
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        // From state 1, moving right reaches the goal (state 2).
        let (next, reward, done) = env.step(1, 1, &mut rng);
        assert_eq!(next, 2);
        assert_eq!(reward, 10.0);
        assert!(done);
    }

    #[test]
    fn trained_agent_beats_random_policy() {
        let env = GridWorld::new(5, 0.1);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut agent = QLearner::new(env.size);
        agent.train(&env, 300, 0.5, 0.95, 0.2, 50, &mut rng);
        let trained = agent.greedy_return(&env, 50, 50, &mut rng);
        let random = QLearner::new(env.size).greedy_return(&env, 50, 50, &mut rng);
        assert!(
            trained > random,
            "trained {trained} should beat untrained {random}"
        );
        assert!(
            trained > 0.0,
            "trained policy should reach the goal: {trained}"
        );
    }

    #[test]
    fn auto_rl_finds_a_good_policy() {
        let env = GridWorld::new(5, 0.1);
        let study = AutoRl::new(env)
            .train_episodes(200)
            .eval_episodes(15)
            .replicates(3)
            .trials(12)
            .seed(7)
            .fit()
            .unwrap();
        let best = study.best_trial().unwrap().unwrap();
        let ret = best.final_value("return").unwrap();
        // A learned policy reaches the goal for a clearly positive return; a
        // random walk pays the step penalty for the full horizon.
        assert!(ret > 2.0, "best robust return was {ret}");
    }

    #[test]
    fn tiny_env_errors_are_clamped_not_panicked() {
        // Size is clamped to >= 2, so this constructs and runs rather than panics.
        let env = GridWorld::new(0, 2.0);
        assert_eq!(env.size, 2);
        assert!(env.slip <= 0.9);
    }
}
