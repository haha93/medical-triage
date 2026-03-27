"""Triage environment engine for the OpenEnv Medical Triage environment."""

from typing import Optional

from app.models import Action, EnvironmentState, Observation, PatientCase, Reward, StepResult
from app.grader import TriageGrader
from app.cases import get_cases_for_task


class TriageEnvironment:
    def __init__(self) -> None:
        self.grader = TriageGrader()
        self.current_task: Optional[str] = None
        self.cases: list[PatientCase] = []
        self.case_index: int = 0
        self.episode_active: bool = False
        self.last_reward: Optional[Reward] = None
        self.last_episode_score: Optional[float] = None
        self._episode_rewards: list[float] = []

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        """Load cases for the given task and return the first observation.

        Args:
            task_id: One of "easy", "medium", "hard".
            seed: Random seed for deterministic case ordering.

        Returns:
            The first patient case observation.

        Raises:
            ValueError: If task_id is not valid (propagated from get_cases_for_task).
        """
        self.cases = get_cases_for_task(task_id, seed)
        self.current_task = task_id
        self.case_index = 0
        self.episode_active = True
        self.last_reward = None
        self.last_episode_score = None
        self._episode_rewards = []
        return self.cases[0].observation

    def step(self, action: Action) -> StepResult:
        """Grade the action against the current case and advance.

        Args:
            action: The agent's triage decision.

        Returns:
            StepResult with observation, reward, done flag, and info dict.

        Raises:
            RuntimeError: If no episode is active.
        """
        if not self.episode_active:
            raise RuntimeError("No active episode. Call reset() first.")

        current_case = self.cases[self.case_index]
        reward = self.grader.grade(action, current_case.ground_truth)
        self._episode_rewards.append(reward.total)
        self.last_reward = reward

        self.case_index += 1

        if self.case_index >= len(self.cases):
            self.episode_active = False
            self.last_episode_score = sum(self._episode_rewards) / len(self._episode_rewards)
            return StepResult(
                observation=current_case.observation,
                reward=reward,
                done=True,
                info={"episode_score": self.last_episode_score},
            )

        return StepResult(
            observation=self.cases[self.case_index].observation,
            reward=reward,
            done=False,
            info={},
        )

    def state(self) -> EnvironmentState:
        """Return a snapshot of the current environment state."""
        current_case: Optional[Observation] = None
        if self.episode_active and self.cases:
            current_case = self.cases[self.case_index].observation

        return EnvironmentState(
            current_case=current_case,
            episode_active=self.episode_active,
            task_id=self.current_task,
            cases_remaining=len(self.cases) - self.case_index if self.episode_active else 0,
            last_reward=self.last_reward,
        )

    def get_episode_score(self) -> Optional[float]:
        """Return the average reward across the completed episode, or None."""
        return self.last_episode_score
