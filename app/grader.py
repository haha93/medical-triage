"""Triage grader for the OpenEnv Medical Triage environment."""

from app.models import Action, GroundTruth, Reward


class TriageGrader:
    URGENCY_WEIGHT: float = 0.4
    DEPARTMENT_WEIGHT: float = 0.3
    ACTIONS_WEIGHT: float = 0.2
    SAFETY_PENALTY_VALUE: float = 0.1

    def grade(self, action: Action, ground_truth: GroundTruth) -> Reward:
        """Compute multi-dimensional reward for a triage action."""
        urgency = self._urgency_score(action.urgency_level, ground_truth.urgency_level)
        department = self._department_score(action.department, ground_truth.department)
        actions = self._actions_score(action.initial_actions, ground_truth.initial_actions)
        safety = self._safety_penalty(action.urgency_level, ground_truth)

        total = (
            urgency * self.URGENCY_WEIGHT
            + department * self.DEPARTMENT_WEIGHT
            + actions * self.ACTIONS_WEIGHT
            + safety
        )
        total = max(0.0, min(1.0, total))

        return Reward(
            total=total,
            urgency_score=urgency,
            department_score=department,
            actions_score=actions,
            safety_penalty=safety,
        )

    def _urgency_score(self, predicted: int, actual: int) -> float:
        """1.0 if exact match, 0.5 if off by 1, 0.0 otherwise."""
        diff = abs(predicted - actual)
        if diff == 0:
            return 1.0
        if diff == 1:
            return 0.5
        return 0.0

    def _department_score(self, predicted: str, actual: str) -> float:
        """1.0 if match, 0.0 otherwise."""
        return 1.0 if predicted == actual else 0.0

    def _actions_score(self, predicted: list[str], actual: list[str]) -> float:
        """Ratio of correct predicted actions to total ground truth actions."""
        if not actual:
            return 0.0
        return len(set(predicted) & set(actual)) / len(actual)

    def _safety_penalty(self, predicted_urgency: int, ground_truth: GroundTruth) -> float:
        """Returns -0.1 if critical patient is under-triaged to level 4 or 5."""
        if ground_truth.is_critical and predicted_urgency in (4, 5):
            return -self.SAFETY_PENALTY_VALUE
        return 0.0
