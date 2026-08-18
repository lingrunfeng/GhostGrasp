from dataclasses import dataclass

from ghost_mgg_core_py.scoring.score_terms import score_hypothesis


@dataclass(frozen=True)
class RankedHypothesis:
    hypothesis: object
    score: object
    validation_state: str
    failure_reason: str


def rank_hypotheses(hypotheses, target_mask, evidence, min_total_score=-float("inf"), weights=None):
    ranked = []
    for hypothesis in hypotheses:
        score = score_hypothesis(hypothesis, target_mask, evidence, weights=weights)
        accepted = score.total >= min_total_score
        ranked.append(
            RankedHypothesis(
                hypothesis=hypothesis,
                score=score,
                validation_state="accepted" if accepted else "rejected",
                failure_reason="" if accepted else "score_below_threshold",
            )
        )

    return sorted(ranked, key=lambda item: item.score.total, reverse=True)
