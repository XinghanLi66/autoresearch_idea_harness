from __future__ import annotations

import json
from typing import Any


def proposal_prompt(packet: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are an autoresearch idea proposal model. Your job is to propose one "
        "novel, implementable research idea from the supplied evidence packet. "
        "Prefer concrete mechanisms, experimental recipes, ablations, metrics, "
        "and failure modes over broad motivation."
    )
    user = (
        "Use only the evidence packet below. Write one proposal that a competent "
        "research worker could implement and evaluate. The proposal must not be "
        "a generic survey or a vague direction.\n\n"
        "Output XML exactly with these tags:\n"
        "<proposal>\n"
        "<title>...</title>\n"
        "<problem>...</problem>\n"
        "<gap>...</gap>\n"
        "<core_idea>...</core_idea>\n"
        "<implementation_plan>...</implementation_plan>\n"
        "<evaluation_plan>...</evaluation_plan>\n"
        "<expected_results>...</expected_results>\n"
        "<risks_and_limitations>...</risks_and_limitations>\n"
        "</proposal>\n\n"
        "Evidence packet JSON:\n"
        f"{json.dumps(packet.get('expert_view', {}), ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def expert_forecast_prompt(
    expert_view: dict[str, Any],
    proposal: dict[str, Any],
    success_definition: str | None,
) -> list[dict[str, str]]:
    system = (
        "You are an expert scientific prediction-market judge. You estimate the "
        "probability that a proposal would succeed if implemented by a competent "
        "research worker. Be calibrated and skeptical."
    )
    user_payload = {
        "success_definition": success_definition,
        "evidence_packet": expert_view,
        "anonymous_proposal": {
            "proposal_id": proposal.get("proposal_id"),
            "label": proposal.get("label"),
            "text": proposal.get("text"),
        },
    }
    user = (
        "Judge the anonymous proposal against the evidence and baseline. "
        "Return only compact valid JSON with these fields:\n"
        "{\n"
        '  "success_probability": 0.0,\n'
        '  "rationale": "one concise paragraph, <=80 words",\n'
        '  "strengths": ["short point 1", "short point 2"],\n'
        '  "weaknesses": ["short point 1", "short point 2"],\n'
        '  "baseline_risk": "one sentence"\n'
        "}\n\n"
        "Use a probability between 0 and 1. Keep arrays to at most two short "
        "strings each. Do not include markdown fences, extra commentary, or model "
        "identity guesses.\n\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def end_to_end_master_prompt(task_packet: dict[str, Any], proposal_index: int) -> list[dict[str, str]]:
    system = (
        "You are the master research ideation model in an autoresearch pipeline. "
        "Generate one concrete, implementable proposal for the benchmark task. "
        "Optimize for ideas a worker can implement exactly and evaluate with the "
        "provided benchmark, not for broad literature commentary."
    )
    user = (
        f"Generate proposal #{proposal_index + 1}. Use only this task packet and "
        "frontline evidence. Avoid proposing a known baseline unless it is combined "
        "with a clearly new, testable mechanism.\n\n"
        "Output XML exactly with these tags:\n"
        "<proposal>\n"
        "<title>...</title>\n"
        "<problem>...</problem>\n"
        "<gap>...</gap>\n"
        "<core_idea>...</core_idea>\n"
        "<implementation_plan>...</implementation_plan>\n"
        "<evaluation_plan>...</evaluation_plan>\n"
        "<expected_results>...</expected_results>\n"
        "<risks_and_limitations>...</risks_and_limitations>\n"
        "</proposal>\n\n"
        "Task packet JSON:\n"
        f"{json.dumps(task_packet, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def end_to_end_forecast_prompt(
    task_packet: dict[str, Any],
    proposal: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are an expert prediction-market judge. Your forecast will be scored "
        "after the worker implements the proposal, using Brier and log score. Be "
        "calibrated: overconfident wrong forecasts reduce your future market weight."
    )
    is_empty = bool(proposal.get("empty_control"))
    payload = {
        "success_definition": (
            "Probability that a competent worker implementing exactly this proposal "
            "will produce a signed benchmark result that passes the task threshold."
            if not is_empty else
            "Control arm: no proposal is supplied. Forecast whether a competent worker, "
            "using only the task context, will produce a signed benchmark result that "
            "passes the task threshold."
        ),
        "task_packet": task_packet,
        "anonymous_proposal": {
            "proposal_id": proposal.get("proposal_id"),
            "label": proposal.get("label"),
            "empty_control": is_empty,
            "text": proposal.get("text") if not is_empty else "(empty worker-only control)",
        },
    }
    user = (
        "Return only compact valid JSON with these fields:\n"
        "{\n"
        '  "success_probability": 0.0,\n'
        '  "rationale": "one concise paragraph, <=80 words",\n'
        '  "feasibility": 0.0,\n'
        '  "novelty_over_worker_best": 0.0,\n'
        '  "expected_delta": 0.0,\n'
        '  "risk": 0.0,\n'
        '  "confidence": 0.0,\n'
        '  "strengths": ["short point 1", "short point 2"],\n'
        '  "weaknesses": ["short point 1", "short point 2"],\n'
        '  "baseline_risk": "one sentence"\n'
        "}\n\n"
        "All probability-like scalar fields are between 0 and 1. expected_delta is "
        "the expected absolute metric delta in percentage points. For empty_control, "
        "set novelty_over_worker_best to 0 and judge only worker-only baseline chance. "
        "Do not include markdown fences or extra commentary.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def master_advice_prompt(
    task_packet: dict[str, Any],
    proposal_text: str,
    worker_status: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are the master model advising a worker who is implementing your "
        "proposal. Give minimal, actionable advice that stays faithful to the "
        "original proposal. Do not add unrelated improvements."
    )
    user = (
        "The worker hit an ambiguity, failure, or under-threshold result. Provide "
        "one concise implementation/debugging instruction. If the proposal is not "
        "salvageable under the task constraints, say so clearly.\n\n"
        f"Task packet:\n{json.dumps(task_packet, ensure_ascii=False, indent=2)}\n\n"
        f"Proposal:\n{proposal_text}\n\n"
        f"Worker status:\n{json.dumps(worker_status, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def worker_prompt(
    task_packet: dict[str, Any],
    proposal_text: str,
    advice: list[str] | None = None,
    *,
    free_hparams: bool = False,
) -> str:
    """Generate the worker prompt.

    Args:
        task_packet:   Task context dict (from task.task_packet()).
        proposal_text: The proposal to implement (empty = worker-only control).
        advice:        Optional master advice strings accumulated so far.
        free_hparams:  If True, use the "freer worker" variant that allows the
                       worker to choose its own training hyperparameters (learning
                       rate, batch size, number of epochs, etc.) rather than being
                       constrained to the default training configuration. Use this
                       when evaluating proposals that are themselves about novel
                       training recipes or optimisation ideas — a constrained worker
                       cannot surface the quality of such proposals.
    """
    advice_block = ""
    if advice:
        advice_block = "\n\n## Master Advice So Far\n" + "\n\n".join(
            f"Advice {i + 1}:\n{item}" for i, item in enumerate(advice)
        )
    if task_packet.get("task_type") == "mle":
        edit_rule = (
            "Work in the current directory. Modify `baseline.py` as the entry point. "
            "You may create helper files only if needed, but `baseline.py` must produce "
            "`submission.csv`."
        )
    else:
        edit_rule = (
            "Modify only `editable_region.py`. It contains the full editable function "
            "or module region. Preserve the exact public signature expected by the task."
        )
    empty_control = not proposal_text.strip()
    opening = (
        "You are a loyal research worker. No proposal is provided; implement your "
        "own strongest idea from task context for the benchmark task.\n\n"
        if empty_control else
        "You are a loyal research worker. Implement exactly the selected proposal "
        "for the benchmark task.\n\n"
    )
    first_rule = (
        "1. Because this is a worker-only control, choose one concrete idea yourself "
        "from the task context; keep it local, implementable, and benchmark-faithful.\n"
        if empty_control else
        "1. Implement only what the proposal says; do not add unrelated improvements.\n"
    )
    proposal_block = (
        "No proposal is provided; implement your own strongest idea from task context."
        if empty_control else proposal_text
    )

    # Rule 3 varies by mode: constrained (default) vs free-hparams
    if free_hparams:
        rule3 = (
            "3. You MAY adjust training hyperparameters (learning rate, batch size, "
            "number of epochs, warmup schedule, optimiser settings) if the proposal "
            "calls for it or if you judge them suboptimal for your implementation. "
            "You must NOT change the evaluation metric, dataset split, or result "
            "reporting format. The HMAC-signed result.json must still be produced.\n"
        )
    else:
        rule3 = (
            "3. Do not change the evaluation setup, dataset, metric, epochs, "
            "architecture, or result reporting.\n"
        )

    return (
        opening +
        "## Rules\n\n"
        f"{first_rule}"
        f"2. {edit_rule}\n"
        f"{rule3}"
        "4. Run `bash run.sh result.json` after your changes.\n"
        "   CRITICAL: call the Bash tool directly with timeout=7200000 — do NOT use "
        "run_in_background, the Task tool, or any other async mechanism. The command "
        "must block until training completes and result.json is written. If you "
        "background the command and end your session early, the training process will "
        "be killed and result.json will never be written.\n"
        "5. Do not fabricate `result.json`; a non-null metric must be produced by "
        "`run.sh` and include the HMAC signature.\n"
        "6. If you are blocked by ambiguity before coding, write `master_question.txt` "
        "with the exact question and stop. The runner may ask the master for advice.\n"
        "7. If execution fails and you cannot recover, write "
        "`{\"val_metric\": null, \"error\": \"brief reason\"}` to `result.json`.\n\n"
        "## Proposal\n\n"
        f"{proposal_block}"
        f"{advice_block}\n\n"
        "## Task Context\n\n"
        f"{task_packet.get('task_context', '')}\n\n"
        "## Your Task\n\n"
        "Implement the proposal, run `bash run.sh result.json`, and stop when "
        "`result.json` exists."
    )
