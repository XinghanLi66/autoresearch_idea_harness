from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .io import iter_jsonl, short_text, stable_id, write_json, write_jsonl
from .prompts import proposal_prompt
from .runway_client import RunwayClient


class ProposalGenerator(ABC):
    generator_id: str
    model_id: str

    @abstractmethod
    def generate(self, packet: dict[str, Any]) -> str:
        raise NotImplementedError


def _ref_titles(packet: dict[str, Any], limit: int = 4) -> list[str]:
    refs = packet.get("expert_view", {}).get("references", [])
    return [r.get("title") or "Untitled reference" for r in refs[:limit]]


def _snippets(packet: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    out = []
    for ref in packet.get("expert_view", {}).get("references", []):
        for snip in ref.get("snippets") or []:
            out.append({**snip, "ref_title": ref.get("title")})
            if len(out) >= limit:
                return out
    return out


def _proposal_xml(title: str, problem: str, idea: str, plan: str, eval_plan: str, risks: str) -> str:
    return f"""<proposal>
<title>{title}</title>
<problem>{problem}</problem>
<core_idea>{idea}</core_idea>
<implementation_plan>{plan}</implementation_plan>
<evaluation_plan>{eval_plan}</evaluation_plan>
<risks_and_limitations>{risks}</risks_and_limitations>
</proposal>"""


class EvidenceSynthesisGenerator(ProposalGenerator):
    generator_id = "evidence_synthesis"
    model_id = "template:evidence_synthesis:v1"

    def generate(self, packet: dict[str, Any]) -> str:
        refs = _ref_titles(packet)
        snippets = _snippets(packet)
        domain = packet.get("paper_type", "method")
        phrase = short_text(" ".join(s.get("text", "") for s in snippets), 700)
        return _proposal_xml(
            title=f"Evidence-grounded {domain} proposal",
            problem=f"The evidence packet suggests unresolved limitations across {', '.join(refs[:3])}.",
            idea=(
                "Combine the strongest recurring mechanism in the reference evidence with a "
                "targeted failure-mode evaluation, rather than proposing a generic extension."
            ),
            plan=(
                "Extract the common pipeline from the selected references, implement a modular "
                f"variant, and use the following evidence as design constraints: {phrase}"
            ),
            eval_plan=packet.get("baseline", {}).get("description", "Compare against the strongest baseline."),
            risks=(
                "The evidence may overrepresent successful settings; validate robustness on at "
                "least one out-of-distribution or ablation-heavy setting."
            ),
        )


class AbstractAblationGenerator(ProposalGenerator):
    generator_id = "abstract_ablation"
    model_id = "template:abstract_only_ablation:v1"

    def generate(self, packet: dict[str, Any]) -> str:
        refs = packet.get("expert_view", {}).get("references", [])
        abstracts = " ".join(r.get("abstract") or "" for r in refs[:4])
        return _proposal_xml(
            title="Abstract-only proposal ablation",
            problem="The selected abstracts point to a broad open problem but provide limited implementation evidence.",
            idea="Use the high-level themes in the abstracts to propose a simple unifying method.",
            plan=f"Build around the abstract-level pattern: {short_text(abstracts, 800)}",
            eval_plan=packet.get("baseline", {}).get("description", "Compare to existing baselines."),
            risks="Because this proposal ignores TeX snippets, it may miss implementation-critical details.",
        )


class ConservativeBaselineGenerator(ProposalGenerator):
    generator_id = "conservative_baseline"
    model_id = "template:conservative_baseline:v1"

    def generate(self, packet: dict[str, Any]) -> str:
        return _proposal_xml(
            title="Conservative baseline extension",
            problem="Existing references likely leave incremental performance or reliability gaps.",
            idea="Start from the strongest reference method and add a minimal controlled modification.",
            plan=(
                "Reproduce the strongest baseline from the evidence packet, add one controlled "
                "module or training change, and keep the rest of the pipeline fixed."
            ),
            eval_plan=(
                "Use paired ablations against the reproduced baseline and report whether the "
                "single modification explains the measured improvement."
            ),
            risks="The idea is intentionally low-risk and may be too incremental for a strong research contribution.",
        )


class LocalCheckpointPlaceholderGenerator(ProposalGenerator):
    generator_id = "local_checkpoint_placeholder"
    model_id = "placeholder:local_checkpoint"

    def generate(self, packet: dict[str, Any]) -> str:
        return _proposal_xml(
            title="Local checkpoint placeholder proposal",
            problem="Placeholder for a future local model adapter under the same evidence interface.",
            idea="A trained proposal model should identify a gap from reference evidence and produce a structured idea.",
            plan=(
                "Replace this placeholder with a local checkpoint call that consumes the same "
                "expert_view payload and emits the same proposal schema."
            ),
            eval_plan=packet.get("baseline", {}).get("description", "Expert forecast against baseline."),
            risks="This placeholder is not a model output and should only be used for pipeline smoke tests.",
        )


class ClaudeGenerator(ProposalGenerator):
    generator_id = "claude"
    model_id = "claude-sonnet-4-6"

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model_id = model

    def generate(self, packet: dict[str, Any]) -> str:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. The `claude` generator calls the standard "
                "Anthropic API; create a key at https://console.anthropic.com/ and export "
                "ANTHROPIC_API_KEY (or add it to .env). ANTHROPIC_BASE_URL may optionally "
                "point at a compatible gateway."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 180.0, "max_retries": 2}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**client_kwargs)
        prompt = (
            "You are an autoresearch idea proposal model. Based on the evidence packet below, "
            "write one structured, implementable research proposal. Do not mention that you are "
            "reading JSON. Output concise XML with proposal/title/problem/core_idea/"
            "implementation_plan/evaluation_plan/risks_and_limitations.\n\n"
            + json.dumps(packet.get("expert_view", {}), ensure_ascii=False)
        )
        msg = client.messages.create(
            model=self.model_id,
            max_tokens=1800,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()


class RunwayProposalGenerator(ProposalGenerator):
    generator_id = "runway_opus47_proposal"
    model_id = "runway:unset"

    def __init__(self, cfg: dict[str, Any], generator_id: str = "runway_opus47_proposal") -> None:
        self.generator_id = generator_id
        gen_cfg = (
            cfg.get("proposal_batches", {})
            .get("runway_generators", {})
            .get(generator_id, {})
        )
        self.key_env = str(gen_cfg.get("key_env", "RUNWAY_OPUS47_API_KEY"))
        self.model_env = str(gen_cfg.get("model_env", "RUNWAY_OPUS47_MODEL"))
        self.model_id = os.environ.get(self.model_env) or str(gen_cfg.get("model", "claude-opus-4-7"))
        self.endpoint = str(gen_cfg.get("endpoint", "google_anthropic"))
        temp = gen_cfg.get("temperature", cfg.get("llm", {}).get("proposal_temperature", 0.7))
        self.temperature = None if temp is None else float(temp)
        self.max_tokens = int(gen_cfg.get("max_tokens", cfg.get("llm", {}).get("proposal_max_tokens", 2400)))
        self.client = RunwayClient(cfg, key_env=self.key_env)

    def generate(self, packet: dict[str, Any]) -> str:
        result = self.client.complete(
            endpoint=self.endpoint,
            model=self.model_id,
            messages=proposal_prompt(packet),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        if not result.text:
            raise RuntimeError(f"{self.generator_id} returned empty proposal text.")
        return result.text


REGISTRY = {
    "evidence_synthesis": EvidenceSynthesisGenerator,
    "abstract_ablation": AbstractAblationGenerator,
    "conservative_baseline": ConservativeBaselineGenerator,
    "local_checkpoint_placeholder": LocalCheckpointPlaceholderGenerator,
    "claude": ClaudeGenerator,
    "runway_opus47_proposal": RunwayProposalGenerator,
    "runway_opus48_proposal": RunwayProposalGenerator,
}


def load_packets(path: Path, limit: int | None) -> list[dict[str, Any]]:
    packets = []
    for row in iter_jsonl(path):
        packets.append(row)
        if limit is not None and len(packets) >= limit:
            break
    return packets


def _ordered_generators(cfg: dict[str, Any], names: list[str]) -> list[ProposalGenerator]:
    gens = []
    for name in names:
        if name not in REGISTRY:
            raise ValueError(f"Unknown generator: {name}")
        cls = REGISTRY[name]
        if issubclass(cls, RunwayProposalGenerator):
            gens.append(cls(cfg, generator_id=name))
        else:
            gens.append(cls())
    return gens


def generate_batches(
    cfg: dict[str, Any],
    packets_path: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    generator_names: list[str] | None = None,
) -> dict[str, Any]:
    packets_path = packets_path or Path(cfg["runs_dir"]) / "evidence_packets" / "v1" / "all.jsonl"
    output_dir = output_dir or Path(cfg["runs_dir"]) / "proposal_batches"
    names = generator_names or list(cfg.get("proposal_batches", {}).get("generators", []))
    generators = _ordered_generators(cfg, names)
    packets = load_packets(packets_path, limit)

    private_rows = []
    expert_rows = []
    for packet in packets:
        proposals = []
        for gen in generators:
            text = gen.generate(packet)
            proposal_id = stable_id("prop", packet["packet_id"], gen.generator_id, text, length=14)
            proposals.append({
                "proposal_id": proposal_id,
                "generator_id": gen.generator_id,
                "model_id": gen.model_id,
                "text": text,
            })

        order = list(range(len(proposals)))
        random.Random(packet["packet_id"]).shuffle(order)
        anonymous = [
            {
                "proposal_id": proposals[i]["proposal_id"],
                "label": chr(ord("A") + j),
                "text": proposals[i]["text"],
            }
            for j, i in enumerate(order)
        ]
        private_rows.append({
            "packet_id": packet["packet_id"],
            "proposal_count": len(proposals),
            "proposals": proposals,
            "anonymous_order": [proposals[i]["proposal_id"] for i in order],
        })
        expert_rows.append({
            "packet_id": packet["packet_id"],
            "success_definition": packet.get("success_definition"),
            "expert_view": packet.get("expert_view"),
            "proposals": anonymous,
        })

    write_jsonl(output_dir / "private.jsonl", private_rows)
    write_jsonl(output_dir / "expert.jsonl", expert_rows)
    summary = {
        "packet_count": len(private_rows),
        "generators": names,
        "proposal_count": sum(r["proposal_count"] for r in private_rows),
        "private_path": str(output_dir / "private.jsonl"),
        "expert_path": str(output_dir / "expert.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
