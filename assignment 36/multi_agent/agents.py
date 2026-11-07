"""3 agent chuyên môn. Agent nào cũng có cùng một giao diện kiểu A2A:

    card                      -> AgentCard (tên + skill)
    handle(message) -> task   nhận A2A Message, trả về Task (status + artifacts)

Agent KHÔNG biết đến nhau, cũng không biết LangGraph. Nó chỉ nhận message và trả task,
nên có thể tách ra thành service HTTP riêng mà không phải sửa logic.
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
from react_agent.tools import search_web

from . import a2a
from .llm import Claude

# --------------------------------------------------------------------------- #
# JSON Schemas (structured outputs: mọi object phải có additionalProperties=false)
# --------------------------------------------------------------------------- #
RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {"claim": {"type": "string"}, "source_ids": {"type": "array", "items": {"type": "integer"}}},
            "required": ["claim", "source_ids"], "additionalProperties": False}},
        "sources": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "title": {"type": "string"}, "url": {"type": "string"}},
            "required": ["id", "title", "url"], "additionalProperties": False}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "findings", "sources", "open_questions"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "strengths": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["major", "minor"]},
                "description": {"type": "string"},
                "suggestion": {"type": "string"},
            },
            "required": ["severity", "description", "suggestion"], "additionalProperties": False}},
        "summary": {"type": "string"},
    },
    "required": ["strengths", "issues", "summary"],
    "additionalProperties": False,
}

SEARCH_TOOL = {
    "name": "search_web",
    "description": "Search the web and return titles, snippets and URLs.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


class BaseAgent:
    card: a2a.AgentCard

    def __init__(self, llm: Claude, log=print):
        self.llm = llm
        self.log = log

    def handle(self, message: dict) -> dict:
        task = a2a.make_task(message)
        a2a.set_status(task, "working")
        skill = message["metadata"]["skill"]
        try:
            if not self.card.has_skill(skill):
                raise ValueError(f"{self.card.name} không có skill '{skill}'")
            artifacts, note = self.run(skill, a2a.get_text(message), a2a.get_data(message))
            for name, parts in artifacts.items():
                a2a.add_artifact(task, name, parts)
            reply = a2a.make_message("agent", self.card.name, message["metadata"]["from"],
                                     [a2a.text_part(note)], message["contextId"], task["id"], skill)
            a2a.set_status(task, "completed", reply)
        except Exception as e:
            reply = a2a.make_message("agent", self.card.name, message["metadata"]["from"],
                                     [a2a.text_part(f"{type(e).__name__}: {e}")], message["contextId"], task["id"], skill)
            a2a.set_status(task, "failed", reply)
        return task

    def run(self, skill: str, instruction: str, data: dict) -> tuple[dict, str]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# (1) Researcher
# --------------------------------------------------------------------------- #
class ResearcherAgent(BaseAgent):
    card = a2a.AgentCard(
        name="researcher",
        description="Researches a topic on the web and returns findings with sources.",
        skills=[a2a.AgentSkill("web_research", "Web research",
                               "Search the web, cross-check facts and return structured findings with source URLs.")],
        url="graph://researcher",
    )
    SYSTEM = """You are a meticulous research agent.
Use search_web several times with different queries (aim for 3-6 searches) to cover the topic from several angles.
Prefer recent, authoritative sources. Every finding must cite the ids of the sources that support it.
Only include sources you actually received from search_web, with their exact URLs. Do not invent facts or URLs.
When you are done searching, return the final JSON."""

    def run(self, skill, instruction, data):
        topic = data.get("topic", instruction)
        focus = data.get("focus") or []

        def execute(name, args):
            return search_web(args.get("query", "")) if name == "search_web" else f"ERROR: unknown tool {name}"

        user = f"Research topic: {topic}\n{instruction}"
        if focus:
            user += "\nPay special attention to:\n- " + "\n- ".join(focus)
        result = self.llm.tool_loop(self.SYSTEM, user, [SEARCH_TOOL], execute, schema=RESEARCH_SCHEMA,
                                    on_tool=lambda n, a, o: self.log(f"    [researcher] {n}: {a.get('query')}"))
        note = f"Research done: {len(result['findings'])} findings, {len(result['sources'])} sources."
        return {"research": [a2a.data_part(result)]}, note


# --------------------------------------------------------------------------- #
# (2) Writer
# --------------------------------------------------------------------------- #
class WriterAgent(BaseAgent):
    card = a2a.AgentCard(
        name="writer",
        description="Writes a well-structured Markdown report from research notes; revises based on feedback.",
        skills=[a2a.AgentSkill("write_report", "Write report", "Draft a report from research findings."),
                a2a.AgentSkill("revise_report", "Revise report", "Revise a draft according to reviewer feedback.")],
        url="graph://writer",
        )
    SYSTEM = """You are a professional technical writer.
Write a report in Markdown using ONLY the facts in the research notes provided.
Structure: title (#), short executive summary, 3-5 sections (##), conclusion, and a "## Sources" list.
Cite sources inline with [n], where n is the source id from the notes; every factual sentence needs a citation.
Do not add facts that are not in the notes. If notes conflict or are missing, say so.
Write in the language requested by the instruction. Return only the report."""

    def run(self, skill, instruction, data):
        research = json.dumps(data["research"], ensure_ascii=False, indent=1)
        user = f"Instruction: {instruction}\nTopic: {data['topic']}\n\n<research_notes>\n{research}\n</research_notes>"
        if skill == "revise_report":
            user += (f"\n\n<previous_draft>\n{data['draft']}\n</previous_draft>\n\n"
                     f"<review>\n{json.dumps(data['review'], ensure_ascii=False, indent=1)}\n</review>\n\n"
                     "Revise the draft to fix every issue in the review (major issues first). If an issue cannot be fixed "
                     "with the research notes, remove or soften that claim instead of keeping it. "
                     "Return the full revised report.")
        draft = self.llm.text(self.SYSTEM, user)
        return {"report": [a2a.text_part(draft)]}, f"Draft ready ({len(draft.split())} words)."


# --------------------------------------------------------------------------- #
# (3) Reviewer
# --------------------------------------------------------------------------- #
class ReviewerAgent(BaseAgent):
    card = a2a.AgentCard(
        name="reviewer",
        description="Critically reviews a report for accuracy against sources, coverage, structure and clarity.",
        skills=[a2a.AgentSkill("review_report", "Review report",
                               "List concrete major/minor issues with suggestions; approve when no major issue remains.")],
        url="graph://reviewer",
    )
    SYSTEM = """You are a strict editor and fact-checker.
Review the draft against the research notes. The notes are the ground truth: a claim is supported
when it matches a finding in the notes and cites at least one of the source ids that finding lists
(citing only some of them is fine). Problems that lie in the notes themselves (source reliability,
a finding naming an organisation different from its source title) are not the writer's fault:
report them as minor at most. Check:
1. Faithfulness: every claim is supported by the notes and cites the right source ids; no invented facts.
2. Coverage: the important findings are included; the topic is answered.
3. Structure and clarity: logical sections, concise, readable.
4. Citations: [n] markers exist and match the Sources list.
Report each problem as an issue:
- "major": a claim not in the notes, a citation to a source that does not support the claim,
  or a central finding left out.
- "minor": style, wording, structure or clarity improvements.
Do not report an issue that has already been fixed. Every issue must have a concrete, actionable suggestion."""

    def run(self, skill, instruction, data):
        user = (f"Topic: {data['topic']}\n\n<research_notes>\n{json.dumps(data['research'], ensure_ascii=False, indent=1)}"
                f"\n</research_notes>\n\n<draft>\n{data['draft']}\n</draft>")
        review = self.llm.structured(self.SYSTEM, user, REVIEW_SCHEMA)
        # Quy tắc duyệt do code quyết định: không còn lỗi major là đạt
        majors = [i for i in review["issues"] if i["severity"] == "major"]
        review["approved"] = not majors
        note = (f"{'APPROVED' if review['approved'] else 'CHANGES REQUESTED'}: "
                f"{len(majors)} major, {len(review['issues']) - len(majors)} minor issues.")
        return {"review": [a2a.data_part(review)]}, note


def build_agents(llm_factory=Claude, log=print) -> dict[str, BaseAgent]:
    """Mỗi agent có client LLM riêng để đếm token riêng (có thể dùng model khác nhau)."""
    return {cls.card.name: cls(llm_factory(), log) for cls in (ResearcherAgent, WriterAgent, ReviewerAgent)}
