"""Instrument ReAct agent của assignment 3.3 qua 2 hook _call_llm / _run_tool."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from react_agent.agent import AgentResult, ReActAgent

from .tracer import Tracer


class TracedReActAgent(ReActAgent):
    def __init__(self, tracer: Tracer, *args, **kw):
        super().__init__(*args, **kw)
        self.tracer = tracer

    def _call_llm(self, messages, step_index):
        with self.tracer.generation(f"react.step{step_index}", self.llm.model, self.system, messages,
                                    metadata={"step": step_index},
                                    model_parameters={"effort": getattr(self.llm, "effort", None),
                                                      "max_tokens": getattr(self.llm, "max_tokens", None)}) as g:
            resp = super()._call_llm(messages, step_index)
            g.set_usage(resp.model, resp.input_tokens, resp.output_tokens,
                        resp.cache_read_tokens, resp.cache_write_tokens)
            g.update(output=resp.text, metadata={"stop_reason": resp.stop_reason, "request_id": resp.request_id})
        return resp

    def _run_tool(self, name, args, step_index):
        with self.tracer.tool(f"tool.{name}", input=args, metadata={"step": step_index}) as t:
            out = super()._run_tool(name, args, step_index)
            t.update(output=out)
            if out.startswith("ERROR"):
                t.update(tool_error=True)
        return out

    def run(self, question: str, metadata: dict | None = None) -> AgentResult:
        with self.tracer.trace("react.run", input={"question": question}, tags=["agent"], metadata=metadata,
                               session_id=(metadata or {}).get("session")) as t:
            res = super().run(question)
            failed = res.status != "success"
            t.update(output=res.answer, query=question, failed=failed,
                     failure_reason=res.status if failed else None,
                     metadata={"steps": len(res.steps), "tools_used": res.tools_used,
                               "parse_errors": sum(1 for s in res.steps if s.error)})
        return res
