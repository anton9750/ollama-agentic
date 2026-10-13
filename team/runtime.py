
"""
Runtime – multi-agent engine with tool calling, memory, and orchestration.
Powered by Ollama (local).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from team.config import Config, load_config
from team.memory import ProjectIndex, SessionMemory
from team.team import TEAM, Agent, Role, get_agent, agents_by_role
from team.tools import ToolRegistry, parse_tool_calls, strip_tool_blocks, ToolResult


# ── Ollama client ──────────────────────────────────────────────

def ollama_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.25,
    timeout: int = 300,
) -> str:
    """Send one non-streaming chat request to Ollama."""

    url = f"{host.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        # Keep the model loaded between requests to reduce reload delays.
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
        },
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return (
                (body.get("message") or {}).get("content", "").strip()
                or "[empty response]"
            )

    except urllib.error.URLError as exc:
        return (
            f"[Ollama not reachable at {host}]\n{exc}\n\n"
            "Start it with: ollama serve\n"
            "Check installed models with: ollama list"
        )

    except Exception as exc:
        return f"[Ollama error: {exc}]"


# ── Helpers ────────────────────────────────────────────────────

def _c(text: str, code: str, enabled: bool = True) -> str:
    """Simple ANSI color."""
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


# ── Runtime ────────────────────────────────────────────────────

class Runtime:
    def __init__(
        self,
        root: Optional[Path] = None,
        cfg: Optional[Config] = None,
    ):
        self.root = (root or Path.cwd()).resolve()
        self.cfg = cfg or load_config(self.root)

        self.memory_dir = self.root / self.cfg.memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = self.root / self.cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.skills_dir = self.root / "skills"
        self.workflows_dir = self.root / "workflows"

        self.session = SessionMemory(
            max_turns=self.cfg.max_session_turns,
            persist_path=self.memory_dir / "session.json",
        )

        self.index = ProjectIndex(
            self.root,
            self.memory_dir / "index.json",
        )

        self.tools = ToolRegistry(self.root, self.cfg)
        self.history_file = self.memory_dir / "history.jsonl"

    # ── Low-level agent call ───────────────────────────────────

    def _call_agent(
        self,
        agent: Agent,
        user: str,
        extra_system: str = "",
        force_tools: bool | None = None,
        max_rounds: int | None = None,
    ) -> str:
        """Call an agent, optionally running a tool-use loop."""

        use_tools = self.cfg.enable_tools and agent.use_tools

        if force_tools is not None:
            use_tools = force_tools

        system = agent.system_prompt

        if use_tools:
            system += "\n\n" + self.tools.describe()

        if extra_system:
            system += "\n\n" + extra_system

        ctx = self.session.context()

        if ctx:
            user = ctx + "\n\n### Current request\n" + user

        rounds = (
            max_rounds
            if max_rounds is not None
            else self.cfg.max_tool_rounds
        )

        final_text = ""
        messages_for_log = []

        # With tools disabled, this makes exactly one model call.
        for round_i in range(rounds if use_tools else 1):
            if self.cfg.verbose and use_tools and round_i > 0:
                print(
                    _c(
                        f"    ↻ tool round {round_i + 1}",
                        "36",
                        self.cfg.color,
                    )
                )

            reply = ollama_chat(
                self.cfg.ollama_host,
                agent.model or self.cfg.default_model,
                system,
                user,
                temperature=agent.temperature,
                timeout=self.cfg.request_timeout,
            )

            messages_for_log.append(reply)

            if not use_tools:
                final_text = reply
                break

            calls = parse_tool_calls(reply)

            if not calls:
                final_text = strip_tool_blocks(reply) or reply
                break

            # Execute tools and collect their results.
            tool_outputs = []

            for name, args in calls:
                if self.cfg.verbose:
                    preview = ", ".join(
                        f"{key}=" for key in list(args)[:3]
                    )
                    print(
                        _c(
                            f"    ⚙ {name}({preview}...)",
                            "33",
                            self.cfg.color,
                        )
                    )

                result = self.tools.execute(name, args)
                tool_outputs.append(result.as_text())

                if self.cfg.verbose and not result.success:
                    print(
                        _c(
                            f"      ⚠ {result.error[:120]}",
                            "31",
                            self.cfg.color,
                        )
                    )

            # Avoid ending with an empty reply if the final permitted
            # tool round also produced tool calls.
            if round_i == rounds - 1:
                final_text = (
                    "The tool-call limit was reached. "
                    "Here are the latest tool results:\n\n"
                    + "\n\n".join(tool_outputs)
                )
                break

            user = (
                "Tool results from your previous calls:\n\n"
                + "\n\n".join(tool_outputs)
                + "\n\nContinue. If you need more tools, emit more "
                "<tool> blocks. Otherwise give your final answer "
                "with no tool blocks."
            )

        if not final_text:
            final_text = "[No final response was produced.]"

        self.session.add(
            "assistant",
            final_text[:2000],
            agent=agent.name,
        )

        return final_text

    def _log(
        self,
        kind: str,
        prompt: str,
        result_summary: str,
    ) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "kind": kind,
            "prompt": prompt[:800],
            "result_summary": result_summary[:500],
        }

        try:
            with open(self.history_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _maybe_extract_and_save(
        self,
        text: str,
        hint: str = "solution",
    ) -> None:
        """Save a code block if automatic saving is enabled."""

        if not self.cfg.auto_save_code:
            return

        blocks = re.findall(
            r"```(?:python|py|javascript|js|ts|tsx|html|css|"
            r"json|bash|sh|go|rust|java)?\s*\n(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        if not blocks:
            return

        largest = max(blocks, key=len)

        if len(largest.strip()) < 60:
            return

        fname = "solution.py"
        lower = (hint + text[:500]).lower()

        if "fastapi" in lower or "flask" in lower:
            fname = "main.py"
        elif "test" in lower:
            fname = "test_solution.py"
        elif "playwright" in lower or "selenium" in lower:
            fname = "browser_script.py"

        path = self.output_dir / fname

        if path.exists():
            stamp = datetime.now().strftime("%H%M%S")
            path = self.output_dir / (
                f"{path.stem}_{stamp}{path.suffix}"
            )

        path.write_text(largest.strip() + "\n", encoding="utf-8")

        if self.cfg.verbose:
            print(
                _c(
                    f"\n  [auto-saved] {path.relative_to(self.root)}",
                    "32",
                    self.cfg.color,
                )
            )

    # ── Public high-level operations ────────────────────────────

    def check_ollama(self) -> str:
        """Check the Ollama connection and list available models."""

        try:
            url = f"{self.cfg.ollama_host.rstrip('/')}/api/tags"

            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            models = [
                model.get("name", "")
                for model in data.get("models", [])
            ]

            return (
                f"Ollama OK @ {self.cfg.ollama_host}\n"
                f"Models: {', '.join(models) or '(none)'}\n"
                f"Default model: {self.cfg.default_model}"
            )

        except Exception as exc:
            return (
                f"Ollama NOT reachable @ {self.cfg.ollama_host}\n"
                f"{exc}"
            )

    def opinion(self, prompt: str) -> dict[str, str]:
        """Ask relevant specialists to answer independently."""

        self.session.add("user", prompt)
        results = {}

        roles = (
            Role.ARCHITECT,
            Role.BUILDER,
            Role.RESEARCHER,
            Role.CRITIC,
            Role.SECURITY,
            Role.OPTIMIZER,
            Role.TESTER,
        )

        for agent in TEAM:
            if agent.role in roles:
                print(
                    _c(
                        f"  → {agent.name} analyzing...",
                        "36",
                        self.cfg.color,
                    )
                )

                results[agent.name] = self._call_agent(
                    agent,
                    prompt,
                    force_tools=False,
                )

        self._log("opinion", prompt, str(list(results.keys())))
        return results

    def fusion(self, prompt: str) -> str:
        """Collect specialist insights and synthesize an answer."""

        self.session.add("user", prompt)

        print(
            _c(
                "  → collecting expert insights...",
                "36",
                self.cfg.color,
            )
        )

        opinions = {}

        for agent in TEAM:
            if agent.role in (
                Role.ARCHITECT,
                Role.BUILDER,
                Role.RESEARCHER,
                Role.SECURITY,
                Role.OPTIMIZER,
            ):
                print(_c(f"    • {agent.name}", "36", self.cfg.color))

                opinions[agent.name] = self._call_agent(
                    agent,
                    prompt,
                    extra_system=(
                        "Give high-density, production-ready "
                        "specifications. Prefer concrete code and file paths."
                    ),
                    force_tools=True,
                    max_rounds=6,
                )

        fusion_agent = get_agent("fusion")

        if not fusion_agent:
            return "No fusion agent defined."

        packed = "\n\n".join(
            f"### {name}\n{text}"
            for name, text in opinions.items()
        )

        user = (
            f"Original request:\n{prompt}\n\n"
            f"Expert team insights:\n\n{packed}\n\n"
            "Synthesize the absolute best solution. "
            "Use write_file for every complete source file. "
            "Zero placeholders."
        )

        print(_c("  → fusion synthesizing...", "36", self.cfg.color))

        result = self._call_agent(
            fusion_agent,
            user,
            max_rounds=10,
        )

        self._log("fusion", prompt, result[:200])
        self._maybe_extract_and_save(result, prompt)

        return result

    def collaborate(self, prompt: str) -> str:
        """
        Full multi-agent loop:
        Architect → Researcher → Builder → Critic → Security
        → Optimizer → Tester → final polish.
        """

        self.session.add("user", prompt)
        parts: list[str] = []

        architect = get_agent("architect")
        researcher = get_agent("researcher")
        main = get_agent("main")
        critic = get_agent("critic")
        security = get_agent("security_auditor")
        optimizer = get_agent("optimizer")
        tester = get_agent("tester")
        fusion = get_agent("fusion")

        # 1. Architecture
        print(_c("  → architect planning...", "36", self.cfg.color))

        plan = self._call_agent(
            architect,
            f"Create a robust architecture plan for:\n{prompt}",
            max_rounds=4,
        )

        parts.append(f"## Architecture Plan\n{plan}")
        print(_c("\n[Plan ready]", "32", self.cfg.color))

        # 2. Research existing code
        print(
            _c(
                "  → researcher surveying workspace...",
                "36",
                self.cfg.color,
            )
        )

        research = self._call_agent(
            researcher,
            f"Survey the current workspace and advise on implementation for:\n"
            f"{prompt}\n\nArchitecture plan:\n{plan}",
            max_rounds=5,
        )

        parts.append(f"## Research Notes\n{research}")

        # 3. Primary implementation
        print(
            _c(
                "  → primary builder implementing...",
                "36",
                self.cfg.color,
            )
        )

        code = self._call_agent(
            main,
            "Implement the plan. Produce complete production-grade "
            "code using write_file.\n\n"
            f"Plan:\n{plan}\n\nResearch:\n{research}\n\n"
            f"Goal: {prompt}",
            max_rounds=12,
        )

        parts.append(f"## Implementation\n{code}")

        # 4. Critic review
        if critic:
            print(_c("  → critic reviewing...", "36", self.cfg.color))

            review = self._call_agent(
                critic,
                f"Review the implementation for goal: {prompt}\n\n"
                f"Recent builder output:\n{code}\n\n"
                "Read the actual files that were written if possible.",
                max_rounds=6,
            )

            parts.append(f"## Critique\n{review}")

            needs_fix = (
                re.search(r"\b([1-6])/10\b", review) is not None
                or any(
                    word in review.lower()
                    for word in (
                        "critical",
                        "major issue",
                        "must fix",
                        "broken",
                    )
                )
            )

            if needs_fix:
                print(
                    _c(
                        "  → builder fixing critic findings...",
                        "36",
                        self.cfg.color,
                    )
                )

                code = self._call_agent(
                    main,
                    "Address these review findings thoroughly:\n"
                    f"{review}\n\nOriginal goal: {prompt}",
                    max_rounds=8,
                )

                parts.append(f"## Post-critique fixes\n{code}")

        # 5. Security audit
        if security:
            print(
                _c(
                    "  → security auditor scanning...",
                    "36",
                    self.cfg.color,
                )
            )

            audit = self._call_agent(
                security,
                f"Security-audit the current implementation for: {prompt}\n"
                "Read the real source files with tools.",
                max_rounds=6,
            )

            parts.append(f"## Security Audit\n{audit}")

            security_issue = any(
                word in audit.lower()
                for word in (
                    "vulnerability",
                    "critical",
                    "high risk",
                    "injection",
                    "secret",
                )
            )

            if security_issue:
                print(
                    _c(
                        "  → builder applying security patches...",
                        "36",
                        self.cfg.color,
                    )
                )

                code = self._call_agent(
                    main,
                    "Apply secure coding fixes based on this audit:\n"
                    f"{audit}",
                    max_rounds=6,
                )

                parts.append(f"## Security patches\n{code}")

        # 6. Optimization review
        if optimizer:
            print(
                _c(
                    "  → optimizer reviewing performance...",
                    "36",
                    self.cfg.color,
                )
            )

            perf = self._call_agent(
                optimizer,
                f"Analyse performance of the current solution for: {prompt}",
                max_rounds=4,
            )

            parts.append(f"## Performance Notes\n{perf}")

        # 7. Tests
        if tester:
            print(_c("  → tester writing & running tests...", "36", self.cfg.color))

            tests = self._call_agent(
                tester,
                f"Write and run tests for the solution of: {prompt}\n"
                "Create test files with write_file, then execute them.",
                max_rounds=8,
            )

            parts.append(f"## Tests\n{tests}")

        # 8. Final fusion polish
        if fusion:
            print(
                _c(
                    "  → fusion final polish...",
                    "36",
                    self.cfg.color,
                )
            )

            final = self._call_agent(
                fusion,
                "Produce the final polished version of the solution for:\n"
                f"{prompt}\n\n"
                "Ensure all files are consistent and complete. Use write_file.",
                max_rounds=6,
            )

            parts.append(f"## Final Polished Solution\n{final}")
            self._maybe_extract_and_save(final, prompt)

        else:
            self._maybe_extract_and_save(code, prompt)

        self._log("collaborate", prompt, "full pipeline completed")
        return "\n\n".join(parts)

    def think(self, prompt: str) -> str:
        """Deep single-path reasoning: research, draft, critique, refine."""

        self.session.add("user", prompt)

        researcher = get_agent("researcher")
        main = get_agent("main")
        critic = get_agent("critic")

        print(_c("  → deep analysis...", "36", self.cfg.color))

        analysis = self._call_agent(
            researcher,
            f"Analyze thoroughly:\n{prompt}",
            max_rounds=5,
        )

        print(_c("  → drafting solution...", "36", self.cfg.color))

        draft = self._call_agent(
            main,
            "Draft a complete solution based on this analysis:\n\n"
            f"{analysis}\n\nRequest: {prompt}",
            max_rounds=10,
        )

        if critic:
            print(_c("  → critique & refine...", "36", self.cfg.color))

            review = self._call_agent(
                critic,
                f"Critique this draft:\n{draft}",
                max_rounds=4,
            )

            final = self._call_agent(
                main,
                "Produce the polished final version.\n\n"
                f"Draft:\n{draft}\n\nReview:\n{review}",
                max_rounds=8,
            )

        else:
            final = draft

        self._log("think", prompt, final[:200])
        self._maybe_extract_and_save(final, prompt)

        return final

    def quick(self, prompt: str) -> str:
        """Fast path: use the main agent for one ordinary chat request."""

        self.session.add("user", prompt)

        main = get_agent("main")

        if not main:
            return "Error: the main agent is not configured."

        print(
            _c(
                "  → main builder (quick mode)...",
                "36",
                self.cfg.color,
            )
        )

        result = self._call_agent(
            main,
            prompt,
            max_rounds=10,
        )

        self._log("quick", prompt, result[:200])
        self._maybe_extract_and_save(result, prompt)

        return result

    # ── Skills and workflows ────────────────────────────────────

    def list_skills(self) -> list[str]:
        if not self.skills_dir.exists():
            return []

        return sorted(
            path.name
            for path in self.skills_dir.iterdir()
            if (path / "SKILL.md").exists()
        )

    def run_skill(self, name: str, context: str = "") -> str:
        skill_path = self.skills_dir / name / "SKILL.md"

        if not skill_path.exists():
            return (
                f"Skill '{name}' not found. "
                f"Available: {', '.join(self.list_skills())}"
            )

        body = skill_path.read_text(encoding="utf-8")
        main = get_agent("main")

        prompt = f"Follow this skill exactly:\n\n{body}\n\n"

        if context:
            prompt += f"User context / task:\n{context}"

        print(
            _c(
                f"  → running skill '{name}'...",
                "36",
                self.cfg.color,
            )
        )

        return self._call_agent(
            main,
            prompt,
            max_rounds=10,
        )

    def list_workflows(self) -> list[str]:
        if not self.workflows_dir.exists():
            return []

        return sorted(
            path.stem
            for path in self.workflows_dir.glob("*.json")
        )

    def run_workflow(self, name: str, prompt: str = "") -> str:
        path = self.workflows_dir / f"{name}.json"

        if not path.exists():
            return (
                f"Workflow '{name}' not found. "
                f"Available: {', '.join(self.list_workflows())}"
            )

        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])

        out = [f"Workflow: {data.get('name', name)}"]

        if data.get("description"):
            out.append(data["description"])

        previous = prompt

        for i, step in enumerate(nodes, 1):
            agent_name = step.get("agent") or step.get("name")
            agent = get_agent(agent_name) if agent_name else None

            step_type = step.get("type", "agent")
            label = step.get("label") or agent_name or step_type

            print(
                _c(
                    f"  → step {i}: {label}",
                    "36",
                    self.cfg.color,
                )
            )

            if step_type == "agent" and agent:
                instruction = step.get(
                    "instruction",
                    "Execute the current goal.",
                )

                user = (
                    f"{instruction}\n\n"
                    f"Goal / previous output:\n{previous}"
                )

                if prompt and i == 1:
                    user = (
                        f"{instruction}\n\nUser request:\n{prompt}"
                    )

                result = self._call_agent(
                    agent,
                    user,
                    max_rounds=step.get("max_rounds", 8),
                )

                previous = result
                out.append(f"\n### Step {i}: {label}\n{result}")

            elif step_type == "skill":
                skill_name = step.get("skill", "")
                result = self.run_skill(
                    skill_name,
                    previous or prompt,
                )

                previous = result
                out.append(
                    f"\n### Step {i}: skill:{skill_name}\n{result}"
                )

            else:
                out.append(
                    f"\n### Step {i}: {label} (skipped – unknown type)"
                )

        self._log("workflow", name, prompt[:200])
        self._maybe_extract_and_save(previous, prompt or name)

        return "\n".join(out)

    # ── Memory helpers ──────────────────────────────────────────

    def memory_index(self, path: str = ".") -> str:
        """Rebuild the project memory index."""
        return self.index.rebuild()

    def memory_query(self, q: str) -> list[str]:
        """Search the project memory index."""
        return self.index.query(q)

    def clear_session(self) -> str:
        """Clear the stored conversation session."""
        self.session.clear()
        return "Session memory cleared."

    def show_team(self) -> str:
        """Display the configured agent roster."""

        lines = ["Agent Team roster:"]

        for agent in TEAM:
            tools = "tools" if agent.use_tools else "no-tools"
            access = "read-only" if agent.read_only else "writer"

            lines.append(
                f"  • {agent.name:18} "
                f"[{agent.role.value:10}] "
                f"temp={agent.temperature}  {access}  {tools}"
            )

        return "\n".join(lines)
