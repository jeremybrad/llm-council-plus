"""Roundtable Mode orchestrator - Multi-round collaborative deliberation."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DEBUG_PROMPTS_DIR
from .council import query_model
from .settings import get_settings

logger = logging.getLogger(__name__)

# Prompt template directory
PROMPTS_DIR = Path(__file__).parent / "prompts" / "roundtable"

# Debug prompts directory
DEBUG_DIR = Path(DEBUG_PROMPTS_DIR)

# Available roles
ROLES = {
    "builder": "Builder",
    "skeptic": "Skeptic",
    "historian": "Historian",
    "pragmatist": "Pragmatist",
    "stylist": "Stylist",
    "contrarian": "Contrarian",
}


@dataclass
class AgentConfig:
    """Configuration for a single agent in the council."""

    model: str
    role: str  # Key from ROLES dict
    label: str  # Display label like "Builder" or "Agent 1"
    context_capsule: dict | None = None  # JIT context injection (c010.capsule.v1)


@dataclass
class RoundResponse:
    """Response from a single agent in a round."""

    agent_label: str
    model: str
    role: str
    content: str
    error: str | None = None
    duration_ms: int | None = None


@dataclass
class RoundResult:
    """Complete result from a single round."""

    round_number: int
    round_name: str  # "opening", "critique", "revision"
    responses: list[RoundResponse] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class RoundtableRun:
    """Complete roundtable run data."""

    run_id: str
    conversation_id: str
    question: str
    context: str = ""
    constraints: str = ""
    status: str = "running"  # running, completed, aborted
    council: list[AgentConfig] = field(default_factory=list)
    rounds: list[RoundResult] = field(default_factory=list)
    moderator_summary: dict[str, Any] | None = None
    chair_final: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "question": self.question,
            "context": self.context,
            "constraints": self.constraints,
            "status": self.status,
            "council": [asdict(a) for a in self.council],
            "rounds": [
                {
                    "round_number": r.round_number,
                    "round_name": r.round_name,
                    "responses": [asdict(resp) for resp in r.responses],
                    "started_at": r.started_at,
                    "completed_at": r.completed_at,
                }
                for r in self.rounds
            ],
            "moderator_summary": self.moderator_summary,
            "chair_final": self.chair_final,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


def load_template(template_name: str) -> str:
    """Load a prompt template from file.

    Args:
        template_name: Template path relative to prompts/roundtable/
                      e.g., "global_system", "roles/builder", "rounds/r1_opening"

    Returns:
        Template content as string
    """
    template_path = PROMPTS_DIR / f"{template_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text()


def load_role_prompt(role_key: str) -> str:
    """Load role-specific prompt add-on."""
    return load_template(f"roles/{role_key}")


def build_agent_system_prompt(role_key: str, context_capsule: dict | None = None) -> str:
    """Build complete system prompt for an agent (global + role + optional context).

    Args:
        role_key: Key from ROLES dict (e.g., "builder", "skeptic")
        context_capsule: Optional JIT context from Brain on Tap (c010.capsule.v1 format)
            Expected shape: {"facts": ["fact1", "fact2", ...], ...}

    Returns:
        Complete system prompt with global + role + optional context facts
    """
    global_prompt = load_template("global_system")
    role_prompt = load_role_prompt(role_key)
    base_prompt = f"{global_prompt}\n\n---\n\n{role_prompt}"

    # Append context facts if provided
    if context_capsule and context_capsule.get("facts"):
        facts = context_capsule["facts"]
        facts_section = "\n\n---\n\n**Contextual Facts (from Brain on Tap):**\n"
        facts_section += "\n".join(f"- {fact}" for fact in facts)
        return base_prompt + facts_section

    return base_prompt


def format_round1_prompt(agent_label: str, question: str, context: str = "", constraints: str = "") -> str:
    """Format the Round 1 (opening) prompt for an agent."""
    template = load_template("rounds/r1_opening")
    return template.format(
        AGENT_LABEL=agent_label,
        QUESTION=question,
        CONTEXT=context or "(No additional context provided)",
        CONSTRAINTS=constraints or "(No specific constraints)",
    )


def format_round2_prompt(agent_label: str, other_messages: str, target_agent: str = "another agent") -> str:
    """Format the Round 2 (critique) prompt for an agent."""
    template = load_template("rounds/r2_critique")
    return template.format(AGENT_LABEL=agent_label, OTHER_MESSAGES=other_messages, TARGET_AGENT=target_agent)


def format_round3_prompt(agent_label: str, your_round1: str, other_messages: str) -> str:
    """Format the Round 3 (revision) prompt for an agent."""
    template = load_template("rounds/r3_revision")
    return template.format(AGENT_LABEL=agent_label, YOUR_ROUND1=your_round1, OTHER_MESSAGES=other_messages)


def format_moderator_prompt(
    question: str,
    constraints: str,
    council_members: list[str],
    round1_outputs: str,
    round2_outputs: str,
    round3_outputs: str,
) -> str:
    """Format the moderator synthesis prompt."""
    template = load_template("moderator")
    return template.format(
        QUESTION=question,
        CONSTRAINTS=constraints or "(No specific constraints)",
        COUNCIL_MEMBERS=", ".join(council_members),
        ROUND1_OUTPUTS=round1_outputs,
        ROUND2_OUTPUTS=round2_outputs,
        ROUND3_OUTPUTS=round3_outputs,
    )


def format_chair_prompt(
    question: str,
    constraints: str,
    output_format: str,
    moderator_summary: str,
    round1_outputs: str,
    round2_outputs: str,
    round3_outputs: str,
) -> str:
    """Format the chair synthesis prompt."""
    template = load_template("chair")
    return template.format(
        QUESTION=question,
        CONSTRAINTS=constraints or "(No specific constraints)",
        OUTPUT_FORMAT=output_format or "Markdown with clear headings",
        MODERATOR_SUMMARY=moderator_summary,
        ROUND1_OUTPUTS=round1_outputs,
        ROUND2_OUTPUTS=round2_outputs,
        ROUND3_OUTPUTS=round3_outputs,
    )


def format_responses_for_context(responses: list[RoundResponse]) -> str:
    """Format a list of responses as context for the next round."""
    formatted = []
    for resp in responses:
        if resp.error:
            formatted.append(f"### {resp.agent_label} ({resp.role})\n[Error: {resp.error}]")
        else:
            formatted.append(f"### {resp.agent_label} ({resp.role})\n{resp.content}")
    return "\n\n---\n\n".join(formatted)


def dump_prompt_to_file(
    run_id: str,
    round_name: str,
    agent_label: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timestamp: str,
) -> None:
    """Dump rendered prompts to debug file for inspection.

    Creates file at: data/debug/prompts/{run_id}/{round_name}/{agent_label}.txt

    Args:
        run_id: The roundtable run ID
        round_name: Name of the round (opening, critique, revision, moderator, chair)
        agent_label: Agent display label (Builder, Skeptic, etc.)
        model: Full model ID (e.g., ollama:llama3.2:3b)
        system_prompt: The complete system prompt
        user_prompt: The complete user prompt
        temperature: Model temperature
        timestamp: ISO timestamp of the query
    """
    try:
        # Sanitize agent label for filename
        safe_label = agent_label.replace(" ", "_").replace("/", "-")

        # Create directory structure
        prompt_dir = DEBUG_DIR / run_id / round_name
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Write prompt file
        prompt_file = prompt_dir / f"{safe_label}.txt"
        content = f"""# Prompt Debug Dump
# Generated: {timestamp}
# Model: {model}
# Temperature: {temperature}
# Round: {round_name}
# Agent: {agent_label}

================================================================================
SYSTEM PROMPT
================================================================================

{system_prompt}

================================================================================
USER PROMPT
================================================================================

{user_prompt}
"""
        prompt_file.write_text(content)
        logger.debug(f"Dumped prompt to {prompt_file}")
    except Exception as e:
        logger.warning(f"Failed to dump prompt for {agent_label}: {e}")


async def query_agent(
    agent: AgentConfig,
    user_prompt: str,
    timeout: float = 120.0,
    temperature: float = 0.5,
    debug_context: dict[str, str] | None = None,
) -> RoundResponse:
    """Query a single agent with its system prompt + user prompt.

    Args:
        agent: Agent configuration
        user_prompt: The round-specific user prompt
        timeout: Request timeout in seconds
        temperature: Model temperature
        debug_context: Optional dict with run_id and round_name for prompt dumping

    Returns:
        RoundResponse with content or error
    """
    import time

    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    system_prompt = build_agent_system_prompt(agent.role, agent.context_capsule)

    # Dump prompts if debug context provided
    if debug_context:
        dump_prompt_to_file(
            run_id=debug_context.get("run_id", "unknown"),
            round_name=debug_context.get("round_name", "unknown"),
            agent_label=agent.label,
            model=agent.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            timestamp=timestamp,
        )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    try:
        response = await query_model(agent.model, messages, timeout, temperature)
        duration_ms = int((time.time() - start_time) * 1000)

        if response.get("error"):
            return RoundResponse(
                agent_label=agent.label,
                model=agent.model,
                role=ROLES.get(agent.role, agent.role),
                content="",
                error=response.get("error_message", "Unknown error"),
                duration_ms=duration_ms,
            )

        content = response.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content else ""

        return RoundResponse(
            agent_label=agent.label,
            model=agent.model,
            role=ROLES.get(agent.role, agent.role),
            content=content,
            error=None,
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error querying agent {agent.label}: {e}")
        return RoundResponse(
            agent_label=agent.label,
            model=agent.model,
            role=ROLES.get(agent.role, agent.role),
            content="",
            error=str(e),
            duration_ms=duration_ms,
        )


async def run_round_parallel(
    agents: list[AgentConfig],
    prompts: dict[str, str],  # agent_label -> prompt
    round_number: int,
    round_name: str,
    temperature: float = 0.5,
    max_parallel: int = 2,
    request: Any = None,
    debug_context: dict[str, str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run a round with bounded parallel execution.

    Yields:
        - First: {"type": "round_start", "round_number": N, "round_name": "...", "total": N}
        - Progress: {"type": "round_progress", "round_number": N, "count": N, "total": N, "response": RoundResponse}
        - Complete: {"type": "round_complete", "round_number": N, "round_result": RoundResult}
    """
    round_result = RoundResult(
        round_number=round_number, round_name=round_name, started_at=datetime.utcnow().isoformat()
    )

    yield {"type": "round_start", "round_number": round_number, "round_name": round_name, "total": len(agents)}

    # Use semaphore for bounded concurrency
    semaphore = asyncio.Semaphore(max_parallel)

    # Build debug context for this round if debug enabled
    round_debug_ctx = None
    if debug_context:
        round_debug_ctx = {"run_id": debug_context.get("run_id", "unknown"), "round_name": round_name}

    async def query_with_semaphore(agent: AgentConfig) -> RoundResponse:
        async with semaphore:
            prompt = prompts.get(agent.label, "")
            return await query_agent(agent, prompt, temperature=temperature, debug_context=round_debug_ctx)

    # Create tasks
    tasks = [asyncio.create_task(query_with_semaphore(agent)) for agent in agents]

    # Process as they complete
    pending = set(tasks)
    completed_count = 0

    try:
        while pending:
            # Check for client disconnect
            if request and await request.is_disconnected():
                logger.info(f"Client disconnected during round {round_number}. Cancelling...")
                for t in pending:
                    t.cancel()
                raise asyncio.CancelledError("Client disconnected")

            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)

            for task in done:
                try:
                    response = await task
                    round_result.responses.append(response)
                    completed_count += 1

                    yield {
                        "type": "round_progress",
                        "round_number": round_number,
                        "count": completed_count,
                        "total": len(agents),
                        "response": asdict(response),
                    }
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in round {round_number} task: {e}")

    except asyncio.CancelledError:
        for t in tasks:
            if not t.done():
                t.cancel()
        raise

    round_result.completed_at = datetime.utcnow().isoformat()

    yield {
        "type": "round_complete",
        "round_number": round_number,
        "round_result": {
            "round_number": round_result.round_number,
            "round_name": round_result.round_name,
            "responses": [asdict(r) for r in round_result.responses],
            "started_at": round_result.started_at,
            "completed_at": round_result.completed_at,
        },
    }


async def run_roundtable(
    conversation_id: str,
    question: str,
    agents: list[AgentConfig],
    moderator_model: str,
    chair_model: str,
    context: str = "",
    constraints: str = "",
    output_format: str = "Markdown with clear headings",
    num_rounds: int = 3,
    max_parallel: int = 2,
    request: Any = None,
    role_context: dict[str, dict] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run a complete roundtable deliberation.

    Yields SSE-compatible events for real-time progress updates.

    Args:
        conversation_id: Parent conversation ID
        question: The user's question
        agents: List of agent configurations
        moderator_model: Model for moderator synthesis
        chair_model: Model for chair final synthesis
        context: Optional shared context
        constraints: Optional constraints
        output_format: Requested output format
        num_rounds: Number of deliberation rounds (default 3)
        max_parallel: Max concurrent model queries
        request: FastAPI request for disconnect checking
        role_context: Optional per-role context injection (JIT Context from Brain on Tap)
            Format: {"builder": {"facts": [...], "claims": [...]}, "skeptic": {"facts": [...]}, ...}
            Keys are role keys (builder, skeptic, historian, pragmatist, stylist, contrarian)

    Yields:
        Events: roundtable_init, round_start, round_progress, round_complete,
                moderator_start, moderator_complete, chair_start, chair_complete
    """
    run_id = str(uuid.uuid4())

    # Apply role_context to agents if provided
    if role_context:
        for agent in agents:
            if agent.role in role_context:
                agent.context_capsule = role_context[agent.role]

    run = RoundtableRun(
        run_id=run_id,
        conversation_id=conversation_id,
        question=question,
        context=context,
        constraints=constraints,
        council=list(agents),
    )

    settings = get_settings()

    # Debug context for prompt dumping (if enabled)
    debug_context = None
    if settings.roundtable_debug_prompts:
        debug_context = {"run_id": run_id}
        logger.info(f"Prompt debug enabled - writing to data/debug/prompts/{run_id}/")

    # Yield initialization
    yield {
        "type": "roundtable_init",
        "run_id": run_id,
        "total_rounds": num_rounds,
        "council_members": [{"label": a.label, "model": a.model, "role": a.role} for a in agents],
    }

    try:
        # === ROUND 1: Opening Statements ===
        round1_prompts = {
            agent.label: format_round1_prompt(
                agent_label=agent.label, question=question, context=context, constraints=constraints
            )
            for agent in agents
        }

        round1_result = None
        async for event in run_round_parallel(
            agents=agents,
            prompts=round1_prompts,
            round_number=1,
            round_name="opening",
            temperature=settings.council_temperature,
            max_parallel=max_parallel,
            request=request,
            debug_context=debug_context,
        ):
            if event["type"] == "round_complete":
                round1_result = RoundResult(
                    round_number=1,
                    round_name="opening",
                    responses=[RoundResponse(**r) for r in event["round_result"]["responses"]],
                    started_at=event["round_result"]["started_at"],
                    completed_at=event["round_result"]["completed_at"],
                )
                run.rounds.append(round1_result)
            yield event

        if num_rounds < 2:
            # Skip rounds 2-3 if only 1 round requested
            pass
        else:
            # === ROUND 2: Critique + Questions ===
            round1_context = format_responses_for_context(round1_result.responses)

            # Assign target agents for questions (rotate)
            agent_labels = [a.label for a in agents]

            round2_prompts = {}
            for i, agent in enumerate(agents):
                # Target the next agent in the list (wrap around)
                target_idx = (i + 1) % len(agents)
                target_agent = agent_labels[target_idx]
                round2_prompts[agent.label] = format_round2_prompt(
                    agent_label=agent.label, other_messages=round1_context, target_agent=target_agent
                )

            round2_result = None
            async for event in run_round_parallel(
                agents=agents,
                prompts=round2_prompts,
                round_number=2,
                round_name="critique",
                temperature=settings.stage2_temperature,  # Lower for precise critique
                max_parallel=max_parallel,
                request=request,
                debug_context=debug_context,
            ):
                if event["type"] == "round_complete":
                    round2_result = RoundResult(
                        round_number=2,
                        round_name="critique",
                        responses=[RoundResponse(**r) for r in event["round_result"]["responses"]],
                        started_at=event["round_result"]["started_at"],
                        completed_at=event["round_result"]["completed_at"],
                    )
                    run.rounds.append(round2_result)
                yield event

            if num_rounds >= 3:
                # === ROUND 3: Revision ===
                round2_context = format_responses_for_context(round2_result.responses)

                # Each agent gets their own Round 1 response + all Round 2 feedback
                round3_prompts = {}
                for agent in agents:
                    # Find this agent's Round 1 response
                    my_round1 = next(
                        (r.content for r in round1_result.responses if r.agent_label == agent.label),
                        "(Your Round 1 response was not found)",
                    )
                    round3_prompts[agent.label] = format_round3_prompt(
                        agent_label=agent.label, your_round1=my_round1, other_messages=round2_context
                    )

                round3_result = None
                async for event in run_round_parallel(
                    agents=agents,
                    prompts=round3_prompts,
                    round_number=3,
                    round_name="revision",
                    temperature=settings.council_temperature,
                    max_parallel=max_parallel,
                    request=request,
                    debug_context=debug_context,
                ):
                    if event["type"] == "round_complete":
                        round3_result = RoundResult(
                            round_number=3,
                            round_name="revision",
                            responses=[RoundResponse(**r) for r in event["round_result"]["responses"]],
                            started_at=event["round_result"]["started_at"],
                            completed_at=event["round_result"]["completed_at"],
                        )
                        run.rounds.append(round3_result)
                    yield event

        # === MODERATOR SYNTHESIS ===
        yield {"type": "moderator_start"}

        # Format all round outputs for moderator
        round1_outputs = format_responses_for_context(run.rounds[0].responses) if len(run.rounds) > 0 else ""
        round2_outputs = (
            format_responses_for_context(run.rounds[1].responses) if len(run.rounds) > 1 else "(Round 2 not executed)"
        )
        round3_outputs = (
            format_responses_for_context(run.rounds[2].responses) if len(run.rounds) > 2 else "(Round 3 not executed)"
        )

        moderator_prompt = format_moderator_prompt(
            question=question,
            constraints=constraints,
            council_members=[a.label for a in agents],
            round1_outputs=round1_outputs,
            round2_outputs=round2_outputs,
            round3_outputs=round3_outputs,
        )

        moderator_system_prompt = load_template("moderator").split("---")[0].strip()
        moderator_messages = [
            {"role": "system", "content": moderator_system_prompt},
            {"role": "user", "content": moderator_prompt},
        ]

        # Dump moderator prompt if debug enabled
        if debug_context:
            dump_prompt_to_file(
                run_id=debug_context["run_id"],
                round_name="moderator",
                agent_label="Moderator",
                model=moderator_model,
                system_prompt=moderator_system_prompt,
                user_prompt=moderator_prompt,
                temperature=settings.chairman_temperature,
                timestamp=datetime.utcnow().isoformat(),
            )

        try:
            moderator_response = await query_model(
                moderator_model, moderator_messages, timeout=180.0, temperature=settings.chairman_temperature
            )

            moderator_content = moderator_response.get("content", "")
            if not isinstance(moderator_content, str):
                moderator_content = str(moderator_content) if moderator_content else ""

            run.moderator_summary = {
                "model": moderator_model,
                "content": moderator_content,
                "error": moderator_response.get("error", False),
            }

        except Exception as e:
            logger.error(f"Moderator synthesis failed: {e}")
            run.moderator_summary = {"model": moderator_model, "content": "", "error": True, "error_message": str(e)}

        yield {"type": "moderator_complete", "moderator_summary": run.moderator_summary}

        # === CHAIR FINAL SYNTHESIS ===
        yield {"type": "chair_start"}

        chair_prompt = format_chair_prompt(
            question=question,
            constraints=constraints,
            output_format=output_format,
            moderator_summary=run.moderator_summary.get("content", ""),
            round1_outputs=round1_outputs,
            round2_outputs=round2_outputs,
            round3_outputs=round3_outputs,
        )

        chair_system_prompt = "You are the Chair. You deliver the final answer with no preamble."
        chair_messages = [{"role": "system", "content": chair_system_prompt}, {"role": "user", "content": chair_prompt}]

        # Dump chair prompt if debug enabled
        if debug_context:
            dump_prompt_to_file(
                run_id=debug_context["run_id"],
                round_name="chair",
                agent_label="Chair",
                model=chair_model,
                system_prompt=chair_system_prompt,
                user_prompt=chair_prompt,
                temperature=settings.chairman_temperature,
                timestamp=datetime.utcnow().isoformat(),
            )

        try:
            chair_response = await query_model(
                chair_model, chair_messages, timeout=180.0, temperature=settings.chairman_temperature
            )

            chair_content = chair_response.get("content", "")
            if not isinstance(chair_content, str):
                chair_content = str(chair_content) if chair_content else ""

            run.chair_final = {
                "model": chair_model,
                "content": chair_content,
                "error": chair_response.get("error", False),
            }

        except Exception as e:
            logger.error(f"Chair synthesis failed: {e}")
            run.chair_final = {"model": chair_model, "content": "", "error": True, "error_message": str(e)}

        run.status = "completed"
        run.completed_at = datetime.utcnow().isoformat()

        yield {"type": "chair_complete", "chair_final": run.chair_final, "run": run.to_dict()}

    except asyncio.CancelledError:
        run.status = "aborted"
        run.completed_at = datetime.utcnow().isoformat()
        yield {"type": "roundtable_aborted", "run": run.to_dict()}
        raise


def get_default_council(models: list[str]) -> list[AgentConfig]:
    """Create a default council from a list of models.

    Assigns roles in order: builder, skeptic, contrarian, historian, pragmatist, stylist

    Args:
        models: List of model IDs (with provider prefix)

    Returns:
        List of AgentConfig objects
    """
    default_roles = ["builder", "skeptic", "contrarian", "historian", "pragmatist", "stylist"]

    council = []
    for i, model in enumerate(models):
        role = default_roles[i % len(default_roles)]
        label = ROLES.get(role, role.title())
        council.append(AgentConfig(model=model, role=role, label=label))

    return council
