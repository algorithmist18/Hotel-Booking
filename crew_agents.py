"""
CrewAI agents (Gemini-powered) for the Hotel Booking AI Manager.
===============================================================

Two agents collaborate, sequentially, to turn the ML models' occupancy /
cancellation forecast into an actionable revenue plan:

1. **Senior Hotel Data Analyst** — reads the raw ML forecast numbers and writes a
   plain-English briefing of operational risks and opportunities.
2. **Director of Revenue Management** — takes that briefing and produces concrete
   pricing and overbooking actions.

The agents are provider-agnostic: pass any CrewAI/LiteLLM model id as ``llm``.
OpenAI (``gpt-4o-mini``, ``gpt-4o``) works out of the box; Gemini
(``gemini/...``) needs the ``crewai[google-genai]`` extra. The API key is never
hard-coded — it is read from the environment variable that matches the provider
(``OPENAI_API_KEY`` / ``GEMINI_API_KEY``), which the caller populates from
Streamlit secrets, a local ``.env``, or a password field.

The heavy ``crewai`` import is deferred into the functions so the rest of the
Streamlit app keeps working even when CrewAI isn't installed.
"""

from __future__ import annotations

import os

# Opt out of CrewAI/OpenTelemetry telemetry BEFORE importing crewai. Otherwise it
# blocks retrying to reach telemetry.crewai.com, which stalls runs in sandboxed
# or egress-restricted deploys (e.g. Streamlit Cloud).
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# Pick up a local .env (GEMINI_API_KEY=...) with no UI. No-op if python-dotenv
# or the file is absent.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

# Default model — OpenAI works out of the box with CrewAI (no extra install).
DEFAULT_LLM = "gpt-4o-mini"

# Selectable providers, their API-key env var, and a few model choices each.
# The model id is passed straight to CrewAI/LiteLLM.
PROVIDERS = {
    "OpenAI": {
        "env": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
    "Gemini": {
        "env": "GEMINI_API_KEY",
        # Availability varies by account; some keys can't access 2.5.
        "models": ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-flash",
                   "gemini/gemini-2.5-pro"],
    },
}


def env_var_for(llm: str) -> str:
    """Map a model id to the API-key environment variable it needs."""
    low = llm.lower()
    if low.startswith("gemini/") or "gemini" in low:
        return "GEMINI_API_KEY"
    if low.startswith("anthropic/") or "claude" in low:
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def crewai_available() -> tuple[bool, str]:
    """Return (installed?, error-message). Lets the UI degrade gracefully."""
    try:
        import crewai  # noqa: F401
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on deploy env
        return False, str(exc)


def build_agents(llm: str = DEFAULT_LLM, verbose: bool = True):
    """Create the two Gemini-powered agents."""
    from crewai import Agent

    data_analyst = Agent(
        role="Senior Hotel Data Analyst",
        goal=("Analyze weekly ML occupancy forecasts and identify operational "
              "risks or opportunities."),
        backstory=("You are an expert at translating raw machine learning "
                   "probabilities into plain English business insights."),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )

    revenue_manager = Agent(
        role="Director of Revenue Management",
        goal=("Maximize revenue by adjusting pricing and overbooking strategies "
              "based on data."),
        backstory=("You are a ruthless optimizer. You take data analyst reports "
                   "and create concrete pricing actions."),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )

    return data_analyst, revenue_manager


def build_crew(forecast_context: str, llm: str = DEFAULT_LLM,
               verbose: bool = True):
    """Assemble the sequential crew around a forecast summary."""
    from crewai import Task, Crew, Process

    data_analyst, revenue_manager = build_agents(llm, verbose)

    analysis_task = Task(
        description=(
            "Here is this week's ML-generated hotel booking forecast:\n\n"
            f"{forecast_context}\n\n"
            "Analyze it. Identify the key operational risks (e.g. high "
            "cancellation exposure, low occupancy) and opportunities (e.g. "
            "upsell potential, pricing headroom). Be specific and reference the "
            "numbers."
        ),
        expected_output=(
            "A concise bullet-point briefing (5-8 bullets) of the most important "
            "operational risks and opportunities, each tied to a figure from the "
            "forecast."
        ),
        agent=data_analyst,
    )

    strategy_task = Task(
        description=(
            "Using the analyst's briefing, create a concrete action plan to "
            "maximize revenue this week. Cover pricing moves (raise/hold/discount "
            "and by roughly how much), overbooking levels given the cancellation "
            "forecast, and any targeted upsell pushes."
        ),
        expected_output=(
            "A numbered action plan (4-6 items). Each item states a specific move, "
            "the trigger from the data, and the expected revenue impact."
        ),
        agent=revenue_manager,
        context=[analysis_task],
    )

    return Crew(
        agents=[data_analyst, revenue_manager],
        tasks=[analysis_task, strategy_task],
        process=Process.sequential,
        verbose=verbose,
    )


def run_advisor(forecast_context: str, api_key: str | None = None,
                llm: str = DEFAULT_LLM, verbose: bool = True) -> str:
    """Run the crew end-to-end and return the final report as text.

    ``api_key`` (if provided) is exported as ``GEMINI_API_KEY`` for LiteLLM.
    Raises if CrewAI isn't installed or the model call fails — the caller shows
    the error in the UI.
    """
    env = env_var_for(llm)
    if api_key:
        os.environ[env] = api_key
    if not os.environ.get(env):
        raise RuntimeError(
            f"No API key found. Set {env} (Streamlit secrets, a local .env, or "
            "the API-key field)."
        )

    crew = build_crew(forecast_context, llm=llm, verbose=verbose)
    return str(crew.kickoff())


if __name__ == "__main__":
    # Quick CLI smoke test:  GEMINI_API_KEY=... python crew_agents.py
    demo_context = (
        "Sample size: 50 upcoming bookings.\n"
        "Predicted cancellation rate: 42.5%\n"
        "Expected check-ins: 29 of 50\n"
        "High-risk bookings (>70% cancel prob): 15\n"
        "Average predicted ADR: 97.24\n"
        "Projected room revenue (ADR sum): 4,860"
    )
    print(run_advisor(demo_context, verbose=True))
