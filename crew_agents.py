"""
CrewAI agents (multi-provider) for the Hotel Booking AI Manager.
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
        "keys_url": "https://platform.openai.com/api-keys",
    },
    "Groq": {
        # Free, fast, and easy to get working — a good fallback provider.
        "env": "GROQ_API_KEY",
        "models": ["groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant",
                   "groq/openai/gpt-oss-20b"],
        "keys_url": "https://console.groq.com/keys",
    },
    "Gemini": {
        "env": "GEMINI_API_KEY",
        # Gemini 3.x family (2.5-* is being deprecated / restricted for new
        # keys). Availability still varies by account — use the "list models"
        # button to see what YOUR key can call.
        "models": ["gemini/gemini-3.5-flash-lite", "gemini/gemini-3.5-flash",
                   "gemini/gemini-3.6-flash", "gemini/gemini-2.5-flash"],
        "keys_url": "https://aistudio.google.com/app/apikey",
    },
    "Anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "models": ["anthropic/claude-3-5-haiku-20241022",
                   "anthropic/claude-3-5-sonnet-20241022"],
        "keys_url": "https://console.anthropic.com/settings/keys",
    },
}


def env_var_for(llm: str) -> str:
    """Map a model id to the API-key environment variable it needs."""
    low = llm.lower()
    if low.startswith("gemini/") or "gemini" in low:
        return "GEMINI_API_KEY"
    if low.startswith("groq/") or "groq" in low:
        return "GROQ_API_KEY"
    if low.startswith("anthropic/") or "claude" in low:
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def list_models(provider: str, api_key: str, timeout: int = 15) -> list[str]:
    """Return the model ids the given key can actually access, provider-formatted.

    Uses each provider's public "list models" endpoint so you can discover a
    working model id (resolves 404 "model not available to your account"). Raises
    with the API's message on auth/other errors so the UI can show it.
    """
    import json
    import urllib.error
    import urllib.request

    prov = provider.lower()

    def _get(url: str, headers: dict) -> dict:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"{exc.code} {exc.reason}: {body}") from None

    if prov == "gemini":
        data = _get(
            "https://generativelanguage.googleapis.com/v1beta/models"
            f"?key={api_key}&pageSize=200",
            {},
        )
        out = [
            "gemini/" + m["name"].split("/")[-1]
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        return sorted(out)

    # OpenAI-compatible listing (OpenAI, Groq).
    base = {
        "openai": "https://api.openai.com/v1/models",
        "groq": "https://api.groq.com/openai/v1/models",
    }.get(prov)
    if not base:
        raise ValueError(f"Model listing isn't supported for '{provider}'.")

    data = _get(base, {"Authorization": f"Bearer {api_key}"})
    ids = [m["id"] for m in data.get("data", [])]
    if prov == "groq":
        ids = [f"groq/{i}" for i in ids]
    return sorted(ids)


def crewai_available() -> tuple[bool, str]:
    """Return (installed?, error-message). Lets the UI degrade gracefully."""
    try:
        import crewai  # noqa: F401
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on deploy env
        return False, str(exc)


# --------------------------------------------------------------------------- #
# Lightweight engine — two sequential LLM calls via each provider's REST API,
# using only the Python standard library. No crewai / chromadb / litellm, so it
# installs and runs anywhere (this is the default the app uses).
# --------------------------------------------------------------------------- #
ANALYST_SYSTEM = (
    "You are a Senior Hotel Data Analyst. You are an expert at translating raw "
    "machine-learning probabilities into plain-English business insights. Given "
    "this week's ML occupancy/cancellation forecast, identify the key "
    "operational risks (e.g. high cancellation exposure, low occupancy) and "
    "opportunities (e.g. upsell potential, pricing headroom). Reply with a "
    "concise bullet-point briefing of 5-8 bullets, each tied to a figure from "
    "the forecast."
)
MANAGER_SYSTEM = (
    "You are the Director of Revenue Management, a ruthless optimizer. Using the "
    "analyst's briefing, produce a concrete action plan to maximize revenue this "
    "week: pricing moves (raise/hold/discount and by roughly how much), "
    "overbooking levels given the cancellation forecast, and targeted upsell "
    "pushes. Reply with a numbered action plan of 4-6 items; each item states a "
    "specific move, the trigger from the data, and the expected revenue impact."
)


def _strip_provider_prefix(model: str) -> str:
    """`gemini/gemini-3.5-flash-lite` -> `gemini-3.5-flash-lite` for REST calls."""
    for pre in ("gemini/", "groq/", "openai/", "anthropic/"):
        if model.startswith(pre):
            return model[len(pre):]
    return model


def chat_once(provider: str, model: str, api_key: str, system: str, user: str,
              temperature: float = 0.3, timeout: int = 60) -> str:
    """One chat completion via the provider's REST API (stdlib only)."""
    import json
    import urllib.error
    import urllib.request

    prov = provider.lower()
    model = _strip_provider_prefix(model)

    def _post(url: str, headers: dict, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"{exc.code} {exc.reason}: {body}") from None

    if prov == "gemini":
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={api_key}")
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature},
        }
        data = _post(url, {"Content-Type": "application/json"}, payload)
        return data["candidates"][0]["content"]["parts"][0]["text"]

    if prov == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model, "max_tokens": 1500, "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = _post(url, headers, payload)
        return data["content"][0]["text"]

    # OpenAI-compatible (OpenAI, Groq).
    base = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
    }.get(prov)
    if not base:
        raise ValueError(f"Provider '{provider}' not supported by the lite engine.")
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": model, "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = _post(base, headers, payload)
    return data["choices"][0]["message"]["content"]


def run_advisor_lite(forecast_context: str, provider: str, model: str,
                     api_key: str, temperature: float = 0.3) -> str:
    """Run the two-agent flow with direct REST calls; return a markdown report.

    Same personas and hand-off as the CrewAI crew (Analyst -> Revenue Manager),
    but with zero heavy dependencies so it deploys reliably.
    """
    if not api_key:
        raise RuntimeError("No API key provided.")

    briefing = chat_once(
        provider, model, api_key, ANALYST_SYSTEM,
        "Here is this week's ML-generated hotel booking forecast:\n\n"
        f"{forecast_context}\n\nAnalyze it.",
        temperature=temperature,
    )
    plan = chat_once(
        provider, model, api_key, MANAGER_SYSTEM,
        "Analyst briefing:\n\n" + briefing + "\n\nNow produce the action plan.",
        temperature=temperature,
    )
    return (
        "### 🧮 Analyst briefing\n\n" + briefing.strip()
        + "\n\n### 💰 Revenue action plan\n\n" + plan.strip()
    )


def make_llm(llm: str = DEFAULT_LLM, temperature: float = 0.3):
    """Wrap a model id in CrewAI's LLM object so temperature (etc.) applies.

    e.g. ``LLM(model="gemini/gemini-3.5-flash-lite", temperature=0.3)``.
    """
    from crewai import LLM
    return LLM(model=llm, temperature=temperature)


def build_agents(llm: str = DEFAULT_LLM, verbose: bool = True,
                 temperature: float = 0.3):
    """Create the two agents, both driven by the chosen LLM."""
    from crewai import Agent

    llm_obj = make_llm(llm, temperature)

    data_analyst = Agent(
        role="Senior Hotel Data Analyst",
        goal=("Analyze weekly ML occupancy forecasts and identify operational "
              "risks or opportunities."),
        backstory=("You are an expert at translating raw machine learning "
                   "probabilities into plain English business insights."),
        llm=llm_obj,
        verbose=verbose,
        allow_delegation=False,
    )

    revenue_manager = Agent(
        role="Director of Revenue Management",
        goal=("Maximize revenue by adjusting pricing and overbooking strategies "
              "based on data."),
        backstory=("You are a ruthless optimizer. You take data analyst reports "
                   "and create concrete pricing actions."),
        llm=llm_obj,
        verbose=verbose,
        allow_delegation=False,
    )

    return data_analyst, revenue_manager


def build_crew(forecast_context: str, llm: str = DEFAULT_LLM,
               verbose: bool = True, temperature: float = 0.3):
    """Assemble the sequential crew around a forecast summary."""
    from crewai import Task, Crew, Process

    data_analyst, revenue_manager = build_agents(llm, verbose, temperature)

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
                llm: str = DEFAULT_LLM, verbose: bool = True,
                env_var: str | None = None, temperature: float = 0.3) -> str:
    """Run the crew end-to-end and return the final report as text.

    ``api_key`` (if provided) is exported as the provider's key env var for
    LiteLLM. ``env_var`` overrides the auto-detected variable (useful for custom
    model ids). Raises if CrewAI isn't installed or the model call fails — the
    caller shows the error in the UI.
    """
    env = env_var or env_var_for(llm)
    if api_key:
        os.environ[env] = api_key
    if not os.environ.get(env):
        raise RuntimeError(
            f"No API key found. Set {env} (Streamlit secrets, a local .env, or "
            "the API-key field)."
        )

    crew = build_crew(forecast_context, llm=llm, verbose=verbose,
                      temperature=temperature)
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
