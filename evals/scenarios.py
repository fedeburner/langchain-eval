"""Scripted benchmark scenarios for evaluating memory strategies.

Each scenario tests a distinct memory competency so evaluation is explanatory,
not anecdotal. We keep scenario definitions declarative so the harness can
iterate, score, and report uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TurnKind(Enum):
    FACT = "fact"
    FILLER = "filler"
    PROBE = "probe"
    SESSION_BREAK = "session_break"


@dataclass(frozen=True)
class Turn:
    kind: TurnKind
    user_message: str
    expected: Optional[str] = None
    probe_label: Optional[str] = None
    expected_keywords: tuple[str, ...] = ()

    def is_probe(self) -> bool:
        return self.kind == TurnKind.PROBE

    def is_session_break(self) -> bool:
        return self.kind == TurnKind.SESSION_BREAK


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    competency: str
    failure_interpretation: str
    turns: list[Turn]


RECENCY_VS_DISTANCE = Scenario(
    name="recency_vs_distance",
    description="Compare short-range and long-range recall under filler noise.",
    competency="retrieval_accuracy",
    failure_interpretation="Likely insufficient long-context retention or poor salience.",
    turns=[
        Turn(TurnKind.FACT, "My name is Jordan and I am a software engineer."),
        Turn(TurnKind.FACT, "My dog's name is Max, he is a golden retriever."),
        Turn(TurnKind.FACT, "I live in Austin, Texas."),
        Turn(TurnKind.FILLER, "What is the capital of Mongolia?"),
        Turn(TurnKind.FILLER, "How do airplanes fly?"),
        Turn(TurnKind.FILLER, "Explain TCP vs UDP in simple terms."),
        Turn(TurnKind.FILLER, "Tell me a fun fact about octopuses."),
        Turn(TurnKind.FILLER, "What year was the Eiffel Tower built?"),
        Turn(
            TurnKind.PROBE,
            "What is my dog's name and breed?",
            expected="Max, golden retriever",
            probe_label="rvd_dog_recall",
            expected_keywords=("max", "golden retriever"),
        ),
        Turn(
            TurnKind.PROBE,
            "Where do I live?",
            expected="Austin, Texas",
            probe_label="rvd_location_recall",
            expected_keywords=("austin", "texas"),
        ),
        Turn(
            TurnKind.PROBE,
            "What is my profession?",
            expected="software engineer",
            probe_label="rvd_profession_recall",
            expected_keywords=("software engineer",),
        ),
    ],
)


CONFLICT_RESOLUTION = Scenario(
    name="conflict_resolution",
    description="Test whether newer contradictory facts overwrite older ones.",
    competency="conflict_resolution",
    failure_interpretation="Likely stale-memory dominance or no recency weighting.",
    turns=[
        Turn(TurnKind.FACT, "I live in Austin, Texas."),
        Turn(TurnKind.FILLER, "What are good places to visit in Japan?"),
        Turn(TurnKind.FILLER, "Give me a quick summary of quantum entanglement."),
        Turn(TurnKind.FACT, "Actually, I just moved to Denver, Colorado."),
        Turn(TurnKind.FILLER, "What are the health benefits of green tea?"),
        Turn(
            TurnKind.PROBE,
            "Where do I live now?",
            expected="Denver, Colorado",
            probe_label="conflict_current_location",
            expected_keywords=("denver", "colorado"),
        ),
        Turn(
            TurnKind.PROBE,
            "What city did I say I lived in before Denver?",
            expected="Austin",
            probe_label="conflict_old_location",
            expected_keywords=("austin",),
        ),
    ],
)


COMPOSITIONAL_RECALL = Scenario(
    name="compositional_recall",
    description="Require combining multiple facts from separated turns.",
    competency="compositional_recall",
    failure_interpretation="Memory retrieval is present but synthesis is weak.",
    turns=[
        Turn(TurnKind.FACT, "My name is Jordan."),
        Turn(TurnKind.FACT, "I lead the platform engineering team."),
        Turn(TurnKind.FILLER, "How does HTTPS work?"),
        Turn(TurnKind.FACT, "My favorite language is Rust."),
        Turn(TurnKind.FILLER, "What are the northern lights?"),
        Turn(TurnKind.FACT, "I recently started learning Go for work."),
        Turn(TurnKind.FILLER, "Recommend a book for long flights."),
        Turn(
            TurnKind.PROBE,
            "Summarize what you know about my role and programming languages.",
            expected="Platform engineering lead, Rust, Go",
            probe_label="compose_role_languages",
            expected_keywords=("platform engineering", "rust", "go"),
        ),
        Turn(
            TurnKind.PROBE,
            "What is my name and what team do I lead?",
            expected="Jordan, platform engineering team",
            probe_label="compose_identity_and_role",
            expected_keywords=("jordan", "platform engineering"),
        ),
    ],
)


NOISE_ROBUSTNESS = Scenario(
    name="noise_robustness",
    description="Evaluate whether critical facts survive heavy irrelevant chatter.",
    competency="signal_retention_under_noise",
    failure_interpretation="Fact salience tracking is weak under distraction.",
    turns=[
        Turn(TurnKind.FACT, "Remember this: my emergency contact is Alice."),
        Turn(TurnKind.FACT, "My blood type is O-negative, please remember that too."),
        Turn(TurnKind.FILLER, "What is photosynthesis?"),
        Turn(TurnKind.FILLER, "What is 19 multiplied by 23?"),
        Turn(TurnKind.FILLER, "Who painted the Mona Lisa?"),
        Turn(TurnKind.FILLER, "What are binary search trees?"),
        Turn(TurnKind.FILLER, "Explain event loops."),
        Turn(TurnKind.FILLER, "What causes rainbows?"),
        Turn(
            TurnKind.PROBE,
            "What name did I ask you to remember as my emergency contact?",
            expected="Alice",
            probe_label="noise_emergency_contact",
            expected_keywords=("alice",),
        ),
        Turn(
            TurnKind.PROBE,
            "What is my blood type?",
            expected="O-negative",
            probe_label="noise_blood_type",
            expected_keywords=("o-negative", "o negative"),
        ),
    ],
)


INSTRUCTION_PREFERENCE_PERSISTENCE = Scenario(
    name="instruction_preference_persistence",
    description="Check if user response-style preferences persist across turns.",
    competency="preference_persistence",
    failure_interpretation="Instruction memory is not retained or not prioritized.",
    turns=[
        Turn(
            TurnKind.FACT,
            "Please prefer concise bullet-point answers when you respond to me.",
        ),
        Turn(TurnKind.FILLER, "Explain what Kubernetes is."),
        Turn(TurnKind.FILLER, "What is a vector database?"),
        Turn(
            TurnKind.PROBE,
            "What response style did I ask you to use?",
            expected="concise bullet-point answers",
            probe_label="pref_style_recall",
            expected_keywords=("concise", "bullet"),
        ),
        Turn(
            TurnKind.PROBE,
            "Now explain the difference between HTTP and HTTPS in the style I requested.",
            expected="concise bullet points",
            probe_label="pref_style_application",
            expected_keywords=("http", "https"),
        ),
    ],
)


CROSS_SESSION_LONGTERM = Scenario(
    name="cross_session_longterm",
    description="Simulate session A/B boundaries and test persistence across sessions.",
    competency="cross_session_persistence",
    failure_interpretation=(
        "Facts are not persisted outside thread/session boundaries, or updates are stale."
    ),
    turns=[
        Turn(TurnKind.FACT, "My name is Jordan and I am a software engineer."),
        Turn(TurnKind.FACT, "I live in Austin, Texas."),
        Turn(TurnKind.FACT, "My dog's name is Max, he is a golden retriever."),
        Turn(TurnKind.SESSION_BREAK, "[SESSION BREAK: start a new chat thread]"),
        Turn(
            TurnKind.PROBE,
            "In this new session, what is my name?",
            expected="Jordan",
            probe_label="longterm_name_after_break",
            expected_keywords=("jordan",),
        ),
        Turn(
            TurnKind.PROBE,
            "In this new session, where do I live?",
            expected="Austin, Texas",
            probe_label="longterm_location_after_break",
            expected_keywords=("austin", "texas"),
        ),
        Turn(TurnKind.FACT, "I moved from Austin to Denver, Colorado."),
        Turn(TurnKind.SESSION_BREAK, "[SESSION BREAK: start another new chat thread]"),
        Turn(
            TurnKind.PROBE,
            "After my update, where do I live now?",
            expected="Denver, Colorado",
            probe_label="longterm_updated_location",
            expected_keywords=("denver", "colorado"),
        ),
    ],
)


DEEP_CONVERSATION = Scenario(
    name="deep_conversation",
    description="Long conversation (25+ turns) with facts planted early and probed late.",
    competency="long_context_retention",
    failure_interpretation=(
        "Strategy cannot retain fine-grained details across extended conversations."
    ),
    turns=[
        Turn(TurnKind.FACT, "My name is Jordan and I am a data engineer at Stripe."),
        Turn(TurnKind.FACT, "My wife's name is Sarah and she is a nurse."),
        Turn(TurnKind.FACT, "We have two cats named Luna and Mochi."),
        Turn(TurnKind.FILLER, "What is the difference between SQL and NoSQL?"),
        Turn(TurnKind.FILLER, "Explain how a hash map works."),
        Turn(TurnKind.FILLER, "What is the CAP theorem?"),
        Turn(TurnKind.FILLER, "Tell me about the history of the internet."),
        Turn(TurnKind.FILLER, "What is a microservice architecture?"),
        Turn(TurnKind.FILLER, "Explain the difference between REST and GraphQL."),
        Turn(TurnKind.FILLER, "What is containerization?"),
        Turn(TurnKind.FILLER, "How does DNS resolution work?"),
        Turn(TurnKind.FILLER, "What is the difference between threads and processes?"),
        Turn(TurnKind.FILLER, "Explain MapReduce in simple terms."),
        Turn(TurnKind.FILLER, "What is eventual consistency?"),
        Turn(TurnKind.FILLER, "How do load balancers work?"),
        Turn(TurnKind.FILLER, "What is the difference between latency and throughput?"),
        Turn(TurnKind.FILLER, "Explain what a CDN does."),
        Turn(TurnKind.FILLER, "What are database indexes?"),
        Turn(TurnKind.FILLER, "Explain the publish-subscribe pattern."),
        Turn(TurnKind.FILLER, "What is a message queue?"),
        Turn(TurnKind.FILLER, "How does OAuth2 work?"),
        Turn(TurnKind.FILLER, "What is rate limiting and why does it matter?"),
        Turn(TurnKind.FILLER, "Explain blue-green deployments."),
        Turn(
            TurnKind.PROBE,
            "What is my wife's name and what does she do?",
            expected="Sarah, nurse",
            probe_label="deep_spouse_recall",
            expected_keywords=("sarah", "nurse"),
        ),
        Turn(
            TurnKind.PROBE,
            "What are the names of my cats?",
            expected="Luna and Mochi",
            probe_label="deep_cats_recall",
            expected_keywords=("luna", "mochi"),
        ),
        Turn(
            TurnKind.PROBE,
            "Where do I work and what is my job title?",
            expected="data engineer at Stripe",
            probe_label="deep_job_recall",
            expected_keywords=("data engineer", "stripe"),
        ),
    ],
)


REPEATED_UPDATES = Scenario(
    name="repeated_updates",
    description="Same fact updated 3 times -- tests whether the latest value wins.",
    competency="update_tracking",
    failure_interpretation=(
        "Strategy confuses old and new values or merges them incorrectly."
    ),
    turns=[
        Turn(TurnKind.FACT, "I live in Austin, Texas."),
        Turn(TurnKind.FILLER, "What is the tallest building in the world?"),
        Turn(TurnKind.FACT, "Actually I just moved. I now live in Denver, Colorado."),
        Turn(TurnKind.FILLER, "What is the speed of light?"),
        Turn(TurnKind.FILLER, "Explain how vaccines work."),
        Turn(TurnKind.FACT, "Update again -- I relocated to Seattle, Washington last week."),
        Turn(TurnKind.FILLER, "What is the Pythagorean theorem?"),
        Turn(TurnKind.FILLER, "Tell me about the Great Wall of China."),
        Turn(
            TurnKind.PROBE,
            "Where do I currently live?",
            expected="Seattle, Washington",
            probe_label="update_current_city",
            expected_keywords=("seattle",),
        ),
        Turn(
            TurnKind.PROBE,
            "What cities have I lived in, in order?",
            expected="Austin, Denver, Seattle",
            probe_label="update_city_history",
            expected_keywords=("austin", "denver", "seattle"),
        ),
    ],
)


SCENARIOS: dict[str, Scenario] = {
    RECENCY_VS_DISTANCE.name: RECENCY_VS_DISTANCE,
    CONFLICT_RESOLUTION.name: CONFLICT_RESOLUTION,
    COMPOSITIONAL_RECALL.name: COMPOSITIONAL_RECALL,
    NOISE_ROBUSTNESS.name: NOISE_ROBUSTNESS,
    INSTRUCTION_PREFERENCE_PERSISTENCE.name: INSTRUCTION_PREFERENCE_PERSISTENCE,
    CROSS_SESSION_LONGTERM.name: CROSS_SESSION_LONGTERM,
    DEEP_CONVERSATION.name: DEEP_CONVERSATION,
    REPEATED_UPDATES.name: REPEATED_UPDATES,
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Available: {list(SCENARIOS)}")
    return SCENARIOS[name]
