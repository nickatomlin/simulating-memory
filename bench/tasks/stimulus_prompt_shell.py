"""Shared C1 vs C2–C4 prompt shell: stimuli framing + formatting rules."""

STIMULI_INTRO_C1 = "This is the stimuli:\n"
STIMULI_INTRO_C2_PLUS = "This is the stimuli that will be presented to the human:\n"
FORMAT_RULES_INTRO_SELF = "Give your response according to the following formatting rules:\n"
FORMAT_RULES_INTRO_PREDICT = "Predict their response according to the following formatting rules:\n"

DEFAULT_SEP_BEFORE_RULES = "\n\n\n"
# Stimulus → Give/Predict with no dashed rule line (digit span tasks and factual_qa).
DIGIT_SPAN_SEP_BEFORE_RULES = "\n\n"


def wrap_stimulus_prompt(
    prefix: str,
    condition_id: str,
    stimulus: str,
    format_rules: str,
    *,
    sep_before_rules: str = DEFAULT_SEP_BEFORE_RULES,
) -> str:
    """Append condition framing, stimulus, separator, Give/Predict line, and format rules."""
    bridge = "\n\n"
    if condition_id == "C1":
        return (
            prefix
            + bridge
            + STIMULI_INTRO_C1
            + stimulus
            + sep_before_rules
            + FORMAT_RULES_INTRO_SELF
            + format_rules
        )
    return (
        prefix
        + bridge
        + STIMULI_INTRO_C2_PLUS
        + stimulus
        + sep_before_rules
        + FORMAT_RULES_INTRO_PREDICT
        + format_rules
    )
