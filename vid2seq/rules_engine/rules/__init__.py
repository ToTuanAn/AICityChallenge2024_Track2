from ..registry import Registry
from .abstract_rules import AbstractRules, DummyRules
from .brightness_rules import BrightnessRules

RULE_REGISTRY = Registry("RULE")

RULE_REGISTRY.register(DummyRules)
RULE_REGISTRY.register(BrightnessRules)