from ..registry import Registry
from .abstract_rules import AbstractRules
from .weather_rules import WeatherRules

RULE_REGISTRY = Registry("RULE")

RULE_REGISTRY.register(WeatherRules)