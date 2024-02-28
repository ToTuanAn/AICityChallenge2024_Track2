from .abstract_rules import AbstractRules 

class WeatherRules(AbstractRules):
    def __init__(self, kwargs):
        super().__init__(kwargs)

    def execute(self, output: str):
        return output