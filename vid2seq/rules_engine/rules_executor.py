from .opts import Opts
from .rules import RULE_REGISTRY

class RulesExecutor:
    def __init__(self, config_path):
        cfg = Opts(cfg=config_path).parse_args()

        self.pedestrian_rules = []
        self.vehicle_rules = []

        for pedestrian_rule in cfg["pedestrian"]:
            self.pedestrian_rules.append(RULE_REGISTRY.get(pedestrian_rule)(cfg["pedestrian"][pedestrian_rule]["params"]))
        
        for vehicle_rule in cfg["vehicle"]:
            self.vehicle_rules.append(RULE_REGISTRY.get(vehicle_rule)(cfg["pedestrian"][vehicle_rule]["params"]))


    def run(self, raw_output, mode="pedestrian"):
        output = raw_output

        if mode == "pedestrian":
            for rule in self.pedestrian_rules:
                output = rule.execute(output)
        elif mode == "vehicle":
            for rule in self.vehicle_rules:
                output = rule.execute(output)
        else:
            raise NotImplementedError

        return output 

    