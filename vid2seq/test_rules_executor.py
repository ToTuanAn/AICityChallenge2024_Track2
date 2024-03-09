from rules_engine.rules_executor import RulesExecutor
import json

def test(rule_config_path, rule_mode, preds_file):
    f = open(preds_file)
 
    preds = json.load(f)
    rule_executor = RulesExecutor(config_path=rule_config_path)
    preds = rule_executor.run(preds, rule_mode=rule_mode)

    print(f"Finished {rule_mode} rules")

    with open(preds_file, 'w') as f:
        json.dump(preds, f, indent=4)


if __name__ == '__main__':
    config_path = "rules_engine/configs/rule_config.yaml"
    rule_mode = "vehicle"
    preds_file = "data/label.json"
    test(config_path, rule_mode, preds_file)
