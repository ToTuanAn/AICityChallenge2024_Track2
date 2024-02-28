from typing import Optional
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml
import json

def load_yaml(path):
    with open(path, "rt") as f:
        return yaml.safe_load(f)

class Config(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, yaml_path):
        super(Config, self).__init__()

        config = load_yaml(yaml_path)
        super(Config, self).update(config)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def save_yaml(self, path):
        print(f"Saving config to {path}...")
        with open(path, "w") as f:
            yaml.dump(dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path):
        print(f"Loading config from {path}...")
        return cls(path)

    def __repr__(self) -> str:
        return str(json.dumps(dict(self), sort_keys=False, indent=4))


class Opts(ArgumentParser):
    def __init__(self, cfg: Optional[str] = None):
        super(Opts, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.cfg_path = cfg

    def parse_args(self, argv=None):
        config = Config(self.cfg_path)
        # config = self.override(config, args.opt)
        return config

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def override(self, global_config, overriden):
        """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
        print("Overriding configurating")
        for key, value in overriden.items():
            if "." not in key:
                if isinstance(value, dict) and key in global_config:
                    global_config[key].update(value)
                else:
                    if key in global_config.keys():
                        global_config[key] = value
                    print(f"'{key}' not found in config")
            else:
                sub_keys = key.split(".")
                assert (
                    sub_keys[0] in global_config
                ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                    global_config.keys(), sub_keys[0]
                )
                cur = global_config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        if sub_key in cur.keys():
                            cur[sub_key] = value
                        else:
                            print(f"'{key}' not found in config")
                    else:
                        cur = cur[sub_key]
        return global_config