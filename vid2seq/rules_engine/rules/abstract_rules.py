class AbstractRules:
    def __init__(self, kwargs):
        pass
    
    def execute(self, preds: str):
        return preds

class DummyRules(AbstractRules):
    def __init__(self, kwargs):
        pass
    
    def execute(self, preds: str):
        return preds