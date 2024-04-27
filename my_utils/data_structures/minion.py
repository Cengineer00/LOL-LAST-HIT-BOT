

class Minion:

    def __init__(
        self,
        minion_type,
        health_level,
        bbox,
        confidence,
    ) -> None:
        self.minion_type = minion_type
        self.health_level = health_level
        self.bbox = bbox
        self.confidence = confidence