from metaurban.component.traffic_light.base_traffic_light import BaseTrafficLight
from metaurban.type import MetaUrbanType


class ScenarioTrafficLight(BaseTrafficLight):
    def set_status(self, status):
        status = MetaUrbanType.parse_light_status(status, simplifying=True)
        if status == MetaUrbanType.LIGHT_GREEN:
            self.set_green()
        elif status == MetaUrbanType.LIGHT_RED:
            self.set_red()
        elif status == MetaUrbanType.LIGHT_YELLOW:
            self.set_yellow()
        elif status == MetaUrbanType.LIGHT_UNKNOWN:
            self.set_unknown()
