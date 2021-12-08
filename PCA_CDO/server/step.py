import equipment


def from_config(config):
    if not 'name' in config:
        raise ValueError("missing 'name' in equipment config")

    if not 'equipments' in config or not isinstance(config['equipments'], list) or len(config['equipments']) == 0:
        raise ValueError("step config should have equipments array with at least one equipment")

    equipments = []
    for equipment_config in config['equipments']:
        equipments.append(equipment.from_config(equipment_config))

    return Step(config['name'], equipments)


class Step():
    """A step in a production process. A step has multiple equipments."""

    def __init__(self, name, equipments):
        self.name = name
        self.equipments = equipments
        self.status = 0

    def get_data(self):
        return {
            'name': self.name,
            'status': self.status,
            'equipments': [
                {**e.get_data(), **{'step': self.name}}
                 for e in self.equipments
            ]
        }

    def get_equipment_by_name(self, name):
        for e in self.equipments:
            if e.name == name:
                return e