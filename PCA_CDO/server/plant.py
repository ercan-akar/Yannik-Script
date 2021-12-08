import step


def from_config(config):
    """Creates a plant from a config dict"""
    if not 'steps' in config:
        raise ValueError("plant config should have a steps array")

    steps = []
    for step_config in config['steps']:
        steps.append(step.from_config(step_config))
    
    return Plant(steps)


class Plant():

    def __init__(self, steps):
        """The config defines the process steps and equipments in the plant.
        """
        self.steps = steps
        self.status = 0

    def get_data(self):
        plant_data = {}

        plant_data['steps'] = []
        for step in self.steps:
            plant_data['steps'].append(step.get_data())
        
        return plant_data

    def get_step_by_name(self, name):
        for step in self.steps:
            if step.name == name:
                return step

    def reset(self):
        start_time = 0 #TODO: use actual time!
        for step in self.steps:
            for equip in step.equipments:
                equip.new_batch(start_time)