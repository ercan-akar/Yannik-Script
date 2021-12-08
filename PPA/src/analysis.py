import json


def check_cases(cases, time_or_batch, classification_or_regression):
    # cases, where both are given
    if any([case['data'] == time_or_batch and case['analysis'] == classification_or_regression for case in cases if 'data' in case.keys() and 'analysis' in case.keys()]):
        return True
    # cases, where only data is given
    if any([case['data'] == time_or_batch for case in cases if 'data' in case.keys() and 'analysis' not in case.keys()]):
        return True
    # cases, where only analysis is given
    if any([case['analysis'] == classification_or_regression for case in cases if 'data' not in case.keys() and 'analysis' in case.keys()]):
        return True
    return False

class Analysis:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def get_name(self):
        return self.name

    def check(self, parameter, time_or_batch, classification_or_regression):
        return self.description[parameter]['always'] or check_cases(self.description[parameter]['cases'], time_or_batch, classification_or_regression)

    def is_available(self, time_or_batch, classification_or_regression):
        return self.check('availability', time_or_batch, classification_or_regression)

    def single_x(self, time_or_batch, classification_or_regression):
        return self.check('single_x', time_or_batch, classification_or_regression)

    def single_y(self, time_or_batch, classification_or_regression):
        return self.check('single_y', time_or_batch, classification_or_regression)

    def get_hyperparameters(self):
        # extract hyperparameters from description
        # return a list of dicts (this might even be equal to what we have in our description
        return self.description['hyperparameters']

    def get_module_name(self):
        return self.description['python_module']['name']

    def get_function_name(self):
        return self.description['python_module']['function']

    def get_common_name_of_hyperparam(self, internal_name):
        for param in self.description['hyperparameters']:
            if param['internal_name'] == internal_name:
                return param['name']
        return None

    def __repr__(self):
        return 'Analysis: {}'.format(self.name)

def read_from_file(filename):
    analyses = json.load(open(filename))

    a = []
    for name in analyses.keys():
        analysis = Analysis(name, analyses[name])
        a.append(analysis)

    print(a)
    return a

if __name__ == '__main__':
    read_from_file('../spec/options.json')
