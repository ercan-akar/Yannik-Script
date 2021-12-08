import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd

def run(inp):
    if inp['analysis_type'] == 'regression':
        rf = RandomForestRegressor()
    else:
        rf = RandomForestClassifier()
    clf = GridSearchCV(rf, inp['hyperparams'])
    

    clf.fit(inp['df'].loc[:, inp['columns']['x']], inp['df'].loc[:, inp['columns']['y']])

    results = {}

    results['hyperparams'] = clf.best_params_
    results['model'] = clf.best_estimator_
    results['score'] = clf.best_score_

    
    results['output'] = {
        'metrics': {
            'q2_score': {
                'value': clf.cv_results_['mean_test_score'][clf.best_index_],
                'style': 'single',
                'name': 'Q² Score'
            }
        },
        'plots': {
            'feature_importances': {
                'value': pd.DataFrame([est.feature_importances_ for est in clf.best_estimator_.estimators_], columns=inp['columns']['x']),
                'style': 'box',
                'name': 'Feature Importance'
            }
        }
    }
    
    if inp['analysis_type'] == 'regression':
        results['output']['metrics'].update({
            'overall_score': {
                'value': r2_score(inp['df'].loc[:, inp['columns']['y']], clf.best_estimator_.predict(inp['df'].loc[:, inp['columns']['x']])),
                'style': 'single',
                'name': 'R² Score (?)'
            },
            'rmse': {
                'value': mean_squared_error(inp['df'].loc[:, inp['columns']['y']], clf.best_estimator_.predict(inp['df'].loc[:, inp['columns']['x']]), squared = False), # square = False -> RMSE
                'style': 'single',
                'name': 'RMSE'
            }
        })
    else:
        y_pred = clf.best_estimator_.predict(inp['df'].loc[:, inp['columns']['x']])
        y_real = inp['df'].loc[:, inp['columns']['y']].to_numpy() # this probably does not work in the general case, leaing a TODO
        
        
        correct = 0
        for (p, r) in zip(y_pred, y_real):
            print(p == r)
            if (p == r).all():
                correct += 1
        accuracy = correct / len(y_pred)
        
        results['output']['metrics'].update({
            'accuracy': {
                'value': accuracy,
                'style': 'single',
                'name': 'Accuracy'
            }
        })

    return results

if __name__ == '__main__':
    iris = datasets.load_iris()

    # turn iris into a pandas df to be closer to what we ant to do

    data_df = pd.DataFrame(iris.data, columns = iris.feature_names)
    target_df = pd.DataFrame(iris.target, columns = ['Target'])

    total_df = pd.concat([data_df, target_df], axis=1)

    total_df['Sample Name'] = pd.Series(['Sample {}'.format(idx+1) for idx in range(total_df.shape[0])])

    inp = {
        'df': total_df,
        'columns': {
            'time': None,
            'batch': 'Sample Name',
            'x': ['sepal length (cm)', 'sepal width (cm)',  'petal length (cm)',  'petal width (cm)'],
            'y': 'Target'
        },
        'data_type': 'batch',
        'analysis_type': 'regression',
        'analysis_method': 'PLS',
        'hyperparams': {
            'n_estimators': range(1,100, 5),
            'max_depth': range(2,7)
        }
    }

    print(run(inp))

# This:
# clf.best_estimator_.transform(iris.data)
# should tell us, how each sample can be represented in component-space (useful for selction stuff)
# this will allow us to create component-component plots
