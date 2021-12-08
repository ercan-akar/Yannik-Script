import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

import pandas as pd

def run(inp):
    pls = PLSRegression()
    clf = GridSearchCV(pls, inp['hyperparams'])

    clf.fit(inp['df'].loc[:, inp['columns']['x']], inp['df'].loc[:, inp['columns']['y']])

    def vip(model):
        # taken from: https://github.com/scikit-learn/scikit-learn/pull/13492/files
        T = model.x_scores_
        W = model.x_weights_
        Q = model.y_loadings_
        w0, w1 = W.shape
        s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        s_sum = np.sum(s, axis=0)
        w_norm = np.array([(W[:, i] / np.linalg.norm(W[:, i]))
                            for i in range(w1)])
        vip_ = np.sqrt(w0 * np.sum(s * w_norm.T ** 2, axis=1) / s_sum)
        return vip_

    results = {}

    results['hyperparams'] = clf.best_params_
    results['model'] = clf.best_estimator_
    results['score'] = clf.best_score_

    #print(clf.cv_results_)
    #print(vip(clf.best_estimator_))

    components = pd.DataFrame(
        clf.best_estimator_.transform(inp['df'].loc[:, inp['columns']['x']]),
        index = inp['df'].loc[:, inp['columns']['batch']],
        columns = ['Component {}'.format(comp + 1) for comp in range(clf.best_params_['n_components'])]
    )

    components = components.merge(pd.concat([inp['df'].loc[:, inp['columns']['y']], inp['df'].loc[:, inp['columns']['batch']]], axis=1), right_on=inp['columns']['batch'], left_index=True).set_index(inp['df'].loc[:, inp['columns']['batch']])

    results['output'] = {
        'metrics': {
            #'r2_score': {
                # 'value': clf.cv_results_['mean_train_score'][clf.best_index_]
            #},
            'q2_score': {
                'value': clf.cv_results_['mean_test_score'][clf.best_index_],
                'style': 'single',
                'name': 'Q² Score'
            },
            'overall_score': {
                'value': r2_score(inp['df'].loc[:, inp['columns']['y']], clf.best_estimator_.predict(inp['df'].loc[:, inp['columns']['x']])),
                'style': 'single',
                'name': 'R² Score (?)'
            },
            'rmse': {
                'value': mean_squared_error(inp['df'].loc[:, inp['columns']['y']], clf.best_estimator_.predict(inp['df'].loc[:, inp['columns']['x']]), squared = False), # square = False -> RMSE
                'style': 'single',
                'name': 'RMSE'
            },
        },
        'plots': {
            'vip': {
                'value': pd.DataFrame([vip(clf.best_estimator_)], columns=inp['columns']['x']),
                'style': 'bar_h',
                'name': 'Variable Importance on Projection'
            },
            'components': {
                'value': components,
                'style': 'scatter',
                'name': 'Component-vs-compenent'
            }
        }
    }

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
            'n_components': [1,2,3,4]
        }
    }

    print(run(inp))

# This:
# clf.best_estimator_.transform(iris.data)
# should tell us, how each sample can be represented in component-space (useful for selction stuff)
# this will allow us to create component-component plots
