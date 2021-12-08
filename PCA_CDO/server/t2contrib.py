import numpy as np
import scipy.linalg as linalg
import scipy.interpolate as interp

def P(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    scores = m.pls.transform(X)
    
    P = m.pls.x_loadings_
    X_scaled = (X - m.pls.x_mean_) / m.pls.x_std_
    for i in range(X.shape[1]):
        for a in components:
            s = scores[:, a]
            p = P[i, a]
            if a == 2:
                s = -s
                p = -p
            contributions[:,i] += s / m.scores_eigenvalues[a] * p * X_scaled[:,i]
            
    return contributions

def P_avg(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    scores = m.pls.transform(X)

    P = m.pls.x_loadings_
    
    mean_interp = interp.interp1d(x=m.time_grid, y=m.mean, axis=0, kind='linear')
    X_scaled = X - mean_interp(t) # (X - m.pls.x_mean_) / m.pls.x_std_
    
    for i in range(X.shape[1]):
        for a in components:
            contributions[:,i] += scores[:,a] / m.scores_eigenvalues[a] * P[i,a] * X_scaled[:,i]
            
    return contributions

def W(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    scores = m.pls.transform(X)
    
    W = m.pls.x_rotations_
    X_scaled = (X - m.pls.x_mean_) / m.pls.x_std_
    for i in range(X.shape[1]):
        for a in components:
            contributions[:,i] += scores[:,a] / m.scores_eigenvalues[a] * W[i,a] * X_scaled[:,i]
            
    return contributions

def PTP(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    P = m.pls.x_loadings_
    
    X_scaled = (X - m.pls.x_mean_) / m.pls.x_std_
    
    PTP = np.linalg.inv(np.dot(P.T,P))
    t_new = np.dot(PTP, np.dot(P.T, X_scaled.T))
    for i in range(X.shape[1]):
        x_i = X_scaled[:,i:i+1]
        p_i = P[i:i+1, :].T
        contributions[:,i] = t_new.T @ np.linalg.inv(m.scores_covariance) @ (PTP @ p_i @ x_i)
            
    return contributions

def sqrtP(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    P = m.pls.x_loadings_
    
    X_scaled = (X - m.pls.x_mean_) / m.pls.x_std_
    
    lamb = np.linalg.inv(linalg.sqrtm(m.scores_covariance))

    for i in range(X.shape[1]):
        c = lamb @ P[i:i+1, :].T @ X_scaled[:, i:i+1].T
        contributions[:,i] = np.sum(c * c, axis=0)
            
    return contributions

def sqrtR(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    R = m.pls.x_rotations_
    
    X_scaled = (X - m.pls.x_mean_) / m.pls.x_std_

    lamb = np.linalg.inv(linalg.sqrtm(m.scores_covariance))

    for i in range(X.shape[1]):
        c = lamb @ R[i:i+1,:].T @ X_scaled[:,i:i+1].T
        contributions[:,i] = np.sum(c * c, axis=0)
            
    return contributions

def gamma(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    R = m.pls.x_rotations_
    gamma = R @ np.linalg.inv(np.diag(m.scores_eigenvalues)) @ R.T
    gamma = linalg.sqrtm(gamma)
    
    X_scaled = (X - m.pls.x_mean_) / m.pls.x_std_

    for i in range(X.shape[1]):
        contributions[:,i] = (gamma[i:i+1] @ X_scaled.T)**2
            
    return contributions

def eigenvectorcom_p(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    scores = m.pls.transform(X)

    contributions = ((scores / np.sqrt(m.scores_eigenvalues)) @ m.pls.x_loadings_.T) ** 2

    assert contributions.shape == X.shape
    return contributions


def eigenvectorcom_r(m, X, t, components=[0,1,2]):
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    scores = m.pls.transform(X)

    contributions = ((scores / np.sqrt(m.scores_eigenvalues)) @ m.pls.x_rotations_.T) ** 2
    
    assert contributions.shape == X.shape
    return contributions

def researchgate(m, X, t, components=[0,1,2]):
    """
   
        'The contribution plot displays the differences, in scaled units,
        for all the terms in the model, between the outlying observation and
        the normal (or average) observation, multiplied by the absolute value
        of the normalized weight.'
    
    source: https://www.researchgate.net/post/A_question_about_PLS_How_can_I_compute_contribution_like_the_contribution_plot_in_Simca-P2
    """
    assert len(X.shape) == 2
    assert X.shape[0] == t.shape[0]

    contributions = np.zeros_like(X)
    
    scores = m.pls.transform(X)
    
    R = np.abs(m.pls.x_rotations_)
    
    X_scaled = (X - m.pls.x_mean_) / m.pls.x_std_
    
    for i in range(X.shape[1]):
        for a in components:
            contributions[:,i] += scores[:,a] / m.scores_eigenvalues[a] * R[i,a] * X_scaled[:,i]
            
    return contributions
