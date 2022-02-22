def ImageToNgs(X, Ngs): 
    X -= np.min(X)
    max_value = np.max(X)
    X = X.astype(float) / max_value
    X = np.floor(X * Ngs) + 1
    X = np.where(X == Ngs + 1, Ngs, X)
    return X
