# The state estimate depends on the choice of parameter matrices F, H, Q and R.
# These can be estimated by maximizing the Gaussian log likelihood function. 
#$$
#\mathscr{L} = -\frac{1}{2} (N \ln(2\pi) + \ln(\sigma) + \frac{\varepsilon^2}{\sigma})
#$$
#N - number of observations

pmts = permutate_params({
    'vb': list(np.logspace(-5, 2, 10)),
    'vm': list(np.logspace(-5, 2, 10)),
})

N = len(x_train)
ll_max, best = -np.inf, None
for p in pmts:
    b, err, ev = kalman_regression_estimator(x_train.values, y_train.values, **p, intercept=False)
    
    err = err[50:]
    ev = ev[50:]
    ll = -0.5*(N * np.log(2*np.pi) + np.log(ev) + err**2/ev).sum()
    if ll > ll_max:
        ll_max = ll
        best = p
best

b, err, evars = kalman_regression_estimator(X[constituents].values, Y.values, **best, intercept=False)
cut_off = slice(15, None)
e_s = pd.Series(err, index=Y.index)[cut_off]
v_s = pd.Series(np.sqrt(evars), index=Y.index)[cut_off]
b_a = pd.DataFrame(b.T, index=Y.index, columns=constituents)[cut_off]
b_a_w = b_a.div(b_a.sum(axis=1), axis=0)

fig(16, 5)
k_prediction = pd.DataFrame(X[constituents].values * b.T, index=Y.index)[cut_off].sum(axis=1)
plt.plot(k_prediction)
plt.plot(Y, 'g'); plt.title('Kalman estimator tracking Russell1000');