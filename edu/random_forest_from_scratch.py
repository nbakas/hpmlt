

from import_libraries import *

# Prepare Data
obs = 1_000
vars = 10
rng = default_rng(seed=0)
Xtr = rng.random((obs, vars))
ytr = sum(Xtr, axis=1) + (rng.random(obs)-1/2)
Xte = rng.random((obs, vars))
yte = sum(Xte, axis=1) + (rng.random(obs)-1/2)


# Decision Tree Regressor
dtree = DecisionTreeRegressor(max_depth=10, random_state=0)
dtree.fit(Xtr, ytr)

# Accuracy
pred_tr = dtree.predict(Xtr)
acc_tr = corrcoef(ytr,pred_tr)[0,1]
acc_tr
pred_te = dtree.predict(Xte)
acc_te = corrcoef(yte,pred_te)[0,1]
acc_te


# Random Forest
nof_trees = 100

# Random Columns
# we need to keep inds_cols, for train AND later for prediction
nof_cols = int(0.7*vars)
inds_cols = zeros((nof_trees,nof_cols),dtype=int)
for j in range(nof_trees):
    rng = default_rng(seed=j+1)
    jj = rng.integers(low=0, high=vars, size=nof_cols)
    inds_cols[j,:] = jj

## Check if the columns have been sampled uniformly
inds_cols_vec = inds_cols.reshape(nof_trees*nof_cols,1)[:,0]
inds_cols_vec
counter = Counter(inds_cols_vec)
vals = counter.values()
keys = counter.keys()
plt.bar(keys,vals)
plt.show()

# Train Random Forest
nof_rows = int(0.7*obs)
all_trees = []
for j in range(nof_trees):
    dtree = DecisionTreeRegressor(max_depth=10, random_state=j+1)
    rng = default_rng(seed=j+1)
    ii = rng.integers(low=0, high=obs, size=nof_rows)
    dtree.fit(Xtr[ii,:][:,jj], ytr[ii])
    all_trees.append(dtree)


# Predict
pred_tr = zeros(obs)
pred_te = zeros(obs)
for j in range(nof_trees):
    j_dtree = all_trees[j]
    pred_tr += j_dtree.predict(Xtr[:,inds_cols[j,:]])
    pred_te += j_dtree.predict(Xte[:,inds_cols[j,:]])
pred_tr /= nof_trees
pred_te /= nof_trees

acc_tr = corrcoef(ytr,pred_tr)[0,1]
acc_tr
acc_te = corrcoef(yte,pred_te)[0,1]
acc_te