# Importing necessary libraries
from import_libraries import *

# Function to find the best split in the dataset
def find_best_split(X):
    # Initialize an array for storing entropy values
    Entropy = zeros(X.shape)
    # Iterate over each feature column
    for j in range(X.shape[1]):
        # Iterate over each data point
        for i in range(X.shape[0]):
            # Find values on the left side of the split
            left_vals = X[:,j][X[:,j] <= X[i,j]]
            # Calculate the entropy for the left side if there are values
            if len(left_vals)>0:
                Entropy[i,j] += mean((left_vals - mean(left_vals))**2)
            # Find values on the right side of the split
            right_vals = X[:,j][X[:,j] > X[i,j]]
            # Calculate the entropy for the right side if there are values
            if len(right_vals)>0:
                Entropy[i,j] += mean((right_vals - mean(right_vals))**2)
    # Find the indices of the minimum entropy value
    i, j = unravel_index(argmin(Entropy), Entropy.shape)
    print(Entropy)
    return i, j

# Create the dataset
X = linspace(1,10,num=10).reshape(-1,1)
y = X.copy()
__DEPTH__ = 2

# Decision Tree Regressor
dtree = DecisionTreeRegressor(max_depth=__DEPTH__)
dtree.fit(X, y)

# Print out the tree structure, thresholds, and leaves
tree_rules = export_text(dtree)
print(tree_rules)

# Plot the decision tree
plot_tree(dtree, filled=True, node_ids=False, proportion=True, label='root') 
plt.show()

# Initialize arrays for storing thresholds, splitters, and indices for each node in the tree
tree_thresholds = zeros((__DEPTH__+1,2**__DEPTH__))
tree_splitters = zeros((__DEPTH__+1,2**__DEPTH__))
tree_inds = zeros((__DEPTH__+1,2**__DEPTH__, len(y)))

# Set the depth and node variables
depth = 0
node = 0

# Find the best split at the current depth and node
i, j = find_best_split(X)

# Store the node indices, threshold, and splitter
tree_inds[depth,node,:] = 1
tree_thresholds[depth,node] = X[i,j]
tree_splitters[depth,node] = j

# Increase the depth by 1
depth = 1

# Set the node variable to 0
node = 0

# Get the splitter from the previous depth and node
j = tree_splitters[depth-1,node]

# Split the dataset based on the threshold value from the previous depth
inds_left = X[:,j] <= tree_thresholds[depth-1,0]
inds_right = X[:,j] > tree_thresholds[depth-1,0]

# Find the best split for the left side of the split
i, j = find_best_split(X[inds_left,:])
# Store the threshold and node indices for the left side
tree_thresholds[depth,0] = X[inds_left,:][i,j]
tree_inds[depth,0,inds_left] = 1

# Find the best split for the right side of the split
i, j = find_best_split(X[inds_right,:])
# Store the threshold and node indices for the right side
tree_thresholds[depth,1] = X[inds_right,:][i,j]
# tree_inds[depth,1,inds_right] = 
