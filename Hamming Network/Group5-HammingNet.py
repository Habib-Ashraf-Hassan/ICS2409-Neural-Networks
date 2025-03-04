import numpy as np

# Define stored examples: 
# e(1) → Normal person
# e(2) → TB-Infected person
examples = np.array([
    [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],  # Normal person
    [1, 1, 1, 1,  1,  1, -1, -1, -1, -1, -1]   # TB-Infected person
])

# Define input vector (new patient symptoms)
V = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1])

# Step 1: Compute Hamming Network Output
theta = len(V) / 2  # Threshold
W = examples  # Weight matrix is same as examples
Y = np.dot(W, V) + theta  # Compute activation values

print("Hamming Network Output (Before MaxNet):", Y)

# Step 2: Apply MaxNet Algorithm
def maxnet(Y, epsilon=0.4, iterations=3):
    """Performs Winner-Take-All competition using MaxNet."""
    Y = np.copy(Y)  # Work with a copy to preserve original values
    for _ in range(iterations):
        Y = np.maximum(Y - epsilon * (np.sum(Y) - Y), 0)  # Apply suppression
    return Y

Y_maxnet = maxnet(Y)

print("MaxNet Output (After Suppression):", Y_maxnet)

# Determine classification result
winner = np.argmax(Y_maxnet)  # Find the index of the strongest neuron
result = "TB-Infected" if winner == 1 else "Normal"

print("Final Classification:", result)
