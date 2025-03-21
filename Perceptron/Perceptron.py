import numpy as np

class Perceptron:
    def __init__(self, input_size=4, learning_rate=0.1):
        # Initialize weights and bias with small random values
        # self.weights = np.random.randn(input_size) * 0.1
        # self.bias = np.random.randn() * 0.1
        # self.learning_rate = learning_rate

        self.weights = [0.1]*input_size
        self.bias = 0.2
        self.learning_rate = learning_rate
        
    def predict(self, inputs):
        # Calculate the weighted sum plus bias
        summation = np.dot(inputs, self.weights) + self.bias
        # Apply step function (1 if positive, 0 otherwise)
        return 1 if summation > 0 else 0
    
    def train(self, training_data, labels, max_epochs=1000):
        converged = False
        epoch = 0
        
        print("Initial weights:", self.weights)
        print("Initial bias:", self.bias)
        print("\nTraining started...\n")
        
        while not converged and epoch < max_epochs:
            errors = 0
            print(f"Epoch {epoch + 1}:")
            
            for inputs, label in zip(training_data, labels):
                # Make prediction
                prediction = self.predict(inputs)
                
                # Calculate error
                error = label - prediction
                
                # Update weights and bias if prediction is wrong
                if error != 0:
                    errors += 1
                    
                    # Print calculation details
                    print(f"  Input: {inputs}, Target: {label}, Prediction: {prediction}")
                    print(f"  Error: {error}")
                    print(f"  Old weights: {self.weights}, Old bias: {self.bias}")
                    
                    # Update weights and bias using Perceptron learning rule
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error
                    
                    print(f"  New weights: {self.weights}, New bias: {self.bias}\n")
                    print("-" * 15)
            
            # Check if the model has converged (no errors)
            if errors == 0:
                converged = True
                print(f"Converged at epoch {epoch + 1}!")
            
            epoch += 1
        
        return converged, epoch

# Generate all possible 2x2 images (16 possibilities)
def generate_training_data():
    # Generate all possible 4-bit patterns
    training_data = []
    for i in range(16):
        # Convert number to 4-bit binary representation
        binary = format(i, '04b')
        # Convert to numpy array of integers
        img = np.array([int(b) for b in binary])
        training_data.append(img)
    return np.array(training_data)

# Generate labels (1 for Bright - 2 or more bright pixels, 0 for Dark - less than 2 bright pixels)
def generate_labels(training_data):
    labels = []
    for img in training_data:
        # Count bright pixels (1s)
        bright_count = np.sum(img)
        # Classify: Bright if 2 or more bright pixels, Dark otherwise
        if bright_count >= 2:
            labels.append(1)  # Bright
        else:
            labels.append(0)  # Dark
    return np.array(labels)

# Test the perceptron on all possible inputs
def test_perceptron(perceptron, test_data, test_labels):
    correct = 0
    print("\nTesting the trained perceptron on all possible inputs:")
    
    for inputs, label in zip(test_data, test_labels):
        prediction = perceptron.predict(inputs)
        bright_count = np.sum(inputs)
        correct += (prediction == label)
        
        # Determine the classification name
        true_class = "Bright" if label == 1 else "Dark"
        pred_class = "Bright" if prediction == 1 else "Dark"
        
        print(f"Image: {inputs.reshape(2, 2)}, Bright pixels: {bright_count}, True: {true_class}, Predicted: {pred_class}")
    
    accuracy = correct / len(test_data) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_data)} correct)")
    return accuracy

# Function to get user input for an image
def get_user_image():
    print("\nEnter a 2x2 image as 4 binary values (0 or 1):")
    
    while True:
        try:
            user_input = input("Enter 4 binary digits (e.g., 1011): ")
            
            # Check if input is exactly 4 characters
            if len(user_input) != 4:
                print("Please enter exactly 4 digits.")
                continue
                
            # Convert to numpy array and validate
            img = np.array([int(b) for b in user_input])
            
            # Check if all values are 0 or 1
            if not all(pixel in [0, 1] for pixel in img):
                print("Please enter only 0s and 1s.")
                continue
                
            return img
            
        except ValueError:
            print("Invalid input. Please enter only 0s and 1s.")

# Function to classify a user-provided image
def classify_user_image(perceptron):
    while True:
        user_img = get_user_image()
        bright_count = np.sum(user_img)
        prediction = perceptron.predict(user_img)
        
        # Determine the classification
        class_name = "Bright" if prediction == 1 else "Dark"
        
        print(f"\nImage: {user_img.reshape(2, 2)}")
        print(f"Number of bright pixels: {bright_count}")
        print(f"Classification: {class_name}")
        
        # Show the calculation
        summation = np.dot(user_img, perceptron.weights) + perceptron.bias
        print(f"\nCalculation: dot({user_img}, {perceptron.weights}) + {perceptron.bias} = {summation}")
        print(f"Since {'summation > 0' if summation > 0 else 'summation <= 0'}, the image is classified as {class_name}.")
        
        # Ask if user wants to try another image
        another = input("\nWould you like to classify another image? (y/n): ")
        if another.lower() != 'y':
            break

def main():
    # Create perceptron
    perceptron = Perceptron(input_size=4, learning_rate=0.1)
    
    # Generate training data and labels
    training_data = generate_training_data()
    labels = generate_labels(training_data)
    
    # Train the perceptron
    converged, epochs = perceptron.train(training_data, labels)
    
    # Final weights and bias
    print("\nFinal weights:", perceptron.weights)
    print("Final bias:", perceptron.bias)
    
    if converged:
        print(f"\nTraining successful! Converged after {epochs} epochs.")
    else:
        print("\nTraining failed to converge within the maximum number of epochs.")
    
    # Test the trained perceptron
    test_perceptron(perceptron, training_data, labels)
    
    # Allow user to input images for classification
    classify_user_image(perceptron)

if __name__ == "__main__":
    main()