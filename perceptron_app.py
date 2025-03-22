import streamlit as st
import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, input_size=4, learning_rate=0.1):
        self.weights = np.array([0.1] * input_size)
        self.bias = 0.2
        self.learning_rate = learning_rate
        self.epoch_log = []

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0, summation

    def train(self, training_data, labels, max_epochs=1000):
        converged = False
        epoch = 0
        
        while not converged and epoch < max_epochs:
            errors = 0
            epoch_data = {"Epoch": epoch + 1, "Weights": self.weights.copy(), "Bias": self.bias, "Errors": 0}
            
            for inputs, label in zip(training_data, labels):
                prediction, _ = self.predict(inputs)
                error = label - prediction
                
                if error != 0:
                    errors += 1
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error
            
            epoch_data["Errors"] = errors
            self.epoch_log.append(epoch_data)
            
            if errors == 0:
                converged = True
            
            epoch += 1
        
        return converged, epoch


def generate_training_data():
    training_data = [np.array([int(b) for b in format(i, '04b')]) for i in range(16)]
    return np.array(training_data)


def generate_labels(training_data):
    labels = [1 if np.sum(img) >= 2 else 0 for img in training_data]
    return np.array(labels)


def draw_image_html(image):
    grid_html = f"""
    <div style="display: grid; grid-template-columns: 30px 30px; grid-template-rows: 30px 30px; gap: 2px;">
        <div style="background-color: {'yellow' if image[0] == 1 else 'gray'}; color: {'red' if image[0] == 1 else 'white'}; display: flex; align-items: center; justify-content: center; width: 30px; height: 30px;">{image[0]}</div>
        <div style="background-color: {'yellow' if image[1] == 1 else 'gray'}; color: {'red' if image[1] == 1 else 'white'}; display: flex; align-items: center; justify-content: center; width: 30px; height: 30px;">{image[1]}</div>
        <div style="background-color: {'yellow' if image[2] == 1 else 'gray'}; color: {'red' if image[2] == 1 else 'white'}; display: flex; align-items: center; justify-content: center; width: 30px; height: 30px;">{image[2]}</div>
        <div style="background-color: {'yellow' if image[3] == 1 else 'gray'}; color: {'red' if image[3] == 1 else 'white'}; display: flex; align-items: center; justify-content: center; width: 30px; height: 30px;">{image[3]}</div>
    </div>
    """
    st.markdown(grid_html, unsafe_allow_html=True)


def main():
    st.title("2x2 Image Perceptron Classifier")
    
    perceptron = Perceptron()
    training_data = generate_training_data()
    labels = generate_labels(training_data)
    converged, epochs = perceptron.train(training_data, labels)
    
    st.subheader("Perceptron Training Results")
    df = pd.DataFrame(perceptron.epoch_log)
    st.dataframe(df)
    
    st.write(f"**Converged at Epoch:** {epochs}")
    st.write(f"**Final Weights:** {perceptron.weights}")
    st.write(f"**Final Bias:** {perceptron.bias}")
    
    st.subheader("Classify Your Own 2x2 Image")
    user_input = st.text_input("Enter 4 binary digits (e.g., 1011):")
    
    if user_input and len(user_input) == 4 and all(c in '01' for c in user_input):
        user_img = np.array([int(c) for c in user_input])
        prediction, summation = perceptron.predict(user_img)
        classification = "Bright" if prediction == 1 else "Dark"
        
        draw_image_html(user_img)
        st.write(f"**Number of Bright Pixels:** {np.sum(user_img)}")
        st.write(f"**Summation Calculation:** ( {user_img} * {perceptron.weights} ) + {perceptron.bias} = {summation}")
        st.subheader(f"**Classification:** {classification}")

if __name__ == "__main__":
    main()
