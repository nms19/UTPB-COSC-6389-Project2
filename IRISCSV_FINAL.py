import math
import random
import threading
import queue
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import numpy as np
import csv

city_scale = 15  # Scale for neuron visualization
padding = 100

# Activation Functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

def tanh(x):
    return math.tanh(x)

def tanh_derivative(output):
    return 1 - output ** 2

def relu(x):
    return max(0, x)

def relu_derivative(output):
    return 1.0 if output > 0 else 0.0

class DataPreprocessor:
    def __init__(self):
        self.X = []
        self.y = []
        self.input_size = 0
        self.output_size = 0
        self.label_mapping = {}
        self.dataset_loaded = False

    def load_data(self, filepath):
        self.X = []
        self.y = []
        self.label_mapping = {}
        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)
                headers = data[0]
                data = data[1:]  # Skip header
                random.shuffle(data)  # Shuffle the data

                # Assuming the last column is the label
                features = [list(map(float, row[:-1])) for row in data]
                labels = [row[-1] for row in data]

                unique_labels = sorted(set(labels))
                self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                numerical_labels = [self.label_mapping[label] for label in labels]

                self.X = features
                self.y = numerical_labels
                self.input_size = len(self.X[0])
                self.output_size = len(unique_labels)
                self.dataset_loaded = True
                print(f"\nData Preprocessing:")
                print(f"Data loaded successfully from '{filepath}'")
                print(f"Number of samples: {len(self.X)}")
                print(f"Number of features (inputs): {self.input_size}")
                print(f"Number of classes (outputs): {self.output_size}")
                print(f"Label mapping: {self.label_mapping}")
                print(f"First sample input: {self.X[0]}")
                print(f"First sample label: {self.y[0]}")
                print(f"Shape of input data: ({len(self.X)}, {self.input_size})")
                print(f"Shape of output data: ({len(self.y)},)")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.dataset_loaded = False

    def get_data(self):
        return self.X, self.y

class Neuron:
    def __init__(self, layer_index, neuron_index, x, y):
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.x = x
        self.y = y
        self.output = 0.0
        self.delta = 0.0
        self.bias = random.uniform(-1, 1)

    def draw(self, canvas, font_size=8):
        # Adjusted positions to prevent overlapping
        canvas.create_oval(self.x - city_scale, self.y - city_scale,
                           self.x + city_scale, self.y + city_scale,
                           fill='white', outline='black')
        canvas.create_text(self.x, self.y - city_scale - 10,
                           text=f"{self.output:.2f}",
                           font=("Arial", font_size), fill='blue')

class Weight:
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.value = random.uniform(-1, 1)

    def draw(self, canvas, color='gray', font_size=8):
        canvas.create_line(self.from_neuron.x, self.from_neuron.y,
                           self.to_neuron.x, self.to_neuron.y, fill=color)
        mid_x = (self.from_neuron.x + self.to_neuron.x) / 2
        mid_y = (self.from_neuron.y + self.to_neuron.y) / 2
        # Adjusted positions and increased font size
        canvas.create_text(mid_x, mid_y,
                           text=f"{self.value:.2f}",
                           font=("Arial", font_size), fill='red')

class NeuralNetwork:
    def __init__(self, layers, ui):
        self.layers = layers
        self.neurons = []
        self.weights = []
        self.ui = ui
        # Initialize activation functions before creating the network
        self.activation_function = sigmoid
        self.activation_derivative = sigmoid_derivative
        self.create_network()

    def set_activation_function(self, name):
        if name == 'Sigmoid':
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif name == 'Tanh':
            self.activation_function = tanh
            self.activation_derivative = tanh_derivative
        elif name == 'ReLU':
            self.activation_function = relu
            self.activation_derivative = relu_derivative

    def create_network(self):
        num_layers = len(self.layers)
        layer_width = (self.ui.w - 2 * padding) / (num_layers - 1 if num_layers > 1 else 1)
        self.neurons = []
        for l_index, num_neurons in enumerate(self.layers):
            layer_neurons = []
            layer_height = (self.ui.h - 2 * padding) / (num_neurons)
            for n_index in range(num_neurons):
                x = padding + l_index * layer_width
                y = padding + n_index * layer_height + layer_height / 2
                neuron = Neuron(l_index, n_index, x, y)
                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)

        self.weights = []
        for l in range(len(self.neurons) - 1):
            for from_neuron in self.neurons[l]:
                for to_neuron in self.neurons[l + 1]:
                    weight = Weight(from_neuron, to_neuron)
                    self.weights.append(weight)

        # Debugging output for network configuration
        print("\nNetwork Configuration:")
        print(f"Number of layers: {len(self.layers)}")
        print(f"Layers (neurons per layer): {self.layers}")
        print(f"Activation function: {self.activation_function.__name__}")
        print(f"Total weights: {len(self.weights)}")

    def forward(self, inputs):
        for i, value in enumerate(inputs):
            self.neurons[0][i].output = value

        for l in range(1, len(self.layers)):
            for neuron in self.neurons[l]:
                total_input = neuron.bias
                for prev_neuron in self.neurons[l - 1]:
                    for weight in self.weights:
                        if weight.from_neuron == prev_neuron and weight.to_neuron == neuron:
                            total_input += prev_neuron.output * weight.value
                neuron.output = self.activation_function(total_input)
        outputs = [neuron.output for neuron in self.neurons[-1]]
        return outputs

    def backward(self, targets, learning_rate):
        # Calculate output layer deltas
        for i, neuron in enumerate(self.neurons[-1]):
            error = targets[i] - neuron.output
            neuron.delta = error * self.activation_derivative(neuron.output)

        # Calculate hidden layer deltas
        for l in reversed(range(1, len(self.layers) - 1)):
            for neuron in self.neurons[l]:
                error = 0.0
                for weight in self.weights:
                    if weight.from_neuron == neuron:
                        error += weight.to_neuron.delta * weight.value
                neuron.delta = error * self.activation_derivative(neuron.output)

        # Update weights and biases
        for weight in self.weights:
            change = learning_rate * weight.from_neuron.output * weight.to_neuron.delta
            weight.value += change

        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.bias += learning_rate * neuron.delta

    def train(self, training_data, epochs, learning_rate, update_queue):
        update_queue.put({'type': 'print', 'message': "\nTraining Started:"})
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            for inputs, targets in training_data:
                outputs = self.forward(inputs)
                loss = sum((t - o) ** 2 for t, o in zip(targets, outputs)) / 2
                total_loss += loss
                self.backward(targets, learning_rate)
                # Send GUI update task less frequently to reduce overhead
                if random.random() < 0.1:
                    update_queue.put({'type': 'update_network', 'network': self})
                # Calculate training accuracy
                predicted_label = outputs.index(max(outputs))
                actual_label = targets.index(max(targets))
                if predicted_label == actual_label:
                    correct_predictions += 1

            avg_loss = total_loss / len(training_data)
            accuracy = (correct_predictions / len(training_data)) * 100
            message = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%"
            update_queue.put({'type': 'print', 'message': message})
            # Include 'epoch' and 'total_epochs' in the update_metrics task
            update_queue.put({'type': 'update_metrics', 'loss': avg_loss, 'accuracy': accuracy, 'epoch': epoch + 1, 'total_epochs': epochs})
        update_queue.put({'type': 'print', 'message': "Training completed."})

        # Send final weights and neuron outputs
        weights = self.get_weights()
        neuron_outputs = self.get_neuron_outputs()
        update_queue.put({'type': 'training_complete', 'weights': weights, 'neuron_outputs': neuron_outputs})

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs.index(max(outputs))

    def get_weights(self):
        weights_per_layer = []
        for l in range(len(self.neurons) - 1):
            layer_weights = []
            for from_neuron in self.neurons[l]:
                weights_to_next_layer = []
                for to_neuron in self.neurons[l + 1]:
                    for weight in self.weights:
                        if weight.from_neuron == from_neuron and weight.to_neuron == to_neuron:
                            weights_to_next_layer.append(weight.value)
                            break  # Found the weight
                layer_weights.append(weights_to_next_layer)
            weights_per_layer.append(layer_weights)
        return weights_per_layer

    def get_neuron_outputs(self):
        outputs_per_layer = []
        for layer in self.neurons:
            outputs = [neuron.output for neuron in layer]
            outputs_per_layer.append(outputs)
        return outputs_per_layer

class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Neural Network Visualization")
        self.option_add("*tearOff", FALSE)
        width, height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (width, height))
        self.state("zoomed")
        self.canvas = Canvas(self, bg='white')
        self.canvas.place(x=200, y=0, width=width - 200, height=height)
        self.w = width - padding * 2 - 200
        self.h = height - padding * 2
        self.network = None
        self.data_processor = DataPreprocessor()
        self.create_menu()
        self.activation_function = 'Sigmoid'
        self.update_queue = queue.Queue()
        self.training_thread = None
        self.create_control_panel()

    def create_menu(self):
        menu_bar = Menu(self)
        self['menu'] = menu_bar
        menu_NN = Menu(menu_bar)
        menu_bar.add_cascade(menu=menu_NN, label='Neural Network', underline=0)
        menu_NN.add_command(label="Load Data", command=self.load_data, underline=0)
        menu_NN.add_command(label="Generate Network", command=self.generate_network, underline=0)
        menu_NN.add_command(label="Train Network", command=self.train_network, underline=0)
        menu_NN.add_command(label="Test Network", command=self.test_network, underline=0)
        menu_NN.add_command(label="Reset Network", command=self.reset_network, underline=0)

    def create_control_panel(self):
        control_frame = Frame(self)
        control_frame.place(x=0, y=0, width=200, height=self.winfo_screenheight())

        self.layer_label = Label(control_frame, text="Hidden Layers (comma-separated):")
        self.layer_label.pack()
        self.layer_entry = Entry(control_frame)
        self.layer_entry.pack()
        self.layer_entry.insert(0, '5')

        self.lr_label = Label(control_frame, text="Learning Rate:")
        self.lr_label.pack()
        self.lr_entry = Entry(control_frame)
        self.lr_entry.pack()
        self.lr_entry.insert(0, '0.1')

        self.epochs_label = Label(control_frame, text="Epochs:")
        self.epochs_label.pack()
        self.epochs_entry = Entry(control_frame)
        self.epochs_entry.pack()
        self.epochs_entry.insert(0, '100')

        self.activation_label = Label(control_frame, text="Activation Function:")
        self.activation_label.pack()
        self.activation_var = StringVar(value='Sigmoid')
        self.activation_menu = OptionMenu(control_frame, self.activation_var, 'Sigmoid', 'Tanh', 'ReLU')
        self.activation_menu.pack()

        # Labels to display training metrics
        self.loss_label = Label(control_frame, text="Loss: N/A")
        self.loss_label.pack(pady=10)
        self.accuracy_label = Label(control_frame, text="Accuracy: N/A")
        self.accuracy_label.pack(pady=10)

        # Progress bar for training progress
        self.progress_label = Label(control_frame, text="Training Progress:")
        self.progress_label.pack()
        self.progress_var = DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=10, pady=5)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data_processor.load_data(file_path)
            if self.data_processor.dataset_loaded:
                print("Data loaded and preprocessed successfully.")

    def generate_network(self):
        if not self.data_processor.dataset_loaded:
            print("Please load a dataset first.")
            return
        try:
            hidden_layers = [int(s) for s in self.layer_entry.get().split(',') if s.strip()]
            layers = [self.data_processor.input_size] + hidden_layers + [self.data_processor.output_size]
            self.network = NeuralNetwork(layers, self)
            self.network.set_activation_function(self.activation_var.get())
            self.draw_network(self.network)
        except Exception as e:
            print(f"Error generating network: {e}")

    def draw_network(self, network):
        self.canvas.delete("all")
        for weight in network.weights:
            weight.draw(self.canvas, font_size=10)
        for layer in network.neurons:
            for neuron in layer:
                neuron.draw(self.canvas, font_size=10)

    def train_network(self):
        if not self.network:
            print("Please generate a network first.")
            return
        try:
            learning_rate = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            X, y = self.data_processor.get_data()

            # One-hot encode the labels
            num_classes = self.data_processor.output_size
            y_one_hot = []
            for label in y:
                one_hot = [0] * num_classes
                one_hot[label] = 1
                y_one_hot.append(one_hot)

            # Split data into training and testing sets (80% train, 20% test)
            split_index = int(0.8 * len(X))
            self.training_data = list(zip(X[:split_index], y_one_hot[:split_index]))
            self.testing_data = list(zip(X[split_index:], y[split_index:]))

            print(f"\nData Split:")
            print(f"Training samples: {len(self.training_data)}")
            print(f"Testing samples: {len(self.testing_data)}")

            self.network.set_activation_function(self.activation_var.get())

            # Reset progress bar and labels
            self.progress_var.set(0)
            self.loss_label.config(text="Loss: N/A")
            self.accuracy_label.config(text="Accuracy: N/A")

            # Run training in a separate thread
            self.training_thread = threading.Thread(target=self.run_training_thread, args=(epochs, learning_rate))
            self.training_thread.daemon = True
            self.training_thread.start()

            # Start the GUI update loop
            self.after(100, self.process_queue)
        except Exception as e:
            print(f"Error during training: {e}")

    def run_training_thread(self, epochs, learning_rate):
        self.network.train(self.training_data, epochs, learning_rate, self.update_queue)

    def process_queue(self):
        try:
            while True:
                task = self.update_queue.get_nowait()
                if task['type'] == 'update_network':
                    self.draw_network(task['network'])
                elif task['type'] == 'print':
                    print(task['message'])
                elif task['type'] == 'update_metrics':
                    loss = task['loss']
                    accuracy = task['accuracy']
                    current_epoch = task['epoch']
                    total_epochs = task['total_epochs']
                    self.loss_label.config(text=f"Loss: {loss:.4f}")
                    self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
                    # Update progress bar
                    progress = (current_epoch / total_epochs) * 100
                    self.progress_var.set(progress)
                elif task['type'] == 'training_complete':
                    weights = task['weights']
                    neuron_outputs = task['neuron_outputs']
                    # Print weights and neuron outputs
                    print("\nFinal Weights per Layer:")
                    for l, layer_weights in enumerate(weights):
                        print(f"Layer {l} to Layer {l+1} Weights:")
                        for from_neuron_idx, weights_to_next_layer in enumerate(layer_weights):
                            print(f"  Neuron {from_neuron_idx} weights: {weights_to_next_layer}")
                    print("\nFinal Neuron Outputs per Layer:")
                    for l, outputs in enumerate(neuron_outputs):
                        print(f"Layer {l} Outputs: {outputs}")
                    self.progress_var.set(100)
        except queue.Empty:
            pass
        if self.training_thread and self.training_thread.is_alive():
            self.after(100, self.process_queue)
        else:
            self.after(1000, self.process_queue)  # Keep checking for any remaining messages

    def test_network(self):
        if not self.network:
            print("Please generate and train the network first.")
            return
        if not hasattr(self, 'testing_data'):
            print("No testing data available.")
            return
        correct_predictions = 0
        total_predictions = len(self.testing_data)
        for inputs, label in self.testing_data:
            prediction = self.network.predict(inputs)
            if prediction == label:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        print(f"\nTesting Results:")
        print(f"Testing completed. Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})")

    def reset_network(self):
        self.canvas.delete("all")
        self.network = None
        print("\nNetwork has been reset.")

if __name__ == '__main__':
    ui = UI()
    ui.mainloop()
