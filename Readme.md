# Screen Time Predictor

This project leverages a deep neural network, implemented using TensorFlow, to predict user screen time based on behavioral data. The dataset, originally sourced from Kaggle, is detailed here: [Mobile Device Usage and User Behavior Dataset](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset).

---

## Features

- **Neural Network Architecture**: A multi-layered perceptron designed for regression tasks with a focus on predictive accuracy.
- **Advanced Training Techniques**: Utilizes early stopping and adaptive learning rate strategies to optimize performance and prevent overfitting.
- **Evaluation Metrics**: Employs MAE, MSE, RMSE, and R² score for comprehensive performance assessment.
- **Data Preprocessing**: Prepares input datasets in `.npz` format for streamlined usage in the model pipeline.

---

## Requirements

### Python Libraries
The following dependencies are necessary to run the project:

- TensorFlow
- NumPy
- scikit-learn

### Installation
Install the required packages via pip:

```bash
pip install tensorflow numpy scikit-learn
```

### Dataset
The dataset must be converted into `.npz` files for model consumption. Files required:

- `Screentime_train.npz`
- `Screentime_validation.npz`
- `Screentime_test.npz`

These files should contain appropriately formatted inputs and target outputs.

---

## Model Architecture

The neural network comprises the following components:

1. **Input Layer**: 9 features representing user behavior.
2. **Hidden Layers**: Four layers with 110 neurons each, employing ReLU activation.
3. **Output Layer**: Single neuron for regression tasks without activation.

---

## Training Configuration

### Hyperparameters
- **Batch Size**: 64
- **Epochs**: 50
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)

### Callbacks
- **Early Stopping**: Halts training if validation loss fails to improve for 3 consecutive epochs and restores the best weights.
- **Learning Rate Reduction**: Decreases the learning rate by 50% if validation loss stagnates for 2 epochs, with a minimum threshold of 1e-6.

---

## Evaluation Metrics

Performance is evaluated using:

- **Mean Absolute Error (MAE)**: Quantifies average prediction error.
- **Mean Squared Error (MSE)**: Emphasizes larger errors due to squaring.
- **Root Mean Squared Error (RMSE)**: Provides a comparable scale to the original data.
- **R² Score**: Indicates the proportion of variance explained by the model.

---

## Code Structure

### Data Loading
```python
npz = np.load('Screentime_train.npz')
train_inputs = npz['inputs'].astype(np.float32)
train_outputs = npz['targets'].astype(np.int64)
# Repeat for validation and test data
```

### Model Definition
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dense(hidden_layer, activation='relu'),
    tf.keras.layers.Dense(hidden_layer, activation='relu'),
    tf.keras.layers.Dense(hidden_layer, activation='relu'),
    tf.keras.layers.Dense(output_size)
])
```

### Training and Callbacks
```python
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=epochs, validation_data=(validation_inputs, validation_outputs), callbacks=[early_stopping, reduce_lr])
```

### Evaluation
```python
mae = mean_absolute_error(actual_outputs, predictions)
mse = mean_squared_error(actual_outputs, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_outputs, predictions)
```

---

## Running the Code

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Ensure the `.npz` dataset files are in the project directory.

3. Execute the Python script:
   ```bash
   python screentime_predictor.py
   ```

4. Review the evaluation metrics displayed in the console.

---

## Results

The model delivers robust predictive capabilities. Example results include:

- **MAE**: ~5.0
- **MSE**: ~30.0
- **RMSE**: ~5.5
- **R² Score**: ~0.85

---

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests with enhancements or bug fixes.

---

## License

This project is distributed under the MIT License.

