import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Force TensorFlow doesn't work with my RTX 5080 so im forcing it to use CPU only. 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Step 1: Load Pre-trained Model
def load_pretrained_model():
    print("STEP 1: Loading Pre-trained InceptionResNetV2 Model")

    # Load the pre-trained model 
    pre_model = keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

    # Display model summary
    print("Pre-trained InceptionResNetV2 Model Summary:")
    pre_model.summary()
    
    # Count and display parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in pre_model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in pre_model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"\nModel Parameters (for write-up):")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # Visualize first layer filters
    visualize_first_layer_filters(pre_model)
    
    return pre_model

# Step 1: Visualize first layer filters
def visualize_first_layer_filters(model, save_path='first_layer_filters.png'):
    print("\nVisualizing first layer filters...")
    
    # Get the first convolutional layer
    first_layer = None
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            first_layer = layer
            break
    
    if first_layer is not None:
        # Get the weights (filters) from the first layer
        filters = first_layer.get_weights()[0]
        print(f"First layer: {first_layer.name}")
        print(f"Filter shape: {filters.shape}")
        
        # Normalize filters for visualization
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        
        # Plot first 32 filters
        n_filters = min(32, filters.shape[-1])
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle('First Layer Filters - InceptionResNetV2', fontsize=16)
        
        for i in range(n_filters):
            ax = axes[i // 8, i % 8]
            # Convert filter to RGB if it has 3 channels
            if filters.shape[-2] == 3:
                ax.imshow(filters[:, :, :, i])
            else:
                ax.imshow(filters[:, :, 0, i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"First layer filters saved as '{save_path}'")
        print("INSIGHT: These filters detect low-level features like edges, textures, and color gradients.")
    else:
        print("No convolutional layer found for filter visualization.")

#Step 2: Load and Pre-process Dataset the cat/dog dataset
def create_data_generators(train_dir, test_dir, target_size=(150, 150), batch_size=32):
    print("\nSTEP 2: Loading and Pre-processing Cat/Dog Dataset")
    
    # Create data generators with augmentation for training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,           
        rotation_range=20,        
        width_shift_range=0.2,    
        height_shift_range=0.2,   
        horizontal_flip=True,     
        zoom_range=0.2,           
        fill_mode='nearest'       
    )
    
    # Test data generator with only rescaling (no augmentation)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,  
        batch_size=batch_size,
        class_mode='binary',      # Binary classification (cat=0, dog=1)
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,  
        batch_size=batch_size,
        class_mode='binary',      # Binary classification
        shuffle=False             
    )
    
    # Display dataset information (for write-up)
    print(f"\nDataset Information:")
    print(f"Training samples: {train_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    print(f"Image shape: {target_size + (3,)}")
    
    return train_generator, test_generator

#Step 3: Create Transfer Learning Model with 'Transfer Head'
def create_transfer_learning_model(pre_model, dense_units=128, dropout_rate=0.5, learning_rate=0.001):
    print("\nSTEP 3: Creating Transfer Learning Model with 'Transfer Head'")

    model = keras.Sequential([pre_model,keras.layers.GlobalAveragePooling2D(),keras.layers.Dense(dense_units, activation='relu'),keras.layers.Dropout(dropout_rate),keras.layers.Dense(1, activation='sigmoid')])
    
    # Display model summary
    print("Transfer Learning Model Architecture Summary:")
    model.summary()
    
    # Count parameters number amount
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"\nTransfer Model Parameters (for write-up):")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # Freeze pre-trained model weights 
    pre_model.trainable = False
    print(f"\nPre-trained weights frozen: pre_model.trainable = {pre_model.trainable}")
    
    # Recount parameters after freezing
    trainable_params_after = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters after freezing: {trainable_params_after:,}")
    
    # Compile model with binary cross-entropy loss 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # Adam optimizer
        loss='binary_crossentropy',# Binary cross-entropy loss 
        metrics=['accuracy']
    )
    
    print(f"Model compiled with:")
    print(f"- Optimizer: Adam (learning_rate={learning_rate})")
    print(f"- Loss: binary_crossentropy")
    print(f"- Metrics: accuracy")
    
    return model

#Step 4: Evaluate Model BEFORE Training
def evaluate_model_before_training(model, test_generator, save_path='before.png'):
    print("\nSTEP 4i: Evaluating Transfer Model BEFORE Training")
    
    # Evaluate on test dataset
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Results BEFORE Training:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions for confusion matrix
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_generator.classes
    
    # Create and display confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix - BEFORE Training (Pre-trained weights only)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Pre-training confusion matrix saved as '{save_path}'")
    print(f"Pre-training test accuracy: {test_accuracy:.4f} (for write-up)")
    
    return test_accuracy

#Step 4ii: Train the Model and Report Per-Epoch Results
def train_model(model, train_generator, test_generator, epochs=10):
    print("\nSTEP 4ii:Training Transfer Learning Model")
    
    print(f"Training model for {epochs} epochs with binary cross-entropy loss...")
    print("Optimization algorithm: Adam optimizer")
    print("Loss function: binary_crossentropy")
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        verbose=1
    )
    
    # Report per-epoch test loss (for write-up)
    print("\nPer-epoch validation loss (for write-up):")
    for epoch, val_loss in enumerate(history.history['val_loss'], 1):
        print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
    
    # Final evaluation
    print("\nFinal model evaluation after training:")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Final test accuracy: {test_accuracy:.4f}")
    
    return history, test_accuracy

#Step 4iii: Sub-Network Transfer Learning
def create_sub_network_model(pre_model, num_layers=100, dense_units=128, dropout_rate=0.5, learning_rate=0.001):
    print("\nSTEP 4iii: Transfer Learning with Sub-Network")
    
    print(f"Creating sub-network using first {num_layers} layers of InceptionResNetV2")
    print(f"Original model has {len(pre_model.layers)} layers")
    
    # Create sub-network by taking only first k layers
    sub_network = tf.keras.Model(
        inputs=pre_model.input,
        outputs=pre_model.layers[min(num_layers-1, len(pre_model.layers)-1)].output
    )
    
    print(f"Sub-network details:")
    print(f"- Using layers 0 to {min(num_layers-1, len(pre_model.layers)-1)}")
    print(f"- Sub-network output shape: {sub_network.output_shape}")
    
    # Create transfer learning model with sub-network
    sub_model = keras.Sequential([
        sub_network,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(dense_units, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Freeze sub-network weights
    sub_network.trainable = False
    print(f"Sub-network weights frozen: {sub_network.trainable}")
    
    # Compile model
    sub_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    print("\nSub-network Transfer Model Summary:")
    sub_model.summary()
    
    return sub_model

def train_sub_network_model(sub_model, train_generator, test_generator, epochs=10):
    """
    Train the sub-network transfer learning model and compare results.
    """
    print(f"\nTraining sub-network model for {epochs} epochs...")
    
    # Train the sub-network model
    sub_history = sub_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        verbose=1
    )
    
    # Report per-epoch test loss
    print("\nSub-network per-epoch validation loss:")
    for epoch, val_loss in enumerate(sub_history.history['val_loss'], 1):
        print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
    
    # Final evaluation
    sub_test_loss, sub_test_accuracy = sub_model.evaluate(test_generator, verbose=1)
    print(f"Sub-network final test accuracy: {sub_test_accuracy:.4f}")
    
    # Generate confusion matrix for sub-network
    test_generator.reset()
    sub_predictions = sub_model.predict(test_generator, verbose=1)
    sub_predicted_classes = (sub_predictions > 0.5).astype(int).flatten()
    true_classes = test_generator.classes
    
    cm_sub = confusion_matrix(true_classes, sub_predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix - Sub-Network Transfer Learning')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('subnetwork.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Sub-network confusion matrix saved as 'subnetwork.png'")
    
    return sub_history, sub_test_accuracy


def generate_confusion_matrix(model, test_generator, save_path='after.png'):
    print("\nGenerating final confusion matrix for Step 4ii...")
    
    # Generate predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_generator.classes
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix - AFTER Training (Step 4ii)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final confusion matrix saved as '{save_path}'")


def training_pipeline():
    # Data paths
    train_dir = "cats_dogs_dataset/dataset/training_set"
    test_dir = "cats_dogs_dataset/dataset/test_set"
    
    print("*********************************************************")
    # STEP 1: Load pre-trained model with filter visualization
    
    pre_model = load_pretrained_model()
    
    print("*********************************************************")
    # STEP 2: Load and pre-process dataset (resize to 150x150x3)
    train_generator, test_generator = create_data_generators(train_dir, test_dir)
    
    print("*********************************************************")
    # STEP 3: Create transfer learning model with frozen weights
    model = create_transfer_learning_model(pre_model)
    
    print("*********************************************************")
    # STEP 4i: Evaluate model BEFORE training (with frozen weights only)
    pre_training_accuracy = evaluate_model_before_training(model, test_generator)
    
    print("*********************************************************")
    # STEP 4ii: Train the model and report per-epoch results
    history, test_accuracy = train_model(model, train_generator, test_generator, epochs=10)
    generate_confusion_matrix(model, test_generator)
    
    print("*********************************************************")
    # STEP 4iii: Sub-network transfer learning
    print("\nANALYSIS: Comparing Full Network vs Sub-Network Transfer Learning")
    
    sub_model = create_sub_network_model(pre_model, num_layers=100)
    sub_history, sub_test_accuracy = train_sub_network_model(sub_model, train_generator, test_generator, epochs=10)
    
    print("*********************************************************")
    # Final comparison and analysis
    print(f"\nFINAL RESULTS")
    print(f"Step 4(i) - Pre-training accuracy: {pre_training_accuracy}")
    print(f"Step 4(ii) - Full network accuracy: {test_accuracy}")
    print(f"Step 4(iii) - Sub-network accuracy: {sub_test_accuracy}")
    print(f"- Improvement from pre-training to full training: {(test_accuracy - pre_training_accuracy)}")
    print(f"- Full network vs sub-network difference: {(test_accuracy - sub_test_accuracy)}")
    print("*********************************************************")

if __name__ == "__main__":
    training_pipeline()