"""
Neural RLT -- our own addition to this project, NOT part of the original RLT paper
(Zhu, Zeng & Kosorok, 2015).

A bagged ensemble of small neural networks: each member is trained on a bootstrap
sample and on a random subset of the features (sqrt(p) by default), and the
ensemble averages their predicted probabilities.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin


class NeuralRLTTree(BaseEstimator, ClassifierMixin):
    """
    Neural Network-based Randomized Learning Tree.
    Uses a deep neural network instead of traditional decision tree.
    """
    
    def __init__(self, max_features="sqrt", hidden_layers=(64, 32, 16),
                 dropout_rate=0.3, learning_rate=0.001, epochs=50,
                 batch_size=32, random_state=None):
        self.max_features = max_features
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.feature_indices_ = None
        self.n_features_in_ = None
        self.classes_ = None
    
    def _build_model(self, input_dim, output_dim):
        """Build the neural network architecture."""
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(input_dim,)))
        
        # Hidden layers with batch normalization and dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                name=f"hidden_{i+1}",
            ))
            model.add(BatchNormalization(name=f"bn_{i+1}"))
            model.add(Dropout(self.dropout_rate, name=f"dropout_{i+1}"))
        
        # Output layer
        if output_dim == 2:
            model.add(Dense(1, activation="sigmoid", name="output"))
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
            )
        else:
            model.add(Dense(output_dim, activation="softmax", name="output"))
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        
        return model
    
    def fit(self, X, y):
        """Train the neural network tree."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)  # Python, NumPy and TensorFlow seeds
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Random feature selection (like traditional RLT)
        if self.max_features == "sqrt":
            n_features = int(np.sqrt(self.n_features_in_))
        elif self.max_features == "log2":
            n_features = int(np.log2(self.n_features_in_))
        elif isinstance(self.max_features, int):
            n_features = self.max_features
        else:
            n_features = self.n_features_in_
        
        n_features = max(1, min(n_features, self.n_features_in_))
        self.feature_indices_ = np.random.choice(
            self.n_features_in_,
            size=n_features,
            replace=False,
        )
        
        # Select features
        X_selected = X[:, self.feature_indices_]
        
        # Build and train model
        self.model = self._build_model(
            input_dim=n_features,
            output_dim=len(self.classes_),
        )
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        )
        
        # Train
        self.model.fit(
            X_selected, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
            validation_split=0.1,
        )
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X_selected = np.asarray(X, dtype=np.float32)[:, self.feature_indices_]
        # Direct call instead of model.predict(): much faster for the API's single-row requests
        proba = self.model(X_selected, training=False).numpy().astype(float)

        if len(self.classes_) == 2:
            return np.hstack([1 - proba, proba])
        return proba
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class NeuralRLTForest(BaseEstimator, ClassifierMixin):
    """
    Neural Network-based Randomized Learning Forest.
    Ensemble of Neural RLT Trees.
    """
    
    def __init__(self, n_estimators=10, max_features="sqrt",
                 hidden_layers=(64, 32, 16), dropout_rate=0.3,
                 learning_rate=0.001, epochs=50, batch_size=32,
                 bootstrap=True, random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        """Train the neural forest (NumPy only)."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.trees_ = []
        if self.random_state is not None:
            tf.config.experimental.enable_op_determinism()  # reproducible results run to run
        rng = np.random.RandomState(self.random_state)
        
        print(f"Training Neural RLT Forest with {self.n_estimators} trees...")
        
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot, y_boot = X[indices], y[indices]
            else:
                X_boot, y_boot = X, y
            
            # Create and train tree
            tree = NeuralRLTTree(
                max_features=self.max_features,
                hidden_layers=self.hidden_layers,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                batch_size=self.batch_size,
                random_state=rng.randint(0, 10000),
            )
            
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            
            if (i + 1) % 5 == 0 or (i + 1) == self.n_estimators:
                print(f" Trained {i + 1}/{self.n_estimators} trees")
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities (average of all trees)."""
        X = np.asarray(X)
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees_])
        return np.mean(all_proba, axis=0)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
