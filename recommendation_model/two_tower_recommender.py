import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd

# --------- STEP 1: Create Synthetic User-Part Interaction Data ---------
users = ['u001', 'u002', 'u003', 'u004', 'u005']
parts = ['p101', 'p102', 'p103', 'p104', 'p105']
data = pd.DataFrame({
    'user_id': np.random.choice(users, 100),
    'part_id': np.random.choice(parts, 100)
})

# TensorFlow dataset
interactions = tf.data.Dataset.from_tensor_slices({
    "user_id": data['user_id'].astype(str).values,
    "part_id": data['part_id'].astype(str).values,
})

# --------- STEP 2: Define Two-Tower Model ---------
unique_users = data['user_id'].unique()
unique_parts = data['part_id'].unique()

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=unique_users),
    tf.keras.layers.Embedding(len(unique_users) + 1, 32)
])

part_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=unique_parts),
    tf.keras.layers.Embedding(len(unique_parts) + 1, 32)
])

class ModCarsRecommender(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.query_model = user_model
        self.candidate_model = part_model
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            candidates=interactions.batch(32).map(lambda x: x["part_id"])
        ))

    def compute_loss(self, features, training=False):
        return self.task(self.query_model(features["user_id"]),
                         self.candidate_model(features["part_id"]))

# --------- STEP 3: Train Model ---------
model = ModCarsRecommender()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(interactions.batch(16), epochs=3)
