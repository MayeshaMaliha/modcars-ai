import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# --------- STEP 1: Generate Synthetic ModCars Data ---------
np.random.seed(42)

car_models = ['Toyota 86', 'Mazda RX-8', 'Nissan 350Z', 'Honda Civic', 'Subaru WRX']
engine_types = ['2.0L Boxer', '1.3L Rotary', '3.5L V6', '1.8L I4', '2.0L Turbo']
part_categories = ['Suspension', 'Exhaust', 'Body Kit', 'Tires', 'Brakes']
mod_intents = ['Performance', 'Looks']
budget_classes = ['Economy', 'Mid', 'Premium']

data = pd.DataFrame({
    'car_model': np.random.choice(car_models, 100),
    'year': np.random.randint(2005, 2023, 100),
    'engine_type': np.random.choice(engine_types, 100),
    'part_category': np.random.choice(part_categories, 100),
    'mod_intent': np.random.choice(mod_intents, 100),
    'budget_class': np.random.choice(budget_classes, 100),
    'compatibility_status': np.random.choice([0, 1], 100, p=[0.3, 0.7])
})

# --------- STEP 2: Prepare the Dataset ---------
X = pd.get_dummies(data.drop('compatibility_status', axis=1))
y = data['compatibility_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# --------- STEP 3: Train the LightGBM Model ---------
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1
}
model = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=10, verbose_eval=False)

# --------- STEP 4: Predict and Evaluate ---------
y_pred = model.predict(X_test)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
