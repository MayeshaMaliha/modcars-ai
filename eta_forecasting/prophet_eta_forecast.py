import pandas as pd
import numpy as np
from prophet import Prophet

# --------- STEP 1: Generate Synthetic Delivery Data ---------
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100)
vendors = ['Vendor_A', 'Vendor_B', 'Vendor_C']
delivery_days = [np.random.poisson(lam=3 + i % 3) for i in range(100)]

df = pd.DataFrame({
    'ds': dates,
    'y': delivery_days,
    'vendor': np.random.choice(vendors, size=100)
})

# Filter for a single vendor (Prophet works per time-series)
vendor_data = df[df['vendor'] == 'Vendor_A'][['ds', 'y']]

# --------- STEP 2: Train Prophet Model ---------
model = Prophet()
model.fit(vendor_data)

# --------- STEP 3: Forecast Future Delivery Times ---------
future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)

# --------- STEP 4: View Results ---------
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
