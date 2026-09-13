"""
Export script.

Will export the trained hybrid CNN, the classical-feature scaler, and the
Isolation Forest artifact from ml_training checkpoints into the lightweight
format backend/app/ml/ loads at runtime. Target: small enough and fast
enough for CPU-only inference on Render's free tier (512MB RAM). Not
implemented yet.
"""
