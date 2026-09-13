"""
Business logic / orchestration layer.

Will contain image validation (format, size, decodability), the pipeline
that calls backend/app/ml/ to run the classical-feature extraction, the
hybrid CNN, and the Isolation Forest anomaly detector, and the mapping of
raw model outputs into the API's structured analysis result. Not
implemented yet.
"""
