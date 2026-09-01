"""Concrete `AITextDetector` implementations.

One module per provider. Everything vendor-specific - endpoints, auth headers,
request and response shapes, error payloads - lives in here and nowhere else,
so a provider swap touches this package alone. `layers.ai_text` defines the
seam; the detection layers depend on that seam and never on a module here.
"""
