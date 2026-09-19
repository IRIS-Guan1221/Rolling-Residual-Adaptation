"""Rolling Residual Adaptation for next-day sales forecasting."""

__version__ = "1.0.0"


def __getattr__(name):
    if name in {"Configuration", "RRAForecaster"}:
        from .model import Configuration, RRAForecaster
        return {"Configuration": Configuration, "RRAForecaster": RRAForecaster}[name]
    raise AttributeError(name)
