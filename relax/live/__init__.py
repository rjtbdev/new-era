"""Live inference service for new-era-relax.

Architecture:
    reader → feature_builder → model_service → dispatcher → sink

The reader produces a GameState snapshot every poll (~1-2 ms cadence). The
feature_builder keeps a rolling window of upcoming hit objects and derives
the 17-dim feature vector for the timing head. The model_service runs
inference and outputs a press decision per hit object. The dispatcher
schedules the press at the predicted absolute time; the sink executes it.

Everything on this side is pure Python + torch and is independent of
hemlock. The reader + sink are swappable: use mock_* for offline testing,
hemlock_* when the hypercalls are wired.
"""
