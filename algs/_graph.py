"""CUDA graph capture as a context manager.

`wp.capture_begin / wp.capture_end` requires a try/except dance to tear
the capture down cleanly if anything inside fails.  This wraps that
boilerplate so the body of a capture reads as the algorithm.

Usage:
    with capture_graph() as cap:
        # ... wp.launch / mjwarp.step / wp.copy calls captured here ...
    self._graph = cap.graph
    wp.capture_launch(self._graph)

`force_module_load=False` is preserved (kernels must be pre-compiled by
a warm-up call before the capture).
"""

import warp as wp


class _Capture:
    def __init__(self) -> None:
        self.graph: "wp.Graph | None" = None

    def __enter__(self) -> "_Capture":
        wp.capture_begin(force_module_load=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Exception inside the body — tear capture down before propagating.
            wp.capture_end()
            return False
        self.graph = wp.capture_end()
        return False


def capture_graph() -> _Capture:
    return _Capture()
