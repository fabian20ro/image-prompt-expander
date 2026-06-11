# Plan: Characterize mflux integration
## Status: Draft
## Lane: characterization

### Goal
Ensure the `mflux` import error handling and model loading logic is robust and correctly handles missing dependencies.

### Steps
1. **Verify Import Error Handling**: Write a test case in `tests/test_image_generator.py` that mocks `mflux` being missing and verifies `image_generator.py` raises the correct `ImportError`.
2. **Verify Model Configuration**: Verify that `_get_model` correctly selects the model config based on the requested model name.
3. **Verify Cache Behavior**: Verify that calling `_get_model` multiple times with the same arguments returns the same instance.
4. **Verify Tiled VAE Configuration**: Verify that the `tiled_vae` flag is correctly applied to the model instance.

### Verification (Done-state)
- Test suite `pytest tests/test_image_generator.py` passes.
- No regressions in existing tests.
