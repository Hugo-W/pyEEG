# TRF Explorer roadmap

The dashboard has a working first release: uploads, validation, model fitting,
regularisation and solver controls, responsive styling, channel-wise plotting,
and previous-fit overlays are implemented.

## Next priorities

- [ ] Add browser-level tests for upload, validation, fitting, and plotting.
- [ ] Add a small dashboard-specific Python test module for endpoint behavior
      and 1-D/channel-first input normalization.
- [ ] Add progress reporting or a background job for long-running fits.
- [ ] Add cancellation and a clear error state for failed computations.
- [ ] Add export of coefficients and fit metadata as downloadable `.npz`/JSON.
- [ ] Add selectable feature and channel traces rather than always plotting the
      first feature across all channels.
- [ ] Add optional train/test scoring and cross-validation controls.
- [ ] Consider a persistent upload/session backend for deployments that need it.

## Design and accessibility

- [ ] Replace inline event handlers with a small modular frontend structure.
- [ ] Add keyboard and screen-reader tests for upload zones and controls.
- [ ] Add a no-network visual fallback for the hosted font imports.
- [ ] Add dark theme support if it remains useful for long analysis sessions.

## Deliberately out of scope for now

- Authentication and multi-user collaboration.
- GPU-specific fitting.
- 3-D visualisation.
- Additional file formats beyond NumPy arrays.
