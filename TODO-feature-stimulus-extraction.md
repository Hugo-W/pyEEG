"""
# Stimulus Feature Extraction Module - Implementation TODO

**Branch:** feature/stimulus-extraction
**Issue:** #15 - Add Stimulus Feature Extraction module
**Status:** In Progress

## Completed
- Create feature branch
- Create pyeeg/features/ subpackage
- Create pyeeg/models/ subpackage
- Add LLM feature extractor (from NNLM/lm_featurize)
- Add syntactic feature extractor (from process_txt/pyProcess/parseMetrics.py)
- Add alignment handler
- Add feature pipeline
- Add feature reducer
- Add refactored TRFEstimator
- Update package __init__.py (partial: `pyeeg/features/` is wired in, but the `pyeeg/models/` subpackage is currently hidden via dotfile rename (commit 5335579) and `models/trf.py` is dead code — tracked in issue #11)

## Next Steps
- Test all components
- Add documentation
- Refactor existing code
- Ensure backward compatibility

## Reference Repositories
- NNLM/lm_featurize
- process_txt/pyProcess/parseMetrics.py

Last Updated: June 5, 2026
"""