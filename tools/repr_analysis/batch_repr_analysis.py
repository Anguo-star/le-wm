"""
Compatibility wrapper.

Batch representation analysis now lives in `tools.repr_analysis.analyze_repr`.
This module is kept only so older shell scripts that call
`python -m tools.repr_analysis.batch_repr_analysis` continue to work.
"""

from tools.repr_analysis.analyze_repr import main


if __name__ == "__main__":
    main()
