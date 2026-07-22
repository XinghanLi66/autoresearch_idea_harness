"""
Mid-edit operations for the cv-dbm-scheduler task.
Creates the 'blank' function from the template.
"""
from pathlib import Path
try:
    from .custom_template import _TEMPLATE
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from custom_template import _TEMPLATE

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        # Region shifted +9 (301->310, 311->320): vendor pre_edit OP 6d inserts
        # 9 lines of NFE-budget instrumentation above this function. Pristine
        # numbering was 301-311. Span 310-320 = get_sigmas_karras..get_sigmas_uniform.
        "start_line": 310,
        "end_line": 320,
        "content": _TEMPLATE,
    },
]
