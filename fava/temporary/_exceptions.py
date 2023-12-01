from typing import Any

_cls_name = "Temporary"

class NotCallableError(Exception):
    def __init__(self, callable_name: Any):
        message = f"< {callable_name} > is not a callable function or class."
        super().__init__(message)


class InvalidMeshError(Exception):
    def __init__(self, mesh_cls: str):            
        message = f"Unknown mesh class < {mesh_cls} >\n\n\tIf you implemented this mesh class, did you register it with the @{_cls_name}.register_mesh decorator?"
        super().__init__(message)


class InvalidAnalysisError(Exception):
    def __init__(self, analysis_attr: str):            
        message = f"Unknown analysis method < {analysis_attr} >\n\n\tIf you implemented this method, did you register it with the @{_cls_name}.register_analysis decorator?"
        super().__init__(message)
