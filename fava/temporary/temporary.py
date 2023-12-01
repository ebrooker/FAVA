
from typing import Any, Dict

from ._exceptions import *


class Temporary:

    __meshes: Dict[str,Any] = {}

    #--------------------------
    # Register and load meshes
    @classmethod
    def register_mesh(cls):
        def decorator(mesh_cls):
            cls.__meshes[mesh_cls.__name__] = mesh_cls
            return mesh_cls
        return decorator

    # @classmethod
    # def load(cls, mesh_name: str, **kwargs):
    #     if mesh_name not in cls.__meshes:
    #         raise InvalidMeshError(mesh_name)
    #     return cls.__meshes[mesh_name](**kwargs)
    @classmethod
    def load(cls):
        pass
    #--------------------------
    

    #--------------------------
    # Register analysis methods
    @classmethod
    def register_analysis(cls, overwrite=False):
        def decorator(analysis_func):
            if not callable(analysis_func):
                raise NotCallableError(analysis_func)
            name = analysis_func.__name__
            if not hasattr(cls, name) or overwrite:
                setattr(cls, name, analysis_func)
            return analysis_func
        return decorator
    #--------------------------


    #--------------------------
    # Register plot methods
    @classmethod
    def register_plot(cls, overwrite=False):
        def decorator(plot_func):
            if not callable(plot_func):
                raise NotCallableError(plot_func)
            name = plot_func.__name__
            if not hasattr(cls, name) or overwrite:
                setattr(cls, name, plot_func)
            return plot_func
        return decorator
    #--------------------------