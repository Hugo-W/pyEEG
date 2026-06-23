"""
Decorators
"""
import warnings
from functools import wraps
from typing import Callable, Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def check_type(func: F) -> F:
    """
    Decorator to check the type of the first argument of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if not args:
            raise ValueError("No arguments provided")
        if not isinstance(args[0], (list, tuple)):
            raise TypeError(f"First argument must be a list or tuple, got {type(args[0])}")
        return func(*args, **kwargs)
    return wrapper

def deprecated_warning(*arg_names):
    """
    Decorator to mark the usage of specific arguments as deprecated.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for arg_name in arg_names:
                if arg_name in kwargs:
                    warnings.warn(
                        f"The {arg_name} argument is deprecated and will be removed in future versions. "
                        f"See {func.__name__} docstring for more info.",
                        DeprecationWarning,
                        stacklevel=2
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

