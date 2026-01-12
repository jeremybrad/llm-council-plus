"""Job registry for Night Shift runner.

Maintains a registry of available job classes that can be executed by the
Night Shift runner. Jobs register themselves using the @register_job decorator.
"""

from typing import Dict, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .jobs.base import BaseJob

# Registry of available jobs
JOB_REGISTRY: Dict[str, Type["BaseJob"]] = {}


def register_job(name: str):
    """Decorator to register a job class.

    Usage:
        @register_job("my_job")
        class MyJob(BaseJob):
            ...
    """
    def decorator(cls: Type["BaseJob"]):
        JOB_REGISTRY[name] = cls
        return cls
    return decorator


def get_available_jobs() -> list[str]:
    """Get list of available job names."""
    return list(JOB_REGISTRY.keys())


def get_job_class(name: str) -> Optional[Type["BaseJob"]]:
    """Get job class by name."""
    return JOB_REGISTRY.get(name)
