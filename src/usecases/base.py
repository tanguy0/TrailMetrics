from abc import ABC, abstractmethod


class UseCase(ABC):
    """Base class for a usecase: orchestrates domain objects to deliver one user-facing task."""

    @abstractmethod
    def execute(self, *args, **kwargs):
        ...
