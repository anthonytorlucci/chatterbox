"""
Base class for researcher agents.
"""
# standard
from abc import ABC

# third party

# langchain

# langgraph

# local

class ResearcherInterface(ABC):
    """Abstract base class defining the interface for researcher agents."""

    # all subclasses should assign a proper value to NAME; assigned to the
    # state "calling_agent" and used in some conditional edges.
    NAME = "ResearchInterface"

    @property
    def name(self) -> str:
        """
        Returns the name identifier for this researcher.

        Returns:
            str: The name of the researcher agent
        """
        return self.NAME
