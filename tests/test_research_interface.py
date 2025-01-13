import pytest
from chatterbox.researcher_interface import ResearcherInterface

def test_name_property():
    """
    Verifies that the NAME property is correctly assigned and returned by the researcher interface.
    """
    researcher_interface = ResearcherInterface()
    assert researcher_interface.name == "ResearchInterface"

def test_name_property_subclassing():
    """
    Verifies that a subclass of ResearcherInterface can assign its own NAME property and it is correctly used by the researcher interface.
    """
    class MyResearcherInterface(ResearcherInterface):
        NAME = "MyResearcher"

    my_researcher_interface = MyResearcherInterface()
    assert my_researcher_interface.name == "MyResearcher"

def test_name_property_inheritance():
    """
    Verifies that a subclass of ResearcherInterface inherits the NAME property from its parent class.
    """
    class MySubclass(ResearcherInterface):
        pass

    my_subclass = MySubclass()
    assert my_subclass.name == "ResearchInterface"
