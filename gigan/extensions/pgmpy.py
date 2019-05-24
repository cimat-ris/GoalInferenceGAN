

class RandomChoice:
    """
    RandomChoice class is a hashable object that can be used along with pgmpy model classes as nodes.
    This allows us to add more functionality for our purposes.
    """

    def __init__(self, name) -> None:
        self.name = name
        self.likelihood = None
        self.proposed_model = None
        self.transition_model = None
        self.samples = []
        self.observed = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return "[ %s ]" % self.name

    def __repr__(self) -> str:
        return "[ %s ]" % self.name

    def __eq__(self, o) -> bool:
        return self.__class__ == o.__class__ and self.name == o.name

    def setunc(self) -> None:
        print("Hello")
