from abc import ABC, abstractmethod


class Term(ABC):
    @abstractmethod
    def __add__(self, other: "Term"):
        pass


class s(Term):
    def __add__(self, other: "Term"):
        pass
