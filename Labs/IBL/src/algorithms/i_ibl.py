import abc


class IIBL(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and
                callable(subclass.fit) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict))

    @abc.abstractmethod
    def fit(self, x_train, y_train):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x_pred):
        raise NotImplementedError
