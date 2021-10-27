class EnvironmentNotFound(Exception):
    """Raised when the environments name
    isn't specified"""
    pass

class WrongNumberOfAgent(Exception):
    """Raised when the number of agent doesn't
    match the environment specification"""
    pass

class WrongNumberOfAction(Exception):
    """Raised when the number of action doesn't
    match the environment specification"""
    pass

class WrongNumberOfState(Exception):
    """Raised when the number of state doesn't
    match the environment specification"""
    pass


class RewardTypeNotFound(Exception):
    """Raised when the type of the reward isn't found
    (For PBeautyGame)"""
    pass

class WrongActionInputLength(Exception):
    """Raised when the the length of the
    input doesn't match the number of agents"""
