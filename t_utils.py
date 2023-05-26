from enum import Enum

class Choice(str, Enum):
    ...

class EInputInitializer(Choice):
    IM = 'InputMatrix'
    RU = 'RandomUniform'


class EModel(Choice):
    ESN = 'ESN'
    PESN = 'Parallel-ESN'
    R = 'Reservoir_to_be_implemented'

    