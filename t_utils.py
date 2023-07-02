from enum import Enum

class Choice(str, Enum):
    ...



class EModel(Choice):
    ESN = 'ESN'
    PESN = 'Parallel-ESN'
    R = 'Reservoir_to_be_implemented'



class EInputInitializer(Choice):
    IM = 'InputMatrix'
    RU = 'RandomUniform'


class InputBiasInitializer(Choice):
    IM = 'InputMatrix'
    RU = 'RandomUniform'
    N = 'None'


class ReservoirActivation(Choice):
    Tanh = 'tanh'
    Relu = 'relu'
    S = 'sigmoid'
    I = 'identity'


class ReservoirInitializer(Choice):
    RNX = 'RegularNX'
    ER = 'ErdosRenyi'
    WSNX = 'WattsStrogatzNX'


class ReadoutLayer(Choice):
    Linear = 'linear'
    SGD = 'sgd'
    MLP = 'mlp'


class ForecastMethod(Choice):
    Classic = 'classic'
    Section = 'section'


class PlotType(Choice):
    Linear = 'linear'
    Contourf = 'contourf'
    RMSE = 'rmse',
    Video = 'video'