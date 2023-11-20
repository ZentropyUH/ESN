from enum import Enum


class Choice(str, Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class EnumModel(Choice):
    ESN = 'ESN'
    PESN = 'Parallel-ESN'


class EnumInputInitializer(Choice):
    IM = 'InputMatrix'
    RU = 'RandomUniform'


class EnumInputBiasInitializer(Choice):
    IM = 'InputMatrix'
    RU = 'RandomUniform'
    N = 'None'


class EnumReservoirActivation(Choice):
    Tanh = 'tanh'
    Relu = 'relu'
    S = 'sigmoid'
    I = 'identity'


class EnumReservoirInitializer(Choice):
    RNX = 'RegularNX'
    ER = 'ErdosRenyi'
    WSNX = 'WattsStrogatzNX'


class EnumReadoutLayer(Choice):
    Linear = 'linear'
    SGD = 'sgd'
    MLP = 'mlp'


class EnumForecastMethod(Choice):
    Classic = 'classic'
    Section = 'section'


class EnumPlotType(Choice):
    Linear = 'linear'
    Contourf = 'contourf'
    RMSE = 'rmse',
    Video = 'video'

class EnumBinMethod(Choice):
    Mean = 'mean'
    Median = 'median'