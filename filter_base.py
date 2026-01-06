import abc
import numpy as np

class FilterBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """データにフィルタを適用する抽象メソッド

        Args:
            data: フィルタを適用するnumpy配列
            sample_rate: サンプルレート

        Returns:
            フィルタが適用されたnumpy配列
        """
        pass
