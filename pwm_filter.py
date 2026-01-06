from filter_base import FilterBase
import numpy as np


class PWMFilter(FilterBase):
    def __init__(self, sample_length: int = 16):
        self.sample_length = sample_length
        super().__init__()

    def apply(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        numpy配列をPWM信号に変換する（単純な比較ベース PWM）

        Args:
            data: 入力音声データ（-1.0〜1.0の範囲）

        Returns:
            PWM信号のnumpy配列（-1.0 または 1.0）
        """
        # 空配列チェック
        if len(data) == 0:
            return np.array([])

        # データ型の正規化（int16 の場合）
        if np.issubdtype(data.dtype, np.integer):
            normalized_data = data.astype(np.float64) / 32768.0
        else:
            normalized_data = data.astype(np.float64)

        # 範囲を -1.0〜1.0 にクリップ
        normalized_data = np.clip(normalized_data, -1.0, 1.0)

        # のこぎり波キャリア信号を生成（-1.0〜1.0の範囲）
        carrier_rate = max(1, len(data) // self.sample_length)
        t = np.arange(len(data))
        carrier = 2.0 * (t % carrier_rate) / carrier_rate - 1.0

        # PWM 信号生成: 音声信号 > キャリア信号 なら 1.0、そうでなければ -1.0
        pwm_signal = np.where(normalized_data > carrier, 1.0, -1.0)

        return pwm_signal
