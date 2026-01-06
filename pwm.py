from wav import WAV
import numpy as np


class PWM(WAV):
    def __init__(self, data: np.ndarray, sample_rate: int, sample_length: int = 16):
        pwm_data = self._to_pwm(data, sample_length)
        super().__init__(pwm_data, sample_rate)

    def _to_pwm(self, data: np.ndarray, sample_length: int) -> np.ndarray:
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
        carrier_rate = max(1, len(data) // sample_length)
        t = np.arange(len(data))
        carrier = 2.0 * (t % carrier_rate) / carrier_rate - 1.0

        # PWM 信号生成: 音声信号 > キャリア信号 なら 1.0、そうでなければ -1.0
        pwm_signal = np.where(normalized_data > carrier, 1.0, -1.0)

        return pwm_signal

    def save(self, filename: str):
        """
        numpy配列をPWM WAVファイルとして保存する

        Args:
            filename: 出力ファイルパス
        """

        # 親クラスのsaveメソッドを呼び出して保存
        super().save(filename)
