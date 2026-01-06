from filter_base import FilterBase
import numpy as np
from scipy import signal


class LowpassFilter(FilterBase):
    def __init__(self, cutoff_frequency: float, order: int = 5):
        """
        Butterworth ローパスフィルターを初期化する

        Args:
            cutoff_frequency: カットオフ周波数（Hz）
            order: フィルターの次数（デフォルト: 5）
        """
        if cutoff_frequency <= 0:
            raise ValueError(
                f"カットオフ周波数は正の値である必要があります（渡された値: {cutoff_frequency}）"
            )
        if order <= 0:
            raise ValueError(f"次数は正の整数である必要があります（渡された値: {order}）")

        self.cutoff_frequency = cutoff_frequency
        self.order = order
        super().__init__()

    def apply(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        numpy配列にButterworthローパスフィルターを適用する

        Args:
            data: 入力音声データ（-1.0〜1.0の範囲 or int16）
            sample_rate: サンプルレート（Hz）

        Returns:
            フィルター適用後のnumpy配列（入力と同じ型）
        """
        # 空配列チェック
        if len(data) == 0:
            return np.array([])

        # ナイキスト周波数の確認
        nyquist_freq = sample_rate / 2.0
        if self.cutoff_frequency >= nyquist_freq:
            raise ValueError(
                f"カットオフ周波数（{self.cutoff_frequency} Hz）はナイキスト周波数（{nyquist_freq} Hz）未満である必要があります"
            )

        # データ型の保存
        original_dtype = data.dtype

        # データ型の正規化（int16 の場合）
        if np.issubdtype(data.dtype, np.integer):
            normalized_data = data.astype(np.float64) / 32768.0
        else:
            normalized_data = data.astype(np.float64)

        # 範囲を -1.0〜1.0 にクリップ
        normalized_data = np.clip(normalized_data, -1.0, 1.0)

        # Butterworth フィルター係数の計算
        normalized_cutoff = self.cutoff_frequency / nyquist_freq
        b, a = signal.butter(self.order, normalized_cutoff, btype="low", analog=False)

        # フィルター適用（双方向フィルタリングで位相歪みなし）
        filtered_data = signal.filtfilt(b, a, normalized_data)

        # 元のデータ型に戻す（int16の場合）
        if np.issubdtype(original_dtype, np.integer):
            filtered_data = np.clip(filtered_data, -1.0, 1.0)
            filtered_data = (filtered_data * 32768.0).astype(original_dtype)

        return filtered_data
