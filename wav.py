from filter_base import FilterBase
from viz_base import VizBase
import numpy as np
from scipy.io import wavfile


class WAV:
    def __init__(self, sample_rate: int):
        if sample_rate <= 0:
            raise ValueError(
                f"サンプルレートは正の値である必要があります（渡された値: {sample_rate}）"
            )

        self.sample_rate = sample_rate

        self.filters: list[FilterBase] = []
        self.data: np.ndarray = None

    def add_filter(self, filter: FilterBase) -> "WAV":
        """
        WAVデータにフィルタを追加する

        Args:
            filter: FilterBaseを継承したフィルタのインスタンス
        """
        self.filters.append(filter)
        return self

    def add_data(self, data: np.ndarray) -> "WAV":
        """
        WAVデータを追加し、登録されたフィルタを順に適用する

        Args:
            data: 1次元numpy配列（-1.0〜1.0のfloat or int16）
        """
        # フィルタを順に適用
        for filter in self.filters:
            data = filter.apply(data, self.sample_rate)

        self.data = data
        return self

    def save(self, filename: str):
        """
        numpy配列をWAVファイルとして保存する

        Args:
            data: 1次元numpy配列（-1.0〜1.0のfloat or int16）
            filename: 出力ファイルパス
            sample_rate: サンプルレート
        """

        # データ型の変換
        if np.issubdtype(self.data.dtype, np.floating):
            # float型の場合、-1.0〜1.0の範囲をint16に変換
            data = np.clip(self.data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
        elif not np.issubdtype(self.data.dtype, np.integer):
            # float型でもint型でもない場合はエラー
            raise ValueError(f"サポートされていないデータ型: {self.data.dtype}")

        # WAVファイル書き込み
        wavfile.write(filename, self.sample_rate, data)

    def visualize(self, visualizer: VizBase):
        """
        PWM信号を指定されたビジュアライザで可視化する

        Args:
            visualizer: VizBaseを継承したビジュアライザのインスタンス
        """
        visualizer.render(self.data, self.sample_rate)
