from viz_base import VizBase
import numpy as np
from scipy.io import wavfile


class WAV:
    def __init__(self, data: np.ndarray, sample_rate: int):
        # 入力検証
        if data.ndim != 1:
            raise ValueError(
                f"1次元配列のみ対応しています（渡された配列: {data.ndim}次元）"
            )

        if sample_rate <= 0:
            raise ValueError(
                f"サンプルレートは正の値である必要があります（渡された値: {sample_rate}）"
            )

        self.data = data
        self.sample_rate = sample_rate

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

