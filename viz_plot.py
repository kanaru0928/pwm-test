import numpy as np
import matplotlib.pyplot as plt
from viz_base import VizBase


class VizPlot(VizBase):
    def _plot(self, data: np.ndarray, sample_rate: int):
        """波形描画の共通処理"""
        # 時間軸を計算
        t = np.arange(len(data)) / sample_rate
        # 図の設定
        plt.figure(figsize=(10, 4))
        plt.plot(t, data)
        plt.xlabel("time [s]")
        plt.ylabel("amplitude")
        plt.title("Audio Waveform")
        plt.grid(True)

    def render(self, data: np.ndarray, sample_rate: int):
        """波形を画面に表示"""
        self._plot(data, sample_rate)
        plt.show()

    def save(self, data: np.ndarray, sample_rate: int, filepath: str):
        """
        波形を画像ファイルとして保存

        :param filepath: 保存先のファイルパス
        :type filepath: str
        """
        self._plot(data, sample_rate)
        plt.savefig(filepath)
        plt.close()
