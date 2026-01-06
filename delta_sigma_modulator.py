from filter_base import FilterBase
import numpy as np


class DeltaSigmaModulator(FilterBase):
    def __init__(self, oversampling_ratio: int = 8):
        """
        1次ΔΣ変調器を初期化する

        Args:
            oversampling_ratio: オーバーサンプリング比（デフォルト: 8）
        """
        if oversampling_ratio <= 0:
            raise ValueError(
                f"オーバーサンプリング比は正の整数である必要があります（渡された値: {oversampling_ratio}）"
            )
        if not isinstance(oversampling_ratio, int):
            raise TypeError(
                f"オーバーサンプリング比は整数である必要があります（渡された値の型: {type(oversampling_ratio)}）"
            )

        # 実用的な範囲の推奨値チェック（警告のみ）
        if oversampling_ratio < 16 or oversampling_ratio > 256:
            print(
                f"警告: オーバーサンプリング比 {oversampling_ratio} は一般的な範囲（16〜256）外です"
            )

        self.oversampling_ratio = oversampling_ratio
        self.integrator_state = 0.0  # 積分器状態の初期化
        super().__init__()

    def apply(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        numpy配列に1次ΔΣ変調を適用する

        Args:
            data: 入力音声データ（-1.0〜1.0の範囲 or int16）
            sample_rate: サンプルレート（Hz）

        Returns:
            ΔΣ変調後のnumpy配列（-1.0 または 1.0、元の長さと同じ）
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

        # オーバーサンプリング（線形補間）
        original_length = len(normalized_data)
        oversampled_length = original_length * self.oversampling_ratio

        # np.interp を使用した線形補間
        original_indices = np.arange(original_length)
        oversampled_indices = np.linspace(
            0, original_length - 1, oversampled_length
        )
        oversampled_data = np.interp(
            oversampled_indices, original_indices, normalized_data
        )

        # ΔΣ変調処理
        modulated_data = self._delta_sigma_modulate(oversampled_data)

        # デシメーション（元の長さに戻す）
        # 単純な間引き（オーバーサンプリング比ごとにサンプリング）
        decimated_data = modulated_data[:: self.oversampling_ratio]

        # 長さの調整（端数処理）
        if len(decimated_data) > original_length:
            decimated_data = decimated_data[:original_length]
        elif len(decimated_data) < original_length:
            # パディング（最後の値で埋める）
            padding = np.full(
                original_length - len(decimated_data), decimated_data[-1]
            )
            decimated_data = np.concatenate([decimated_data, padding])

        return decimated_data

    def _delta_sigma_modulate(self, data: np.ndarray) -> np.ndarray:
        """
        1次ΔΣ変調のコアロジック

        Args:
            data: オーバーサンプリングされた入力データ

        Returns:
            1ビット出力（-1.0 または 1.0）
        """
        output = np.zeros_like(data)
        integrator = self.integrator_state  # 前回の状態を引き継ぐ

        for i in range(len(data)):
            # 誤差を計算（入力 - 前回の出力）
            if i == 0:
                error = data[i] - 0.0  # 初回は0からスタート
            else:
                error = data[i] - output[i - 1]

            # 積分器に誤差を蓄積
            integrator += error

            # 量子化（符号関数）
            output[i] = 1.0 if integrator >= 0.0 else -1.0

        # 次回のために積分器状態を保存
        self.integrator_state = integrator

        return output

    def reset(self):
        """積分器状態をリセット（連続した音声データ間での使用時）"""
        self.integrator_state = 0.0
