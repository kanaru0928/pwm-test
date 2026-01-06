from delta_sigma_modulator import DeltaSigmaModulator
from lowpass_filter import LowpassFilter
from pwm_filter import PWMFilter
from wav import WAV
from viz_plot import VizPlot
import numpy as np


def main():
    print("numpy配列をWAVファイルに変換するデモ")

    # サンプルデータ生成（440Hzサイン波、1秒間）
    sample_rate = 96000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t)

    # WAVファイルとして保存
    output_file = "output.wav"
    wav = WAV(sample_rate=sample_rate).add_data(audio_data)
    wav.save(output_file)
    print(f"WAVファイルを保存: {output_file}")

    visualizer = VizPlot()
    print("WAV波形を表示...")
    wav.visualize(visualizer)

    # PWM変換のデモ
    print("\nPWM変換のデモ")
    pwm_filter = DeltaSigmaModulator()
    pwm = WAV(sample_rate=sample_rate).add_filter(pwm_filter).add_data(audio_data)
    pwm_output_file = "output_pwm.wav"
    pwm.save(pwm_output_file)
    print(f"PWMファイルを保存: {pwm_output_file}")

    # PWM波形の可視化
    print("PWM波形を表示...")
    pwm.visualize(VizPlot())

    # ローパスフィルタ適用のデモ
    print("\nローパスフィルタ適用のデモ")
    lowpass_filter = LowpassFilter(cutoff_frequency=1000.0)
    lowpasss = (
        WAV(sample_rate=sample_rate)
        .add_filter(pwm_filter)
        .add_filter(lowpass_filter)
        .add_data(audio_data)
    )
    lowpass_output_file = "output_pwm_lowpass.wav"
    lowpasss.save(lowpass_output_file)
    print(f"ローパスフィルタ適用後のPWMファイルを保存: {lowpass_output_file}")
    print("ローパスフィルタ適用後のPWM波形を表示...")
    lowpasss.visualize(VizPlot())


if __name__ == "__main__":
    main()
