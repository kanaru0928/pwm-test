from wav import WAV
from viz_plot import VizPlot
from pwm import PWM
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
    wav = WAV(data=audio_data, sample_rate=sample_rate)
    wav.save(output_file)
    print(f"WAVファイルを保存: {output_file}")

    visualizer = VizPlot()
    print("WAV波形を表示...")
    wav.visualize(visualizer)

    # PWM変換のデモ
    print("\nPWM変換のデモ")
    pwm = PWM(data=audio_data, sample_rate=sample_rate)
    pwm_output_file = "output_pwm.wav"
    pwm.save(pwm_output_file)
    print(f"PWMファイルを保存: {pwm_output_file}")

    # PWM波形の可視化
    print("PWM波形を表示...")
    pwm.visualize(VizPlot())


if __name__ == "__main__":
    main()
