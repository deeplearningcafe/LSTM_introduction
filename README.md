[Japanese/[English](README_EN.md)]
# 入門: Long Short-Term Memory（LSTM）ネットワーク

## 概要
このプロジェクトは、PyTorchを使用して単変量時系列データの予測にLSTM（Long Short-Term Memory）モデルを適用する入門的なサンプルです。主に株価の予測を対象としており、基本的なLSTMの理解と実装手順を提供します。



## 問題設定

時系列データのモデリングや予測において、リカレントニューラルネットワーク（RNN）は有力なツールとなっています。その中でも、LSTM（Long Short-Term Memory）は短期的および長期的な依存関係を学習する能力に優れ、広く使用されています。この記事では、LSTMの基本原理、PyTorchを用いた実装手順、そして実際のデータにLSTMを適用する方法に焦点を当てます。

## 前提条件
- python==3.11.5
- pytorch==2.0.1
- pytorch-cuda==11.8
- numpy==1.25.2
- matplotlib==3.8.0
- pandas==2.0.3


## プロジェクトの構成
- `generate_data.py`：データセットのロードおよび前処理を行うファイル。
- `training.py`：LSTMモデルのトレーニングおよび評価のためのファイル。
- `models.py`: LSTMモデルの定義と実装が含まれるファイル。


## トレーニング
`generate_data.py`のパスを変えて、自分が持てるデータセットを決定する。用いたデータセットは10年ほどのAmazonの株式投資でした。
その後tensorboardのコメントを自分の好みに変わって。ターミナルでこのコマンドーを実行すると学習が始まります。
```bash
python -m training.py
```


## データ処理

入力データ（過去の値）とターゲットデータ（将来の値）を含む、適切なサイズのウィンドウが生成されます。`self.sequence_length`は、モデルが考慮する過去のデータの数を指定します。
`self.future_time_steps`は、モデルが予測する未来のデータの数を指定します。

xは形状を `(batch_size, input_size, sequence_length)` に変更されます。yは、includeInputがTrueの場合は `(batch_size, self.sequence_length + future_time_steps)`、そうでない場合は `(batch_size, future_time_steps)` に変更されます。

# 学習ループ

LSTMモデルを訓練するためには、損失関数と最適化手法の選択が重要です。一般的な回帰タスクでは、平均二乗誤差（MSE）が使用されます。トレーニング中には、バックプロパゲーションと勾配降下法を使用してモデルを最適化します。


## メトリクス

訓練が終了したら、モデルの評価が行われます。予測と実際の値の比較には、RMSE（Root Mean Squared Error）、MAE（Mean Absolute Error）、R²（Coefficient of Determination）などのメトリクスが利用されます。そのメトリクスはこの関数によってもっと深い分析ができます。

このメソッドは、トレーニングまたは検証の各エポック後にログにメトリクスを記録するためのものです。以下は、このメソッドの主な機能と手順の概要です。コードでメトリクスの名前や計算方法が指摘されてる。

その後各メトリクスの平均値が取得され、ログメッセージが構築されます、ターミナルで読めるから学習してる時にとても便利だと思います。最後にTensorBoardにメトリクスの値が書き込まれます。各メトリクス（loss、RMSE、MAEなど）に対して、エポックごとに値が書き込まれます。


## モデル

LSTMは、通常のRNNが直面する勾配消失問題を解決するために開発されました。このネットワークは、ゲート（input gate、forget gate、output gate）を導入することで、長期的な記憶の保持と選択的な情報の取捨選択を可能にします。これにより、LSTMは長期的な依存関係を学習しやすくなります。

## 結論

トレーニングされたモデルを使用して、未来の株価動向を予測しました。予測の精度は、テストデータを使用して評価されました。我々はモデルの性能を評価するために、異なる評価指標を使用しました。

生憎、モデルは株式市場の価格予測が正確に出来ない、でもLSTMの仕組みや強さが分かるようになりました。他のモデルや特徴量エンジニアリングを行うなら、きっと精度を上がることが出来ます。

![推論グラフ](results.png)


詳細については、プロジェクトのファイルとコードを参照してください。ご質問や提案がある場合は、気軽にお知らせください。

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

zenn記事
[入門: Long Short-Term Memory（LSTM）ネットワーク](https://zenn.dev/fuwamoekissaten/articles/2b306e2c8f1871)


## References
https://github.com/pytorch/examples/blob/main/time_sequence_prediction/README.md


## LICENSE
このプロジェクトはMITライセンスの下でライセンスされています。詳細については[LICENSE.md](LICENSE)ファイルを参照してください。
