# Conv-TasNet

このプロジェクトは、音声分離モデルであるConv-TasNetの実装です。
話者分離タスクを目的として、単一チャネルおよび多チャネルのConv-TasNetが実装されています。

## 動作環境 (Requirements)

このプロジェクトには、Python 3.xと以下のライブラリが必要です。

*   PyTorch
*   NumPy
*   SciPy
*   ... (その他、必要なライブラリを追記してください)

セットアップを簡単にするために、`requirements.txt`ファイルを作成することをお勧めします。お使いの環境で以下のコマンドを実行すると生成できます。
```bash
pip freeze > requirements.txt
```

## セットアップ (Installation)

1.  **リポジトリをクローン**
    ```bash
    git clone https://github.com/your-username/Conv-TasNet.git
    cd Conv-TasNet
    ```

2.  **(推奨) 仮想環境の作成と有効化**
    ```bash
    python -m venv venv
    # Windowsの場合
    .\venv\Scripts\activate
    # macOS/Linuxの場合
    source venv/bin/activate
    ```

3.  **必要なライブラリをインストール**
    `requirements.txt`がある場合:
    ```bash
    pip install -r requirements.txt
    ```
    もしなければ、「動作環境」セクションに記載されているライブラリを手動でインストールしてください。

## 使い方 (Usage)

以下は、学習および評価スクリプトの実行例です。プレースホルダー引数 (`<...>`) は、実際のパスやパラメータに置き換えてください。

### 学習 (Training)

*   **単一チャネル Conv-TasNet**
    ```bash
    python src/ConvTasNet_train.py --train_data <学習データへのパス> --valid_data <検証データへのパス> --model_save_path <モデルの保存先パス>
    ```

*   **多チャネル Conv-TasNet**
    ```bash
    python src/Multi_Channel_ConvTasNet_train.py --train_data <学習データへのパス> --valid_data <検証データへのパス> --model_save_path <モデルの保存先パス>
    ```

### 評価 (Evaluation)

*   **単一チャネル Conv-TasNet**
    ```bash
    python src/ConvTasNet_test.py --model_path <学習済みモデルへのパス> --test_data <テストデータへのパス>
    ```

*   **多チャネル Conv-TasNet**
    ```bash
    python src/Multi_Channel_ConvTasNet_test.py --model_path <学習済みモデルへのパス> --test_data <テストデータへのパス>
    ```

## ディレクトリ構成 (Directory Structure)

```
Conv-TasNet/
├── src/                # ソースコードディレクトリ
│   ├── models/         # モデル構造の定義
│   ├── data/           # データ読み込みや前処理スクリプト
│   ├── utils/          # 共通で利用する関数やクラス
│   ├── evaluation/     # モデル評価用スクリプト
│   ├── losses.py       # 損失関数の実装
│   ├── datasetClass.py # データセットクラスの定義
│   ├── ConvTasNet_train.py # 単一チャネルConv-TasNetの学習スクリプト
│   ├── ConvTasNet_test.py  # 単一チャネルConv-TasNetの評価スクリプト
│   ├── Multi_Channel_ConvTasNet_train.py # 多チャネルConv-TasNetの学習スクリプト
│   ├── Multi_Channel_ConvTasNet_test.py  # 多チャネルConv-TasNetの評価スクリプト
│   └── ...
├── config/             # 設定ファイル (ハイパーパラメータなど)
├── scripts/            # ヘルパースクリプト
├── mymodule/           # カスタムPythonモジュール
├── Document/           # プロジェクト関連ドキュメント
│   └── 改善案.md
├── README.md           # このファイル
└── .gitignore          # Gitの追跡から除外するファイルやディレクトリ
```
