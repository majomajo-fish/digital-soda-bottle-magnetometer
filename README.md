# レーザースポット位置記録プログラム (rec_data.py)

## 概要

このプログラムは、カメラを使用してレーザースポットの位置を連続的に記録するためのツールです。主に地磁気観測実験において、サスペンションマグネトメーターのミラーに反射したレーザースポットの位置変化を測定するために使用されます。

## 機能

- **リアルタイム画像取得**: カメラから連続的に画像を取得
- **レーザースポット位置解析**: 画像の赤チャンネルからスポット中心位置を計算
- **複数スポット対応**: 最大6つのスポットを同時に追跡可能
- **リアルタイム表示モード**: カメラ位置調整用のリアルタイムプレビュー機能
- **自動c_y調整**: スポットのy座標を自動追跡するモード
- **パラメータ動的変更**: 実行中にパラメータファイルを変更することで設定を反映

## 必要な環境

### Python バージョン
- Python 3.8以上

### 必要なパッケージ
```bash
pip install opencv-python numpy pandas matplotlib
```

## 使い方

### 基本的な使い方

```bash
python rec_data.py <パラメータファイル>
```

### リアルタイム表示モード（カメラ位置調整用）

```bash
python rec_data.py <パラメータファイル> --realtime
# または
python rec_data.py <パラメータファイル> -r
```

### 例

```bash
# 通常モード（データ記録）
python rec_data.py params_init_a1.txt

# リアルタイム表示モード（カメラ調整用）
python rec_data.py params_init_a1.txt --realtime
```

## パラメータファイル

パラメータファイルは以下の形式で設定します。コメントは `#` で始めます。

```
250, 600      # Line 1: c_y - Y-coordinates for analysis (comma-separated for multiple spots, -1 for auto mode)
              # Line 2: (reserved)
              # Line 3: (reserved)
11            # Line 4: n_conv - Smoothing window width [pixels]
2150.0        # Line 5: distance - Distance from mirror to screen [mm] (for reference)
0.21          # Line 6: pixlen - Pixel size [mm/pixel] (for reference)
/path/to/data # Line 7: Base directory for output data
              # Line 8: (reserved)
0             # Line 9: Camera index (0 = built-in camera, 1 = external camera, etc.)
/path/to/temp # Line 10: Temporary image storage directory
              # Line 11: Archive image storage directory (optional)
5.0           # Line 12: Capture interval [seconds]
5             # Line 13: Time averaging width [minutes] (for reference)
10            # Line 14: File write frequency (write every N captures)
```

### パラメータの詳細

| 行番号 | パラメータ | 説明 | 例 |
|--------|------------|------|-----|
| 1 | c_y | 解析対象のY座標。複数の場合はカンマ区切り。-1で自動モード | `250, 600` |
| 4 | n_conv | 平滑化フィルタの幅（ピクセル数） | `11` |
| 5 | distance | ミラーからスクリーンまでの距離 [mm] | `2150.0` |
| 6 | pixlen | 1ピクセルあたりの長さ [mm/pixel] | `0.21` |
| 7 | base_dir | データ出力先のベースディレクトリ | `/path/to/data` |
| 9 | camera | カメラのインデックス番号 | `0` |
| 10 | temp_dir | 一時ファイルの保存先 | `/path/to/temp` |
| 11 | archive_dir | アーカイブ画像の保存先（省略可） | `/path/to/archive` |
| 12 | interval | 撮影間隔 [秒] | `5.0` |
| 14 | flush_freq | ファイル書き込み頻度（N回に1回書き込み） | `10` |

## 出力データ

### ディレクトリ構造

```
<base_dir>/
├── data/
│   ├── p1/
│   │   └── pix_magdata.txt    # スポット1のデータ
│   ├── p2/
│   │   └── pix_magdata.txt    # スポット2のデータ
│   └── ...
├── param/
│   └── YYYYMMDD_HHMMSS_params_init_a1.txt  # パラメータファイルのバックアップ
└── images/ (または指定されたarchive_dir)
    └── archived_images/
        └── snapshot-YYYYMMDD_HHMMSS.jpg    # 定期的に保存される画像
```

### データファイル形式

`pix_magdata.txt` の形式:

```
2025/12/04 10:30:00, 640, 250
2025/12/04 10:30:05, 641, 251
2025/12/04 10:30:10, 639, 249
```

各行は `日時, X座標 [pixel], Y座標 [pixel]` の形式です。

## 動作の詳細

### スポット位置検出アルゴリズム

1. カメラから画像を取得
2. 画像の赤チャンネル（R）を抽出
3. 指定されたy座標範囲（c_y ± 75ピクセル、自動モードでは±10ピクセル）を切り出し
4. x方向の強度分布を計算（y方向の平均）
5. 移動平均フィルタ（n_conv幅）で平滑化
6. 最大値の位置をスポット中心のx座標とする
7. x座標が決まったら、その列でy方向の強度分布から中心y座標を計算

### 自動c_yモード

- パラメータファイルのc_y値を `-1` に設定すると自動モードになります
- 検出されたy座標の中央値から自動的にc_yを更新します（100回ごと）
- 自動モードでは解析範囲が狭く（±10ピクセル）なり、より安定した追跡が可能です

### パラメータの動的変更

プログラム実行中にパラメータファイルを編集すると、以下の設定が自動的に反映されます：
- c_y値（スポット位置）
- n_conv（平滑化幅）
- 撮影間隔
- ファイル書き込み頻度

## トラブルシューティング

### カメラが認識されない

1. カメラが正しく接続されているか確認
2. 他のアプリケーションがカメラを使用していないか確認
3. カメラインデックスを変更してみる（0, 1, 2...）

### スポットが検出されない

1. `--realtime` オプションでカメラ画像を確認
2. c_y値がスポットのy座標付近にあるか確認
3. レーザーの明るさを調整
4. 部屋の照明を暗くする

### データが保存されない

1. base_dirのパスが正しいか確認
2. 書き込み権限があるか確認
3. ディスク容量に余裕があるか確認