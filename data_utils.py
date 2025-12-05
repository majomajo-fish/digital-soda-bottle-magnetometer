#!/usr/bin/env python3
"""
data_utils.py

plot_pix_magdata.pyから分離した処理で共通で使用される関数をまとめたモジュール
"""

import os
import json
import pickle
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def get_param_value(line):
    """コメント部分（#以降）を除去して値を取得"""
    return line.split('#')[0].strip()


def load_parameters(paramfile):
    """パラメータファイルを読み込んで設定値を返す"""
    try:
        with open(paramfile, 'r') as f:
            lines = f.readlines()
        
        distance = float(get_param_value(lines[4]))
        pixlen = float(get_param_value(lines[5]))
        datadir = get_param_value(lines[6])  # 7行目
        
        # 時間平均幅パラメータを読み込み（13行目、デフォルトは1分）
        if len(lines) > 12 and get_param_value(lines[12]).strip():
            try:
                window_size = int(get_param_value(lines[12]))
            except ValueError:
                window_size = 1
        else:
            window_size = 1
        
        return {
            'distance': distance,
            'pixlen': pixlen,
            'datadir': datadir,
            'window_size': window_size
        }
    except Exception as e:
        raise Exception(f"パラメータファイル読み込みエラー: {e}")


def get_kak_cache_path():
    """KAKデータキャッシュファイルのパスを取得"""
    cache_dir = os.path.expanduser("~/.cache/plot_pix_magdata")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "kak_data_cache.pkl")


def is_kak_cache_valid(cache_path, max_age_minutes=5):
    """KAKデータキャッシュが有効かチェック（5分以内）"""
    if not os.path.exists(cache_path):
        return False
    
    cache_mtime = os.path.getmtime(cache_path)
    current_time = time.time()
    age_minutes = (current_time - cache_mtime) / 60
    
    return age_minutes < max_age_minutes


def load_kak_cache(cache_path):
    """KAKデータキャッシュを読み込み"""
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data.get('times', []), cache_data.get('values', []), cache_data.get('timestamp', None)
    except Exception as e:
        print(f"  → KAKキャッシュ読み込みエラー: {e}")
        return [], [], None


def save_kak_cache(cache_path, times, values):
    """KAKデータをキャッシュに保存"""
    try:
        cache_data = {
            'times': times,
            'values': values,
            'timestamp': datetime.now().isoformat()
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  → KAKデータをキャッシュに保存: {len(values)}ポイント")
    except Exception as e:
        print(f"  → KAKキャッシュ保存エラー: {e}")


def clear_kak_cache():
    """KAKキャッシュファイルを削除"""
    cache_path = get_kak_cache_path()
    try:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"KAKキャッシュを削除しました: {cache_path}")
            return True
        else:
            print("KAKキャッシュファイルは存在しません")
            return False
    except Exception as e:
        print(f"KAKキャッシュ削除エラー: {e}")
        return False


def load_observation_data(fname):
    """観測データを読み込む関数"""
    
    # データ形式を検出する
    with open(fname, 'r') as f:
        first_line = f.readline().strip()
        # カンマの数でデータ形式を判断
        comma_count = first_line.count(',')

    # 時刻、x, y形式の場合
    if comma_count == 2:
        # カラム名：時刻、x、y
        columns = ['time', 'x', 'y']
        
        # データ読み込み
        df = pd.read_csv(fname, names=columns, sep=',')
        
        # 時系列データとして処理
        # 時刻をUTに変換 (JSTからUTへの変換: -9時間)
        times = [datetime.strptime(row['time'].strip(), '%Y/%m/%d %H:%M:%S') - timedelta(hours=9) for _, row in df.iterrows()]
        x_values = [float(row['x']) for _, row in df.iterrows()]
        y_values = [float(row['y']) for _, row in df.iterrows()]
        
        return {
            'times': times,
            'x': x_values,
            'y': y_values,
            'format': 'xy',
            'data_points': len(times)
        }
        
    # 従来の形式（時刻、x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6）
    else:
        # カラム名
        columns = ['time', '05s', '15s', '25s', '35s', '45s', '55s','y05s', 'y15s', 'y25s', 'y35s', 'y45s', 'y55s']
        
        # データ読み込み
        df = pd.read_csv(fname, names=columns, sep=',')
        
        # 時系列データを展開
        times = []
        x_values = []
        y_values = []
        
        for _, row in df.iterrows():
            # 時刻をUTに変換 (JSTからUTへの変換: -9時間)
            t0 = datetime.strptime(row['time'].strip(), '%Y/%m/%d %H:%M:%S') - timedelta(hours=9)
            
            # x方向のデータ
            for i, col in enumerate(columns[1:7]):  # x方向の6列
                times.append(t0 + timedelta(seconds=10*i))
                x_values.append(float(row[col]))
            
            # y方向のデータ
            for i, col in enumerate(columns[7:13]):  # y方向の6列
                times.append(t0 + timedelta(seconds=10*i))
                y_values.append(float(row[col]))
        
        return {
            'times': times,
            'x': x_values,
            'y': y_values,
            'format': 'multi_time',
            'data_points': len(times)
        }


def calculate_declination(x_values, distance, pixlen):
    """偏角を計算"""
    # x方向のピクセル位置から偏角を計算
    position = np.array(x_values) * pixlen
    declination_rad = np.arctan(position / distance)  # ラジアン
    declination = np.rad2deg(declination_rad) * 0.5 * 60.  # arcmin
    
    # 相対偏角（最後の値を基準）
    declination_relative = [d - declination[-1] for d in declination]
    
    return declination_relative


def calculate_averages(times, values, window_size):
    """指定した時間窓で移動平均を計算"""
    # timeとvaluesをDataFrameに変換して時間ベースの処理をしやすくする
    df = pd.DataFrame({'time': times, 'value': values})
    df.set_index('time', inplace=True)
    
    # window_size分間隔でリサンプリング
    averages = []
    avg_times = []
    std_values = []
    
    resampled = df.resample(f'{window_size}min')
    for name, group in resampled:
        if not group.empty:
            # 各ウィンドウの平均値と標準偏差を計算
            averages.append(group['value'].mean())
            std_values.append(group['value'].std())
            avg_times.append(name)
    
    return {
        'times': avg_times,
        'means': averages,
        'stds': std_values,
        'window_size': window_size
    }


def detect_spots(datadir):
    """スポットディレクトリを検出する（単一ファイルも含む）"""
    data_dir = os.path.join(datadir, 'data')
    spots = []
    
    if not os.path.exists(data_dir):
        return spots
    
    # p1, p2, p3, ... ディレクトリを検索
    i = 1
    while True:
        spot_dir = os.path.join(data_dir, f'p{i}')
        spot_file = os.path.join(spot_dir, 'pix_magdata.txt')
        
        if os.path.exists(spot_file):
            spots.append({
                'id': i,
                'dir': spot_dir,
                'file': spot_file
            })
            i += 1
        else:
            break
    
    # スポットディレクトリが見つからなかった場合、単一ファイルをp1として扱う
    if not spots:
        single_file = find_observation_file(datadir)
        if single_file:
            # p1ディレクトリを作成
            p1_dir = os.path.join(data_dir, 'p1')
            os.makedirs(p1_dir, exist_ok=True)
            
            # ファイルをp1ディレクトリにコピー（または移動）
            import shutil
            p1_file = os.path.join(p1_dir, 'pix_magdata.txt')
            if not os.path.exists(p1_file):
                shutil.copy2(single_file, p1_file)
            
            spots.append({
                'id': 1,
                'dir': p1_dir,
                'file': p1_file
            })
    
    return spots


def find_observation_file(datadir):
    """観測データファイルを検索"""
    file1 = os.path.join(datadir, 'data', 'pix_magdata.txt')
    file2 = os.path.join(datadir, 'data', 'pix_magdata_continue.txt')
    
    if os.path.exists(file1):
        return file1
    elif os.path.exists(file2):
        return file2
    else:
        return None


def load_json_data(filepath):
    """JSONファイルを読み込み、時刻文字列をdatetimeオブジェクトに変換"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 時刻文字列をdatetimeオブジェクトに変換する関数
        def convert_times(data_dict):
            if 'times' in data_dict:
                converted_times = []
                for t in data_dict['times']:
                    if isinstance(t, str):
                        # ISO形式の時刻文字列をdatetimeに変換
                        if 'Z' in t:
                            converted_times.append(datetime.fromisoformat(t.replace('Z', '+00:00')))
                        else:
                            converted_times.append(datetime.fromisoformat(t))
                    else:
                        converted_times.append(t)
                data_dict['times'] = converted_times
            return data_dict
        
        # ネストされた辞書内の時刻も変換
        def convert_nested_times(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, dict):
                        obj[key] = convert_times(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                convert_times(item)
            return obj
        
        # トップレベルの時刻を変換
        data = convert_times(data)
        
        # ネストされた時刻を変換
        data = convert_nested_times(data)
        
        return data
    except Exception as e:
        print(f"JSONデータ読み込みエラー: {e}")
        return {}


def save_json_data(data_dict, output_path):
    """データをJSONファイルに保存（datetimeオブジェクトを文字列に変換）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # datetimeオブジェクトを文字列に変換する関数
    def convert_datetime_to_string(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    if value and hasattr(value[0], 'isoformat'):
                        # datetimeオブジェクトのリスト
                        obj[key] = [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in value]
                    else:
                        # ネストされたリストをチェック
                        obj[key] = [convert_datetime_to_string(item) if isinstance(item, dict) else item for item in value]
                elif isinstance(value, dict):
                    obj[key] = convert_datetime_to_string(value)
                elif hasattr(value, 'isoformat'):
                    obj[key] = value.isoformat()
        return obj
    
    # データ変換
    converted_data = convert_datetime_to_string(data_dict.copy())
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"JSONデータ保存エラー: {e}")
        return False


def estimate_time_range_from_observation_data(datadir):
    """観測データから時間範囲を推定"""
    # 観測データファイルを確認
    obs_file = find_observation_file(datadir)
    
    if obs_file:
        try:
            # 観測データファイルから時間範囲を取得
            with open(obs_file, 'r') as f:
                lines = f.readlines()
            
            # 最初と最後の行から時間を取得
            first_line = lines[0].strip().split(',')[0]
            last_line = lines[-1].strip().split(',')[0]
            
            # 時刻をUTに変換 (JSTからUTへの変換: -9時間)
            start_time = datetime.strptime(first_line, '%Y/%m/%d %H:%M:%S') - timedelta(hours=9)
            end_time = datetime.strptime(last_line, '%Y/%m/%d %H:%M:%S') - timedelta(hours=9)
            
            # 前後30分の余裕を持たせる
            start_time -= timedelta(minutes=30)
            end_time += timedelta(minutes=30)
            
            print(f"観測データから時間範囲を推定: {start_time} - {end_time}")
            return start_time, end_time
            
        except Exception as e:
            print(f"観測データからの時間範囲推定に失敗: {e}")
    
    # デフォルト: 現在時刻から24時間前
    KAK_DELAY_HOURS = 12
    end_time = datetime.now() - timedelta(hours=KAK_DELAY_HOURS + 1)
    start_time = end_time - timedelta(hours=24)
    
    print(f"デフォルト時間範囲を使用: {start_time} - {end_time}")
    return start_time, end_time


def filter_time_range(times, values, start_time, end_time):
    """指定した時間範囲でデータをフィルタリング"""
    filtered_times = []
    filtered_values = []
    
    # start_time, end_timeをdatetimeに変換し、offset-naiveにする
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time).replace(tzinfo=None)
    elif hasattr(start_time, 'tz_localize'):
        start_time = start_time.replace(tzinfo=None)
        
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time).replace(tzinfo=None)
    elif hasattr(end_time, 'tz_localize'):
        end_time = end_time.replace(tzinfo=None)
    
    for i, t in enumerate(times):
        # 時刻の型を統一（offset-naiveに変換）
        if isinstance(t, str):
            t_naive = pd.to_datetime(t).replace(tzinfo=None)
        elif hasattr(t, 'tz_localize'):
            t_naive = t.replace(tzinfo=None)
        elif hasattr(t, 'replace') and hasattr(t, 'tzinfo'):
            t_naive = t.replace(tzinfo=None)
        else:
            t_naive = pd.to_datetime(t).replace(tzinfo=None)
        
        if start_time <= t_naive <= end_time:
            filtered_times.append(t)
            filtered_values.append(values[i])
    
    return filtered_times, filtered_values


def apply_3rd_order_fit(times, values, stds=None):
    """3次多項式フィッティングを適用"""
    if len(times) < 4:  # 3次多項式には最低4点必要
        return values, [0] * len(values)
    
    # 数値的安定性のために時間をUNIX時間に変換し、開始時間からの経過秒数を計算
    start_timestamp = pd.to_datetime(times[0]).timestamp()
    timestamps = [(pd.to_datetime(t).timestamp() - start_timestamp) for t in times]
    
    # 重みを計算（標準偏差の逆数、標準偏差が0の場合は小さな値を設定）
    weights = None
    if stds:
        weights = []
        for std in stds:
            if std > 0:
                weights.append(1.0 / (std + 0.01))  # 標準偏差の逆数を重みに
            else:
                weights.append(1.0)  # 標準偏差が0の場合は等重み
    
    try:
        # 3次関数フィッティング
        poly_coefs = np.polyfit(timestamps, values, 3, w=weights)
        poly_fit = np.poly1d(poly_coefs)
        
        # フィッティング曲線の値を計算
        fit_values = [poly_fit(t) for t in timestamps]
        
        # 残差計算 (元データ - フィッティング曲線)
        residuals = [v - fit_values[i] for i, v in enumerate(values)]
        
        return fit_values, residuals
        
    except np.linalg.LinAlgError:
        print("警告: 3次関数フィッティングが収束しませんでした。1次関数にフォールバックします。")
        try:
            # 1次関数にフォールバック
            poly_coefs = np.polyfit(timestamps, values, 1, w=weights)
            poly_fit = np.poly1d(poly_coefs)
            
            fit_values = [poly_fit(t) for t in timestamps]
            residuals = [v - fit_values[i] for i, v in enumerate(values)]
            
            return fit_values, residuals
        except:
            print("警告: 1次関数フィッティングも失敗しました。フィッティングをスキップします。")
            return values, [0] * len(values)