# -*- coding: utf-8 -*-

import os
import sys
import yaml
import time
import json
import shutil
import datetime
import requests
import subprocess
import pandas as pd

from google.cloud import storage
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def _exec_cmd(cmd, verbose=True):
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout_data, stderr_data = p.communicate()
    if verbose:
        if len(stderr_data) > 0:
            print(stderr_data.decode('utf8'))
            return False
        else:
            print(stdout_data.decode('utf8'))
            return stdout_data.decode('utf8')
    else:
        if len(stderr_data) > 0:
            return False
        else:
            return stdout_data.decode('utf8')


class ForecastFlow:

    def __init__(self):
        pass

    def setup(self, project_name, bucket_name, is_timeout=False):
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.is_timeout = is_timeout

        # authenticate
        auth = 'gcloud auth activate-service-account --key-file /content/credential.json'
        _exec_cmd(auth, verbose=False)

        check_bucket_exist = 'gsutil ls gs://{}/'.format(self.bucket_name)
        raw_string = _exec_cmd(check_bucket_exist, verbose=False)
        if raw_string is False:
            print('ユーザ名または認証キーに誤りがあります。')
            sys.exit()

        # listing projects already exists on gcs buckets
        get_projects_already_exist = "gsutil ls gs://{}/{}".format(self.bucket_name, self.project_name)
        raw_string = _exec_cmd(get_projects_already_exist, verbose=False)
        if self.is_timeout is False:
            if raw_string is not False:
                projects_already_exist = [obj_name.split('/')[-2] for obj_name in raw_string.split('\n')
                                          if len(obj_name) > 0]
                if self.project_name in projects_already_exist:
                    is_delete_remote = input('同じ名称のプロジェクトがリモートに存在します。\n'
                                             + '削除しますか？ 過去の実行結果をダウンロードする場合は削除しないでください [y/N]\n')
                    if is_delete_remote in ('y', 'Y', 'ｙ', 'Ｙ'):
                        delete_old_project = 'gsutil rm gs://{}/{}/*'.format(self.bucket_name, self.project_name)
                        _exec_cmd(delete_old_project, verbose=False)
                    else:
                        if self.is_timeout:
                            pass
                        else:
                            print('プロジェクト名を変更してリトライするか、結果のダウンロードを行ってください。')
                            sys.exit()
        elif self.is_timeout is True:
            if raw_string is False:
                print('プロジェクト名に誤りがあります。')
                sys.exit()

        # make base directory
        if self.is_timeout is False:
            if self.project_name in os.listdir('/content'):
                is_delete_local = input('同じ名称のプロジェクトがローカルに存在します。初期化しますか？ 初期化するとデータは全て消去されます。[y/N]\n')
                if is_delete_local in ('y', 'Y', 'ｙ', 'Ｙ'):
                    is_new = True
                    shutil.rmtree(self.project_name)
                    os.mkdir(os.path.join('/content', self.project_name))
                else:
                    is_new = False
            else:
                os.mkdir(os.path.join('/content', self.project_name))
                is_new = True

            self.dir_train = os.path.join('/content', self.project_name, 'train')
            self.dir_test = os.path.join('/content', self.project_name, 'test')
            self.dir_pred = os.path.join('/content', self.project_name, 'pred')

            if is_new:
                # make directory for train data
                os.mkdir(self.dir_train)
                # make directory for test data
                os.mkdir(self.dir_test)
                # make directory for pred data
                os.mkdir(self.dir_pred)
                print('アップロード用のディレクトリが生成されました')
                dir_structure = F"""
                {self.project_name}
                   |
                   +---- train ... [必須] 訓練用データ（ID＋特徴量＋正解）
                   |
                   +---- test ...  [任意] 精度検証用データ（ID＋特徴量＋正解）
                   |
                   +---- pred ...  [任意] 予測用データ（ID＋特徴量）

                """
                print(dir_structure)
                print('データをそれぞれのディレクトリにアップロードしてください。')

    def _stop_process(self, err_msg):
        print(err_msg)
        sys.exit()

    def _data_loader(self, partition):
        path = os.path.join('/content', self.project_name, partition)
        file_list = [os.path.join(path, fname) for fname in os.listdir(path)
                     if os.path.splitext(fname)[1] == '.csv']

        if len(file_list) == 0:
            raise FileNotFoundError

        for N, f in enumerate(file_list):
            try:
                if N == 0:
                    df = pd.read_csv(f)
                else:
                    df = pd.concat([df, pd.read_csv(f)], axis=0)
            except MemoryError:
                raise MemoryError

        return df

    def _column_validation(self, df, partition):

        if partition == 'train':
            id_err_msg1 = 'データエラー: 訓練用CSVデータに IDカラム "{}" が含まれていません。'
            id_err_msg2 = 'データエラー: 訓練用CSVデータの IDカラム "{}" に欠損値が含まれています。'
            target_err_msg1 = 'データエラー: 訓練用CSVデータに 正解カラム "{}" が含まれていません。'
            target_err_msg2 = 'データエラー: 訓練用CSVデータの 正解カラム "{}" に欠損値が含まれています。'
        elif partition == 'test':
            id_err_msg1 = 'データエラー: 精度検証用CSVデータに IDカラム "{}" が含まれていません。'
            id_err_msg2 = 'データエラー: 精度検証用CSVデータの IDカラム "{}" に欠損値が含まれています。'
            target_err_msg1 = 'データエラー: 精度検証用CSVデータに 正解カラム "{}" が含まれていません。'
            target_err_msg2 = 'データエラー: 精度検証用CSVデータの 正解カラム "{}" に欠損値が含まれています。'
        elif partition == 'pred':
            id_err_msg1 = 'データエラー: 予測用CSVデータに IDカラム "{}" が含まれていません。'
            id_err_msg2 = 'データエラー: 予測用CSVデータの IDカラム "{}" に欠損値が含まれています。'

        col = df.columns

        is_err = False

        # ID Validation
        for _id in self.ID_list:
            if not _id in col:
                is_err = True
                print(id_err_msg1.format(_id))
            else:
                null_count = df[_id].isnull().sum()
                if null_count > 0:
                    is_err = True
                    print(id_err_msg2.format(_id))

        if partition == 'pred':
            return is_err
        elif partition in ('train', 'test'):
            # Target Validation
            if self.Target not in col:
                print(target_err_msg1.format(self.Target))
            else:
                null_count = df[self.Target].isnull().sum()
                if null_count > 0:
                    is_err = True
                    print(target_err_msg2.format(self.Target))
            return is_err

    def _remove_datetime(self):

        exclude_columns = [self.Target]
        exclude_columns.extend(self.ID_list)
        col_features = [_col for _col in self.df_train.columns if _col not in exclude_columns]

        # train
        df_features_train = pd.DataFrame(data=[[_col, str(self.df_train[_col].dtype)] for _col in col_features],
                                         columns=['column', 'dtype_train'])
        df_dt_train = df_features_train.loc[df_features_train['dtype_train'].str.contains('datetime')]
        if len(df_dt_train) > 0:
            print('以下の訓練用データの特徴量は日時型であるため、エラーになります。')
            for _dt_col in df_dt_train.columns:
                print('  - {}'.format(_dt_col))
            print('以下の選択肢から、上記データの扱い方を決めてください')
            print(' 1. 訓練用データから削除する')
            print(' 2. ID にする')
            print(' 3. マニュアルで対処する\n')
            operate_id = input()

            if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                del self.df_train.loc[:, list(df_dt_train['column'])]
            elif operate_id in ('2', '２', 2, '二', 'に', '②', '弐'):
                self._stop_process('ID に上記のデータを追加してリトライしてください。')
            else:
                self._stop_process('データを見直し、修正してからリトライしてください。')
        # test
        df_features_test = pd.DataFrame(data=[[_col, str(self.df_test[_col].dtype)] for _col in col_features],
                                         columns=['column', 'dtype_test'])
        df_dt_test = df_features_test.loc[df_features_test['dtype_test'].str.contains('datetime')]
        if len(df_dt_test) > 0:
            print('以下の精度検証用データの特徴量は日時型であるため、エラーになります。')
            for _dt_col in df_dt_test.columns:
                print('  - {}'.format(_dt_col))
            print('以下の選択肢から、上記データの扱い方を決めてください')
            print(' 1. 精度検証用データから削除する')
            print(' 2. ID にする')
            print(' 3. マニュアルで対処する\n')
            operate_id = input()

            if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                del self.df_test.loc[:, list(df_dt_test['column'])]
            elif operate_id in ('2', '２', 2, '二', 'に', '②', '弐'):
                self._stop_process('ID に上記のデータを追加してリトライしてください。')
            else:
                self._stop_process('データを見直し、修正してからリトライしてください。')
        # pred
        if self.is_pred:
            df_features_pred = pd.DataFrame(data=[[_col, str(self.df_pred[_col].dtype)] for _col in col_features],
                                            columns=['column', 'dtype_pred'])
            df_dt_pred = df_features_pred.loc[df_features_pred['dtype_pred'].str.contains('datetime')]
            if len(df_dt_pred) > 0:
                print('以下の予測用データの特徴量は日時型であるため、エラーになります。')
                for _dt_col in df_dt_pred.columns:
                    print('  - {}'.format(_dt_col))
                print('以下の選択肢から、上記データの扱い方を決めてください')
                print(' 1. 予測用データから削除する')
                print(' 2. ID にする')
                print(' 3. マニュアルで対処する\n')
                operate_id = input()

                if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                    del self.df_pred.loc[:, list(df_dt_pred['column'])]
                elif operate_id in ('2', '２', 2, '二', 'に', '②', '弐'):
                    self._stop_process('ID に上記のデータを追加してリトライしてください。')
                else:
                    self._stop_process('データを見直し、修正してからリトライしてください。')

    def data_check(self):

        # == Load Data ==
        # train
        try:
            self.df_train = self._data_loader(partition='train')
        except FileNotFoundError:
            self._stop_process(
                '訓練用CSVデータが < {} > の中にありません。\n'.format(os.path.join(self.project_name, 'train'))
                + 'データをアップロードしてからリトライしてください。')
        except MemoryError:
            self._stop_process(
                '訓練用CSVデータの容量が許容量を超えています。\n'
                + 'ご不便おかけし申し訳ありませんが、データ容量を小さくしてからリトライしてください。')
        # test
        try:
            self.df_test = self._data_loader(partition='test')
        except FileNotFoundError:
                print('精度検証用CSVデータが < {} > の中にありません。\n'.format(os.path.join(self.project_name, 'test')))
                print('以下の選択肢から精度検証用データの準備方法を決めてください。\n')
                print('=' * 40)
                print(' 1. マニュアルでアップロードする\n')
                print(' 2. 訓練データをランダムに分割する\n')
                operate_id = input()

                if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                    self._stop_process('データをアップロードしてからリトライしてください。')
                elif operate_id in ('2', '２', 2, '二', 'に', '②', '弐'):
                    print('訓練データから精度検証データに移行する割合を入力してください [ 0.0 ～ 1.0 の範囲 ]\n')
                    test_size = input()
                    try:
                        test_size = float(test_size)
                    except:
                        self._stop_process('数値で入力してください。')
                    if test_size < 0.0 or test_size >= 1.0:
                        self._stop_process('0 から 1 の範囲で入力してください。')

                self.df_train, self.df_test = train_test_split(self.df_train,
                                                               train_size=1 - test_size,
                                                               test_size=test_size)
                print('分割に成功しました。\n')

        except MemoryError:
            self._stop_process(
                '精度検証用CSVデータの容量が許容量を超えています。\n'
                + 'ご不便おかけし申し訳ありませんが、データ容量を小さくしてからリトライしてください。')
        # pred
        try:
            self.df_pred = self._data_loader(partition='pred')
        except FileNotFoundError:
            is_not_pred = input('予測用CSVデータが < {} > の中にありません。\n'.format(os.path.join(self.project_name, 'pred'))
                                + 'このまま進めて よろしいですか？[y/N]\n')
            if is_not_pred in ('y', 'Y', 'ｙ', 'Ｙ'):
                self.is_pred = False
            else:
                self.is_pred = True
                self._stop_process('データをアップロードしてからリトライしてください。')
        except MemoryError:
            self._stop_process(
                '予測用データの容量が許容量を超えています。\n'
                + 'ご不便おかけし申し訳ありませんが、データ容量を小さくしてからリトライしてください。')
        else:
            self.is_pred = True

        # check column number
        numCols_train = self.df_train.shape[1]
        numCols_test = self.df_test.shape[1]
        if numCols_train != numCols_test:
            print('訓練用と精度検証用データで、カラム数が異なります。')
            print('訓練用:     {}'.format(numCols_train))
            print('精度検証用: {}'.format(numCols_test))

        numRows_train = len(self.df_train)
        print('訓練用データ:')
        print('    行数:     {}'.format(numRows_train))
        print('    カラム数: {}'.format(numCols_train))

        numRows_test = len(self.df_test)
        print('精度検証用データ:')
        print('    行数:     {}'.format(numRows_test))
        print('    カラム数: {}'.format(numCols_test))

        if self.is_pred:
            numCols_pred = self.df_pred.shape[1]
            numRows_pred = len(self.df_pred)
            print('予測用データ:')
            print('    行数:     {}'.format(numRows_pred))
            print('    カラム数: {}'.format(numCols_pred))

        print('=' * 60)
        dtype_list = [[_col, str(self.df_train[_col].dtype), self.df_train[_col].isnull().any()]
                      for _col in self.df_train.columns]
        return pd.DataFrame(dtype_list, columns=['column', 'data type', 'is Null'], index=range(1, numCols_train+1))

    def common_validation(self, ID, Target):
        self.ID = ID
        self.Target = Target

        self.ID_list = [_id.strip() for _id in self.ID.split(',')]

        if self.Target in self.ID_list:
            self._stop_process('IDの中に正解データを含めることはできません。')

        # == Validation for id and target columns ==
        is_col_err_train = self._column_validation(self.df_train, 'train')
        is_col_err_test = self._column_validation(self.df_test, 'test')
        if self.is_pred:
            is_col_err_pred = self._column_validation(self.df_pred, 'pred')
            if is_col_err_train or is_col_err_test or is_col_err_pred:
                print('=' * 30)
                self._stop_process('データを見直し、修正してからリトライしてください。')
        else:
            if is_col_err_train or is_col_err_test:
                print('=' * 30)
                self._stop_process('データを見直し、修正してからリトライしてください。')

        # == Validation for features ==
        # remove datetime data
        self._remove_datetime()

        # check feature column difference between train, test and pred
        exclude_columns = [self.Target]
        exclude_columns.extend(self.ID_list)

        col_features_train = set([_col for _col in self.df_train.columns if _col not in exclude_columns])
        col_features_test = set([_col for _col in self.df_test.columns if _col not in exclude_columns])

        col_diff_train_test = list(col_features_train - col_features_test)
        if len(col_diff_train_test) > 0:
            print('以下の訓練用データの特徴量は、検証用データには存在しないためエラーになります\n')
            for _col in col_diff_train_test:
                print('  - {}'.format(_col))
            print('=' * 40)
            print('以下の選択肢から、上記データの扱い方を決めてください')
            print(' 1. 訓練用データから削除する')
            print(' 2. マニュアルで対処する\n')
            operate_id = input()

            if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                del self.df_train.loc[:, col_diff_train_test]
            else:
                self._stop_process('データを見直し、修正してからリトライしてください。')
            col_features_train = set([_col for _col in self.df_train.columns if _col not in exclude_columns])

        col_diff_test_train = list(col_features_test - col_features_test)
        if len(col_diff_test_train) > 0:
            print('以下の精度検証用データの特徴量は、訓練用データには存在しないためエラーになります\n')
            for _col in col_diff_test_train:
                print('  - {}'.format(_col))
            print('=' * 40)
            print('以下の選択肢から、上記データの扱い方を決めてください')
            print(' 1. 精度検証用データから削除する')
            print(' 2. マニュアルで対処する\n')
            operate_id = input()

            if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                del self.df_test.loc[:, col_diff_train_test]
            else:
                self._stop_process('データを見直し、修正してからリトライしてください。')

        if self.is_pred:
            col_features_pred = set([_col for _col in self.df_pred.columns if _col not in exclude_columns])

            col_diff_train_pred = list(col_features_train - col_features_pred)
            if len(col_diff_train_pred) > 0:
                print('以下の訓練・精度検証用データの特徴量は、予測用データには存在しないためエラーになります\n')
                for _col in col_diff_train_pred:
                    print('  - {}'.format(_col))
                print('=' * 40)
                print('以下の選択肢から、上記データの扱い方を決めてください')
                print(' 1. 訓練用データおよび精度検証用データから削除する')
                print(' 2. マニュアルで対処する\n')
                operate_id = input()

                if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                    del self.df_train.loc[:, col_diff_train_pred]
                    del self.df_test.loc[:, col_diff_train_pred]
                else:
                    self._stop_process('データを見直し、修正してからリトライしてください。')

            col_diff_pred_train = list(col_features_pred - col_features_train)
            if len(col_diff_pred_train) > 0:
                print('以下の予測用データの特徴量は、訓練・精度検証用データには存在しないためエラーになります\n')
                for _col in col_diff_pred_train:
                    print('  - {}'.format(_col))
                print('=' * 40)
                print('以下の選択肢から、上記データの扱い方を決めてください')
                print(' 1. 精度検証用データから削除する')
                print(' 2. マニュアルで対処する\n')
                operate_id = input()

                if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                    del self.df_pred.loc[:, col_diff_pred_train]
                else:
                    self._stop_process('データを見直し、修正してからリトライしてください。')

    def _clf_validation(self):
        """
        クラス分類問題として扱う場合の検証
        """
        target_train = set(self.df_train[self.Target])
        target_test = set(self.df_test[self.Target])

        targets_not_exist_in_train = list(target_test - target_train)
        if len(targets_not_exist_in_train) > 0:
            print('以下の正解ラベルは精度検証用データのみに存在し、訓練用データには出現しないためエラーになります。')
            for _target in targets_not_exist_in_train:
                print('  - {}'.format(_target))
            print('=' * 40)
            print('以下の選択肢から、上記データの扱い方を決めてください')
            print(' 1. 精度検証用データから削除する')
            print(' 2. 精度検証用データから訓練用データに移動する')
            print(' 3. 上記以外の方法をマニュアルで実行する\n')
            operate_id = input()

            if operate_id in ('1', '１', 1, '一', 'いち', '①', '壱'):
                del self.df_test.loc[self.df_test[self.Target].isin(targets_not_exist_in_train)]
            elif operate_id in ('2', '２', 2, '二', 'に', '②', '弐'):
                df_tmp = self.df_test.loc[self.df_test[self.Target].isin(targets_not_exist_in_train)]
                self.df_train = pd.concat([self.df_train, df_tmp])
                del self.df_test.loc[self.df_test[self.Target].isin(targets_not_exist_in_train)]
                del df_tmp
            else:
                self._stop_process('データを見直し、修正してからリトライしてください。')

        target_dtype_train = self.df_train[self.Target].dtype
        if target_dtype_train not in ('int', 'object', 'bool'):
            err_msg = ('訓練用データの正解データが、整数型/文字列型/論理値 以外のデータ型であるためエラーになります。\n'
                       + 'データを見直し、修正してからリトライしてください。')
            self._stop_process(err_msg)

        target_dtype_test = self.df_test[self.Target].dtype
        if target_dtype_test not in ('int', 'object', 'bool'):
            err_msg = ('精度検証用データの正解データが、整数型/文字列型/論理値 以外のデータ型であるためエラーになります。\n'
                       + 'データを見直し、修正してからリトライしてください。')
            self._stop_process(err_msg)

    def _rgs_validation(self):
        """
        回帰問題として扱う場合の検証
        """
        target_dtype_train = self.df_train[self.Target].dtype
        target_dtype_test = self.df_test[self.Target].dtype

        if target_dtype_train not in ('int', 'float'):
            err_msg = ('訓練用の正解データに 数値 以外のデータが含まれるため、エラーになります。\n'
                       + 'データを見直し、修正してからリトライしてください。')
            self._stop_process(err_msg)

        if target_dtype_test not in ('int', 'float'):
            err_msg = ('精度検証用の正解データに 数値 以外のデータが含まれるため、エラーになります。\n'
                       + 'データを見直し、修正してからリトライしてください。')
            self._stop_process(err_msg)

    def specific_validation(self, Task, Metric):
        self.Task = Task
        self.Metric = Metric

        if self.Task == 'classification':
            self._clf_validation()
        elif self.Task == 'regression':
            self._rgs_validation()

    def check_dtype(self):

        df_dtype_ID = pd.DataFrame(self.ID_list, columns=['column'])
        df_dtype_ID['type'] = 'ID'

        df_dtype_Target = pd.DataFrame([self.Target], columns=['column'])
        df_dtype_Target['type'] = 'Target'

        exclude_columns = [self.Target]
        exclude_columns.extend(self.ID_list)

        col_features = [_col for _col in self.df_train.columns if _col not in exclude_columns]
        df_dtype_features = pd.DataFrame(col_features, columns=['column'])
        dtype_features = []
        for _col in col_features:
            dtype = str(self.df_train[_col].dtype)
            if dtype in ('int64', 'float64'):
                dtype_features.append('Numerical Feature')
            else:
                dtype_features.append('Categorical Feature')
        df_dtype_features['type'] = dtype_features

        df_dtype = pd.concat([df_dtype_ID, df_dtype_Target, df_dtype_features])
        df_dtype.index = df_dtype['column']
        del df_dtype['column']

        return df_dtype

    def make_conf(self, slack_webhook_url=None, comment=None, max_evals=None):
        self.slack_webhook_url = slack_webhook_url
        self.comment = comment
        self.max_evals = max_evals

        if self.slack_webhook_url is None:
            self.slack_webhook_url = 'https://hooks.slack.com/services/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' # ForecastFlowUsers.sys_notifications

        if self.comment is None:
            self.comment = 'None'

        if self.max_evals is None:
            self.max_evals = 10

        conf_dict = {
            'Learning': {
                'Task': [self.Task],
                'Metrics': [self.Metric]},
            'Dataset': {
                'Target': [self.Target],
                'ID': self.ID_list},
            'Slack': [self.slack_webhook_url],
            'Comment' : [self.comment],
            'Timestamp': [time.strftime('%Y-%m-%d %H:%M:%S')],
            'MaxEvals': [self.max_evals]}

        with open(os.path.join('/content', self.project_name, 'conf.yml'), 'w') as f:
            yaml.dump(conf_dict, f, default_flow_style=False)

        for f in [_fname for _fname in os.listdir(self.dir_train) if os.path.splitext(_fname)[1] == '.csv']:
            os.remove(os.path.join(self.dir_train, f))
        self.df_train.to_csv(os.path.join(self.dir_train, 'train.csv'), index=False)

        for f in [_fname for _fname in os.listdir(self.dir_test) if os.path.splitext(_fname)[1] == '.csv']:
            os.remove(os.path.join(self.dir_test, f))
        self.df_test.to_csv(os.path.join(self.dir_test, 'test.csv'), index=False)

        if self.is_pred:
            for f in [_fname for _fname in os.listdir(self.dir_pred) if os.path.splitext(_fname)[1] == '.csv']:
                os.remove(os.path.join(self.dir_pred, f))
            self.df_pred.to_csv(os.path.join(self.dir_pred, 'pred.csv'), index=False)

    def upload(self):
        upload = "gsutil -m cp -r {} gs://{}/".format(self.project_name, self.bucket_name)
        _exec_cmd(upload, verbose=False)

    def run(self):

        with open('/content/credential.json', 'r') as f:
            dict_credential = json.load(f)
            password = dict_credential['private_key_id']

        data = {
            'username': self.bucket_name,
            'password': password,
            'modelname': self.project_name,
            'max_evals': self.max_evals}

        # sending post request and saving response as response object
        req = requests.post(
            #url='https://echodolphin.ml/ff_api',
            url='https://echodolphin.org/v1/ml-run',
            json=data)

        # extracting response text
        print(req.text)
        res = json.loads(req.text)
        if res['message'] == '0':
            print('リクエストが正常に受け付けられました。完了までしばらくお待ちください。')
        elif res['message'] == '1':
            print('エラー: MaxEvals が許可された範囲を超えています。')
        elif res['message'] == '2':
            print('エラー: ユーザ名に誤りがあります。')


    def download(self):
        dl = 'gsutil -m cp -r gs://{}/{} .'.format(self.bucket_name, self.project_name)
        _exec_cmd(dl, verbose=False)

