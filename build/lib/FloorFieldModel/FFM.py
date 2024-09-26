import os
import sqlite3

import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib.resources

from .cp import p_ij
from .dcm import L1norm, L2norm, Linfnorm
from .hc import handle_collisions
from .sql import create_sqlite, save_sqlite


class FloorFieldModel:
    def __init__(self, Map="random", SFF=None, method="L2"):
        self.current_step = 0
        self.create_directories()
        
        if Map == "random":
            with importlib.resources.path('FloorFieldModel.examples.map', '') as folder_path:
                npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

                random_npy = np.random.choice(npy_files)

                Map = os.path.join(folder_path, random_npy)
        self.filename, ext = os.path.splitext(os.path.basename(Map))
        self.original = self.load_map(Map, ext)
        self.Exit = np.where(self.original == 3)

        if SFF:
            self.SFF = np.load(
                os.path.join("SFF", f"{self.filename}_{method}.npy")
            )
        else:
            self.SFF = self.initialize_sff(method=method)
            np.save(os.path.join("SFF", f"{self.filename}_{method}"), self.SFF)

        self.initialize_dff()
        print(self.SFF)
        print()
        print(self.original)

    def create_directories(self):
        os.makedirs("map", exist_ok=True)
        os.makedirs("SFF", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("output", exist_ok=True)

    def load_map(self, map_path, ext):
        if ext.lower() == ".xlsx":
            return (
                pd.read_excel(map_path, header=None)
                .fillna(0)
                .values.astype(np.int8)
            )
        elif ext.lower() == ".npy":
            return np.load(map_path)
        else:
            raise ValueError("未知のファイル拡張子です")

    def initialize_sff(self, method="L2"):
        if method == "L1":
            return L1norm(self.original)
        elif method == "L2":
            return L2norm(self.original)
        elif method == "Linf":
            return Linfnorm(self.original)
        else:
            raise ValueError("未知のメソッドです")

    def initialize_dff(self):
        self.DFF = np.zeros_like(self.original)

    def params(
        self,
        N=0,
        inflow=None,
        k_S=3,
        k_D=1,
        d="Neumann",
    ):
        """シミュレーションパラメータの設定"""
        self.N = N
        self.inflow = inflow
        self.k_S = k_S
        self.k_D = k_D
        self.parameters = [k_S, k_D]
        self.d = d

        self.paraname = (
            f"ks{str(k_S).replace('.', '')}_kd{str(k_D).replace('.', '')}"
        )
        self.create_parameter_directories()
        self.initialize_db()
        self.Map = np.copy(self.original)

        if self.N > 0:
            self.initialize_positions()
        elif self.inflow:
            self.check_and_initialize_inflow()

    def create_parameter_directories(self):
        """パラメータに基づいたディレクトリの作成"""
        os.makedirs(f"data/{self.paraname}", exist_ok=True)
        os.makedirs(f"output/{self.paraname}", exist_ok=True)

    def initialize_db(self):
        """データベースの初期化"""
        self.number = 0
        self.dbname = f"{self.filename}_{self.number}.db"
        while os.path.exists(os.path.join("data", self.paraname, self.dbname)):
            self.number += 1
            self.dbname = f"{self.filename}_{self.number}.db"
        np.random.seed(self.number)
        create_sqlite(os.path.join("data", self.paraname, self.dbname))

    def initialize_positions(self):
        Zero = np.argwhere(self.original == 0)
        if len(Zero) < self.N:
            raise ValueError(
                "指定された歩行者数が空きセルの数を超えています。"
            )
        pos_indices = np.random.choice(len(Zero), self.N, replace=False)
        self.positions = Zero[pos_indices]
        for pos in self.positions:
            self.Map[tuple(pos)] = 1

    def check_and_initialize_inflow(self):
        """単一レイヤーの場合の流入処理"""
        if self.inflow and 4 in self.original:
            self.positions = np.argwhere(self.original == 4)
            self.inflow -= len(self.positions)
            for pos in self.positions:
                self.Map[tuple(pos)] = 1

    def move(self):
        candidate_list, idx = self.get_candidate_list()

        # 累積確率を使用して乱数選択を最適化
        cumulative_probs = np.cumsum(self.movement_probs, axis=0)
        random_values = np.random.rand(len(self.movement_probs[0]))
        candidate_index = [
            np.searchsorted(cumulative_probs[:, n], random_values[n])
            for n in range(len(random_values))
        ]
        
        candidates = self.positions + candidate_list[candidate_index]

        # 衝突処理
        unique_positions, counts = np.unique(
            candidates, axis=0, return_counts=True
        )
        if sum(counts > 1) > 0:
            candidates = handle_collisions(
                self.positions, candidates, unique_positions, counts
            )

        # 歩行者の移動
        self.update_positions(candidates)


    def get_candidate_list(self):
        if self.d == "Neumann":
            candidate_list = np.array(
                [(0, 1), (-1, 0), (0, -1), (1, 0), (0, 0)]
            )  # 右上左下止
            idx = [0, 1, 2, 3, 4]
        else:
            candidate_list = np.array(
                [
                    (0, 1),
                    (1, 1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (-1, -1),
                    (1, 0),
                    (1, -1),
                    (0, 0),
                ]
            )  # 右右下上右上左左上下左下止
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        return candidate_list, idx

    def update_positions(self, candidates):
        # 歩行者の実際の移動処理
        self.positions = candidates

        # 基本的な除去のロジック
        # self.remove_pedestrians()

    def exit_check(self):
        self.exit_pedestrian = np.argwhere(
            (self.original == 3) & (self.Map == 1)
        )
        self.positions = np.argwhere((self.original != 3) & (self.Map == 1))

    def entry_check(self):
        self.entry_pedestrian = np.argwhere(
            (self.original == 4) & (self.Map != 1)
        )

    def remove_pedestrians(self):
        # シミュレーションエリアから出た歩行者を除去
        # self.Map[self.exit_pedestrian] = 3
        self.positions = np.argwhere(self.Map == 1)
        for pos in self.positions:
            self.Map[tuple(pos)] = 1
        self.N -= len(self.positions)

    def update(self):
        # Dynamic Floor Field の更新
        self.update_DFF()

        # 流入処理
        # if self.inflow:
        #     self.process_inflow()

        # マップの状態更新
        self.update_map()

    def update_DFF(self, alpha=0.2, delta=0.2):
        center = (1 - alpha) * (1 - delta) * self.DFF
        up = alpha * (1 - delta) / 4 * np.roll(self.DFF, shift=1, axis=0)
        down = alpha * (1 - delta) / 4 * np.roll(self.DFF, shift=-1, axis=0)
        left = alpha * (1 - delta) / 4 * np.roll(self.DFF, shift=1, axis=1)
        right = alpha * (1 - delta) / 4 * np.roll(self.DFF, shift=-1, axis=1)
        self.DFF = center + up + down + left + right
        self.DFF[self.Map == 2] = 0
        self.DFF[self.Map == 3] = 0
        self.DFF[self.Map == 1] += 1

    def process_inflow(self):
        inflow_max = min(len(self.entry_pedestrian), self.inflow)
        choice = np.random.choice(
            len(self.entry_pedestrian), inflow_max, replace=False
        )
        new_ped = self.entry_pedestrian[choice]
        self.inflow -= len(new_ped)
        self.positions = np.r_[self.positions, new_ped]

    def update_map(self):
        # マップの状態を更新
        self.Map = np.copy(self.original)
        for pos in self.positions:
            self.Map[tuple(pos)] = 1

    def update_step(self):
        self.exit_check()
        self.entry_check()
        if len(self.positions) != 0:
            self.calculate_movement_probs()
            self.move()
        if self.inflow:
            self.process_inflow()
        self.save_state()
        self.update()
        self.remove_pedestrians()

    def run(self, steps=10):
        for step in tqdm(range(steps)):
            self.current_step = step
            self.update_step()
            if len(self.positions) == 0:
                break

    def calculate_movement_probs(self):
        self.movement_probs = p_ij(
            self.Map,
            self.positions,
            self.SFF,
            self.DFF,
            params=self.parameters,
            d=self.d,
        )

    def save_state(self):
        save_sqlite(
            os.path.join("data", self.paraname, self.dbname),
            self.positions,
        )

    def anim(self, frame):
        if self.footprints is False:
            self.Map = np.copy(self.original)
        else:
            self.Map *= 0.99
            self.Map[self.Map < 0.1] = 0
            self.Map[self.original == 2] = 2
            self.Map[self.original == 3] = 3
        self.c.execute(
            "SELECT x, y FROM positions WHERE step_id=?", (frame + 1,)
        )
        data = self.c.fetchall()
        for x, y in data:
            self.Map[int(y), int(x)] = 1
            self.sum[int(y), int(x)] += 1

        gradient_mask = (self.Map >= 0) & (self.Map < 1)

        # グラデーションカラーマップを適用
        colored_data = self.gradient_cmap(self.normalize(self.Map))

        # 2 (黒) と 3 (緑) に対応する色を適用
        colored_data[(self.Map == 1) & ~gradient_mask] = [1, 0, 0, 1]  # 赤
        colored_data[(self.Map == 2) & ~gradient_mask] = [0, 0, 0, 1]  # 黒
        colored_data[(self.Map == 3) & ~gradient_mask] = [0, 1, 0, 1]  # 緑

        self.ax.clear()  # 描画をクリア
        self.ax.set_title(
            f"step = {frame}, pedestrian = {len(data)}", fontsize=32
        )

        self.ax.imshow(colored_data)
        self.pbar.update(1)

    def plot(self, footprints=False):
        self.footprints = footprints
        self.Map = np.copy(self.original).astype(np.float64)
        self.sum = np.zeros_like(self.Map, dtype=np.int64)

        # self.originalが3である位置のインデックスを取得
        self.indices_of_interest = np.where(self.original == 3)
        # print(self.indices_of_interest)
        # print(len(self.indices_of_interest[0]))

        import matplotlib.animation as animation
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from numpy.lib.stride_tricks import as_strided

        # self.cmap = plt.cm.colors.ListedColormap(
        #     ["white", "red", "black", "green"]
        # )
        # 白から赤へのグラデーション用カラーマップ
        self.gradient_cmap = mcolors.LinearSegmentedColormap.from_list(
            "gradient", ["white", "blue"]
        )

        # 0から1の範囲を[0, 1]に正規化する関数
        self.normalize = mcolors.Normalize(vmin=0, vmax=1)

        self.conn = sqlite3.connect(
            os.path.join("data", self.paraname, self.dbname)
        )
        self.c = self.conn.cursor()
        self.c.execute(
            "SELECT seq FROM sqlite_sequence WHERE name=?", ("steps",)
        )
        self.last_step_id = self.c.fetchone()[0]

        # last_step_idを利用してpositionsテーブルからデータを取得

        self.pbar = tqdm(total=self.last_step_id)
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        ani = animation.FuncAnimation(
            self.fig, self.anim, frames=int(self.last_step_id), interval=100
        )
        plt.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
        )
        # plt.tight_layout()
        ani.save(
            os.path.join(
                "output", self.paraname, f"{self.filename}_{self.number}.mp4"
            ),
            writer="ffmpeg",
            fps=10,
        )
        # plt.show()


if __name__ == "__main__":
    FFM = FloorFieldModel(r"map/test.npy", method="Linf")
    # FFM = FloorFieldModel(
    #     r"map/Umeda_underground.npy", SFF=True, method="Linf"
    # )
    FFM.params(N=0, inflow=5, k_S=3, k_D=1, d="Moore")
    # FFM.params(N=100, k_S=3, k_D=1, d="Moore")
    FFM.run(steps=10000)
    FFM.plot(footprints=True)

