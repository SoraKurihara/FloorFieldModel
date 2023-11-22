import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import skfmm
from tqdm import tqdm


def prompt_continue():
    while True:
        answer = input("実行を続けますか [y/n]? ").lower()
        if answer in ["y", "n"]:
            return answer == "y"
        else:
            print("無効な入力です。'y' または 'n' を入力してください。")


class FloorFieldModel:
    def __init__(
        self,
        Map,
        SFF=None,
        num=0,
        Position=None,
        add_name=None,
        pedestrian_count=1,
        inflow=False,
    ):
        self.inflow = inflow
        os.makedirs("map", exist_ok=True)
        os.makedirs("SFF", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        basename = os.path.basename(Map)
        self.filename, ext = os.path.splitext(basename)
        if num:
            np.random.seed(num)
            self.filename = self.filename + f"_{num}"
        if add_name:
            self.filename = self.filename + f"_{add_name}"
        if ext.lower() == ".xlsx":
            self.original = (
                pd.read_excel(Map, header=None).fillna(0).values.astype(np.int8)
            )
        elif ext.lower() == ".npy":
            self.original = np.load(Map)
        else:
            print("知らん拡張子")

        if os.path.exists(os.path.join("data", f"{self.filename}.db")):
            print(f"{self.filename}.db はすでに存在しています．")
            if prompt_continue():
                print("上書きして実行を続けます。")
                os.remove(os.path.join("data", f"{self.filename}.db"))
            else:
                print("実行を中止します。")
                sys.exit()
        conn = sqlite3.connect(os.path.join("data", f"{self.filename}.db"))
        c = conn.cursor()

        # ステップテーブルの作成
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """
        )

        # 位置テーブルの作成
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step_id INTEGER,
            x INTEGER,
            y INTEGER,
            FOREIGN KEY(step_id) REFERENCES steps(id)
        )
        """
        )

        conn.commit()
        conn.close()

        self.Map = np.copy(self.original)

        self.Exit = np.array(list(zip(*np.where(self.Map == 3))))
        self.N = pedestrian_count

        if SFF:
            print(f"{SFF} を読み込みます．")
            self.SFF = np.load(SFF)
        else:
            self.new_sff()
            # self.initialize_sff()
            name = self.filename.rsplit("_", 1)
            np.save(os.path.join("SFF", name[0]), self.SFF)
        self.initialize_dff()
        if self.inflow:
            self.positions = np.argwhere(self.Map == 4)
            self.N -= len(self.positions)
            for pos in self.positions:
                self.Map[tuple(pos)] = 1
        else:
            if (type(Position) == np.ndarray) or (type(Position) == list):
                self.positions = Position
                for y, x in self.positions:
                    self.Map[int(y), int(x)] = 1
            else:
                self.initialize_positions()
        self.save_positions()
        print(self.SFF)
        print()
        print(self.Map)

    def new_sff(self):
        phi = np.ones_like(self.Map)
        phi[self.Map == 3] = 0
        mask = self.Map == 2
        phi = np.ma.MaskedArray(phi, mask)
        self.SFF = skfmm.distance(phi, dx=1)
        self.SFF = np.where(self.SFF.mask, -1, self.SFF.data)

    def initialize_sff(self):
        self.SFF = np.zeros_like(self.Map, dtype=np.int64)
        self.SFF[self.Map == 2] = -2
        self.SFF[self.Map == 0] = -1
        self.SFF[self.Map == 4] = -1

        count = 0
        while -1 in self.SFF:
            print(f"\r{count}", end="")
            starting_points = self.SFF == count
            for shift, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1)]:
                increment = np.roll(starting_points, shift=shift, axis=axis) & (
                    self.SFF == -1
                )
                self.SFF[increment] = count + 1
            count += 1
        self.SFF[self.SFF == -2] = -1

    def initialize_positions(self):
        Zero = np.argwhere(self.Map == 0)
        pos_indices = np.random.choice(len(Zero), self.N, replace=False)
        self.positions = Zero[pos_indices]

        for pos in self.positions:
            self.Map[tuple(pos)] = 1

    def initialize_dff(self):
        self.DFF = np.zeros_like(self.Map)

    def save_positions(self):
        conn = sqlite3.connect(os.path.join("data", f"{self.filename}.db"))
        c = conn.cursor()

        # 新しいステップを追加
        c.execute("INSERT INTO steps DEFAULT VALUES")
        step_id = c.lastrowid

        # 位置データを追加
        for y, x in zip(self.positions[:, 0], self.positions[:, 1]):
            c.execute(
                "INSERT INTO positions (step_id, x, y) VALUES (?, ?, ?)",
                (step_id, int(x), int(y)),
            )

        conn.commit()
        conn.close()

    def calculate_movement_probabilities(self, k_S=3, k_D=1):
        directions = [(1, 0), (-1, 0), (1, 1), (-1, 1), (0, 0)]  # 上下左右止
        M_vals, S_vals, D_vals = [], [], []

        for shift, axis in directions:
            M_vals.append(
                np.roll(self.Map, shift=shift, axis=axis)[
                    self.positions[:, 0], self.positions[:, 1]
                ]
            )
            S_vals.append(
                np.roll(self.SFF, shift=shift, axis=axis)[
                    self.positions[:, 0], self.positions[:, 1]
                ]
            )
            D_vals.append(
                np.roll(self.DFF, shift=shift, axis=axis)[
                    self.positions[:, 0], self.positions[:, 1]
                ]
            )

        S_vals = np.array(S_vals)
        S_max = S_vals.max(axis=0)
        S_vals = np.where(S_vals == -1, S_max, S_vals)
        S_vals = S_vals - S_vals.min(axis=0)

        # 各方向の移動確率を計算
        probs = np.array(
            [
                np.exp(-k_S * val[0] + k_D * val[1])
                * (val[2] != 2)
                * ((val[2] != 1) & (i != 4))
                for i, val in enumerate(zip(S_vals, D_vals, M_vals))
            ]
        )
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # 上下左右止
        for idx, pos in enumerate(self.positions):
            for i, dir in enumerate(directions):
                next_pos = tuple(np.array(pos) + np.array(dir))
                if self.original[next_pos] == 3:
                    probs[:, idx] = 0
                    probs[i, idx] = 1
                    break

        sums = np.sum(probs, axis=0)
        sums[sums == 0] = 1

        probs = probs / sums
        probs_tmp = 1 - probs[:-1, :].sum(axis=0)
        probs_tmp[probs_tmp < 0] = 0
        probs[-1, :] = probs_tmp
        self.movement_probs = probs

    def handle_collisions(self, candidates, unique_positions, counts):
        duplicates = unique_positions[counts > 1]

        # 重複している位置ごとの処理
        for dup in duplicates:
            # 位置が重複しているすべてのインデックスを取得
            indices = np.where((candidates == dup).all(axis=1))[0]

            # 1/2 の確率でアクションを選択
            if np.random.random() < 0.5:
                # 一人のみをその位置に残し、他のすべての人を元の位置に戻す
                chosen_one = np.random.choice(indices)
                for idx in indices:
                    if idx != chosen_one:
                        candidates[idx] = self.positions[idx]
            else:
                # すべての人を元の位置に戻す
                for idx in indices:
                    candidates[idx] = self.positions[idx]

        return candidates

    def move(self):
        candidate_list = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)])  # 上下左右止
        candidate_index = [
            np.random.choice([0, 1, 2, 3, 4], p=self.movement_probs[:, n])
            for n in range(len(self.movement_probs[0]))
        ]
        candidates = self.positions + candidate_list[candidate_index]
        # print(candidates)
        unique_positions, counts = np.unique(candidates, axis=0, return_counts=True)
        if sum(counts > 1) > 0:
            candidates = self.handle_collisions(candidates, unique_positions, counts)
        self.positions = candidates

        to_remove = []
        for i, pos in enumerate(self.positions):
            if any((pos == exit_pos).all() for exit_pos in self.Exit):
                to_remove.append(i)
        self.positions = np.delete(self.positions, to_remove, axis=0)
        self.N -= len(to_remove)

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

    def update(self):
        self.update_DFF()
        if self.inflow:
            new_ped = np.argwhere(self.Map == 4)
            new_ped = new_ped[: self.N]
            self.N -= len(new_ped)
            self.positions = np.r_[self.positions, new_ped]
        self.Map = np.copy(self.original)
        for pos in self.positions:
            self.Map[tuple(pos)] = 1

    def run(self, steps=10):
        for _ in tqdm(range(steps)):
            self.calculate_movement_probabilities()
            self.move()
            self.save_positions()
            self.update()
            if len(self.positions) == 0:
                break
        print(self.DFF)

    def anim(self, frame):
        self.c.execute("SELECT x, y FROM positions WHERE step_id=?", (frame + 1,))
        data = self.c.fetchall()
        # data = self.all_data[frame]
        Map = np.copy(self.original)
        for x, y in data:
            Map[int(y), int(x)] = 1
        self.ax.clear()  # 描画をクリア
        self.ax.set_title(f"step = {frame}, pedestrian = {len(data)}", fontsize=32)
        self.ax.imshow(Map, cmap=self.cmap, vmin=0, vmax=3)
        self.pbar.update(1)

    def plot(self):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        self.cmap = plt.cm.colors.ListedColormap(["white", "red", "black", "green"])

        self.conn = sqlite3.connect(os.path.join("data", f"{self.filename}.db"))
        self.c = self.conn.cursor()
        self.c.execute("SELECT seq FROM sqlite_sequence WHERE name=?", ("steps",))
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
            os.path.join("output", f"{self.filename}.mp4"), writer="ffmpeg", fps=10
        )
        plt.show()
        self.conn.close()


# if __name__ == "__main__":
#     model = FloorFieldModel(
#         r"01_Simulation/Data/Map.xlsx",
#         num=1,
#         add_name="Obstacle",
#         pedestrian_count=100,
#     )
#     model.run(steps=300)
#     model.plot()
