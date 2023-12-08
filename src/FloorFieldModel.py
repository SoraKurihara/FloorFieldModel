import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cp.calc_probability import p_ij
from .dcm.distance_calc_method import L1norm, L2norm
from .hc.handle_collisions import handle_collisions
from .sql.create_and_save_sqlite import create_sqlite, save_sqlite


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
        method="L2",
    ):
        os.makedirs("map", exist_ok=True)
        os.makedirs("SFF", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        basename = os.path.basename(Map)
        self.filename, ext = os.path.splitext(basename)

        if ext.lower() == ".xlsx":
            self.original = (
                pd.read_excel(Map, header=None)
                .fillna(0)
                .values.astype(np.int8)
            )
        elif ext.lower() == ".npy":
            self.original = np.load(Map)
        else:
            print("知らん拡張子")

        self.Exit = np.array(list(zip(*np.where(self.original == 3))))

        if SFF:
            print(f"{SFF} を読み込みます．")
            self.SFF = np.load(SFF)
        else:
            self.initialize_sff(method=method)
            np.save(os.path.join("SFF", f"{self.filename}_{method}"), self.SFF)
        self.initialize_dff()
        print(self.SFF)
        print()
        print(self.original)

    def initialize_sff(self, method="L2"):
        if method == "L1":
            self.SFF = L1norm(self.original)
        elif method == "L2":
            self.SFF = L2norm(self.original)

    def initialize_dff(self):
        self.DFF = np.zeros_like(self.original)

    def params(
        self,
        N=0,
        inflow=None,
        k_S=3,
        k_D=1,
        k_Dir=None,
        k_Str=None,
    ):
        """parameters setting

        Args:
            N (int): num of pedestrian. Defaults to 0.
            inflow (None or int): inflow. Defaults to False.
            k_S (int): Static Floor Field parameter. Defaults to 3.
            k_D (int): Dynamic Floor Field parameter. Defaults to 1.
            k_Dir (None or int): Direction parameter. Defaults to None.
            k_Str (None or int): Stress parameter. Defaults to None.
        """
        self.N = N
        self.inflow = inflow
        self.k_S = k_S
        self.k_D = k_D
        self.k_Dir = k_Dir
        self.k_Str = k_Str
        self.parameters = [k_S, k_D, k_Dir, k_Str]

        self.paraname = (
            f"ks{str(k_S).replace('.', '')}_kd{str(k_D).replace('.', '')}"
        )
        os.makedirs(rf"data/{self.paraname}", exist_ok=True)
        os.makedirs(rf"output/{self.paraname}", exist_ok=True)

        self.number = 0
        self.dbname = f"{self.filename}_{self.number}.db"
        while os.path.exists(os.path.join("data", self.paraname, self.dbname)):
            self.number += 1
            self.dbname = f"{self.filename}_{self.number}.db"
        np.random.seed(self.number)
        self.Map = np.copy(self.original)

        if self.N == 0:
            if self.inflow == None:
                self.inflow = 100
        else:
            self.initialize_positions()

        if self.inflow:
            if 4 not in self.original:
                print("流入口が見つかりません．")
                sys.exit()
            self.positions = np.argwhere(self.original == 4)
            self.inflow -= len(self.positions)
            for pos in self.positions:
                self.Map[tuple(pos)] = 1

        self.directions = np.ones(len(self.positions[:, 0])) * 4
        self.stresses = np.zeros(len(self.positions[:, 0]))

        create_sqlite(os.path.join("data", self.paraname, self.dbname))
        save_sqlite(
            os.path.join("data", self.paraname, self.dbname), self.positions
        )

    def initialize_positions(self):
        Zero = np.argwhere(self.original == 0)
        pos_indices = np.random.choice(len(Zero), self.N, replace=False)
        self.positions = Zero[pos_indices]
        self.directions = np.ones(self.N) * 4

        for pos in self.positions:
            self.Map[tuple(pos)] = 1

    def move(self):
        candidate_list = np.array(
            [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        )  # 上下左右止
        candidate_index = [
            np.random.choice([0, 1, 2, 3, 4], p=self.movement_probs[:, n])
            for n in range(len(self.movement_probs[0]))
        ]
        candidates = self.positions + candidate_list[candidate_index]
        unique_positions, counts = np.unique(
            candidates, axis=0, return_counts=True
        )
        if sum(counts > 1) > 0:
            candidates = handle_collisions(
                self.positions, candidates, unique_positions, counts
            )
        self.stresses[np.all(self.positions == candidates, axis=1)] += 1
        self.stresses[np.all(self.positions != candidates, axis=1)] = 0
        self.positions = candidates
        self.directions = np.array(candidate_index)

        to_remove = []
        for i, pos in enumerate(self.positions):
            if any((pos == exit_pos).all() for exit_pos in self.Exit):
                to_remove.append(i)
        self.positions = np.delete(self.positions, to_remove, axis=0)
        self.directions = np.delete(self.directions, to_remove)
        self.stresses = np.delete(self.stresses, to_remove)
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
            new_ped = new_ped[: self.inflow]
            new_dir = np.ones(len(new_ped)) * 4
            new_str = np.zeros(len(new_ped))
            self.inflow -= len(new_ped)
            self.positions = np.r_[self.positions, new_ped]
            self.directions = np.r_[self.directions, new_dir]
            self.stresses = np.r_[self.stresses, new_str]
        self.Map = np.copy(self.original)
        for pos in self.positions:
            self.Map[tuple(pos)] = 1

    def run(self, steps=10):
        for _ in tqdm(range(steps)):
            self.movement_probs = p_ij(
                self.Map, self.positions, self.SFF, self.DFF, self.parameters
            )
            self.move()
            save_sqlite(
                os.path.join("data", self.paraname, self.dbname),
                self.positions,
            )
            self.update()
            if len(self.positions) == 0:
                break
        print(self.DFF)

    def anim(self, frame):
        self.c.execute(
            "SELECT x, y FROM positions WHERE step_id=?", (frame + 1,)
        )
        data = self.c.fetchall()
        # data = self.all_data[frame]
        Map = np.copy(self.original)
        for x, y in data:
            Map[int(y), int(x)] = 1
        self.ax.clear()  # 描画をクリア
        self.ax.set_title(
            f"step = {frame}, pedestrian = {len(data)}", fontsize=32
        )
        self.ax.imshow(Map, cmap=self.cmap, vmin=0, vmax=3)
        self.pbar.update(1)

    def plot(self):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        self.cmap = plt.cm.colors.ListedColormap(
            ["white", "red", "black", "green"]
        )

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
        plt.show()
        self.conn.close()
