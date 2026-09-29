"""Choice/Score用のlistwise pointer readoutと、Noul用のsigmoid readoutを定義する。

PointerReadoutはdecision位置（"Answer:"直後）と各選択肢末尾の隠れ状態から選択肢logitを計算する。
NoulReadoutはdecision位置の隠れ状態を真である確率へ変換する。
"""

from __future__ import annotations

import torch
from torch import nn


class PointerReadout(nn.Module):
    """Choice / Score共有のpointer-style listwise scorer。

        z_i = (W_q LN(h_dec)) · (W_k LN(h_opt_i)) / sqrt(d)

    選択肢数Kはrequestごとに可変で、固定サイズの出力層は使わない。

    LayerNormをprojection前に挟むのは、生のbackbone隠れ状態のノルムが
    （massive activation次元のせいで）巨大かつサンプルごとに変動するため
    （実測: ||h_dec|| ~ 250, 最大成分 ~130）。LayerNormなしだとdot productの
    絶対スケールがhidden stateのノルムにそのまま比例し、softmaxがほぼ確実に
    0/1へ飽和する（訓練前後どちらも観測済み: 訓練前spread~66、訓練後spread~11000、
    mean max-prob~0.9985）。LayerNormで各トークンのスケールを揃え、affine補正を
    学習可能にすることで、この飽和を構造的に防ぐ。
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ln_dec = nn.LayerNorm(hidden_size)
        self.ln_opt = nn.LayerNorm(hidden_size)
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h_dec: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """`h_dec`: (d,)、`h_opts`: (K, d) -> 各選択肢のlogit `z`: (K,)。

        softmaxはここでは取らない（cross-entropy lossと組み合わせやすくするため、呼び出し側に委ねる）。
        """
        q = self.w_q(self.ln_dec(h_dec))  # (d,)
        k = self.w_k(self.ln_opt(h_opts))  # (K, d)
        return (k @ q) / (self.hidden_size**0.5)  # (K,)


class MLPPointerReadout(nn.Module):
    """`PointerReadout`の非線形版。線形readout（bilinear dot product）でOOD Kendall tauが
    改善しなかった場合（`results/2026-09-23_gate1_pooling_and_severity_data.md` Gate 1-b:
    support_tickets held-outでtau+0.005のまま動かず）の候補。`decision`とdecisionと各選択肢の
    隠れ状態を`[h_dec; h_opt; h_dec*h_opt]`へ連結し、2層MLPでlogitへ落とす
    （elementwise積の項でbilinear相互作用を明示的に残しつつ、非線形性を追加する）。
    """

    def __init__(self, hidden_size: int, mlp_hidden: int | None = None) -> None:
        super().__init__()
        mlp_hidden = mlp_hidden or hidden_size
        self.ln_dec = nn.LayerNorm(hidden_size)
        self.ln_opt = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size * 3, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, 1)

    def forward(self, h_dec: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """`h_dec`: (d,)、`h_opts`: (K, d) -> 各選択肢のlogit `z`: (K,)。"""
        d = self.ln_dec(h_dec)  # (d,)
        o = self.ln_opt(h_opts)  # (K, d)
        d_rep = d.unsqueeze(0).expand_as(o)  # (K, d)
        feat = torch.cat([d_rep, o, d_rep * o], dim=-1)  # (K, 3d)
        return self.fc2(self.act(self.fc1(feat))).squeeze(-1)  # (K,)


class NoulReadout(nn.Module):
    """Noul（yes/no）用のdecision位置隠れ状態への線形変換。

        p_true = sigmoid(w^T h_dec + b)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, h_dec: torch.Tensor) -> torch.Tensor:
        """`h_dec`: (d,) -> logit（スカラー）。sigmoidは呼び出し側で明示的に取る。"""
        return self.linear(h_dec.unsqueeze(0)).squeeze()


def score_expectation(probabilities: torch.Tensor) -> float:
    """Scoreの最終値を確率加重平均`sum_i i * p_i`で計算する。レベルは1始まり。"""
    levels = torch.arange(1, probabilities.shape[0] + 1, dtype=probabilities.dtype)
    return float((levels * probabilities).sum())


def _smoke_check() -> None:
    torch.manual_seed(0)
    hidden_size = 16

    with torch.no_grad():
        pointer = PointerReadout(hidden_size)
        h_dec = torch.randn(hidden_size)
        h_opts = torch.randn(4, hidden_size)

        z = pointer(h_dec, h_opts)
        p = torch.softmax(z, dim=-1)
        print(f"PointerReadout: z.shape={tuple(z.shape)}, p={p.tolist()}, sum(p)={float(p.sum()):.6f}")
        assert z.shape == (4,), "zの形がKと一致しない"
        assert abs(float(p.sum()) - 1.0) < 1e-5, "softmax後の確率の合計が1にならない"

        score = score_expectation(p)
        print(f"score (probability-weighted mean over levels 1..4): {score:.4f}")
        assert 1.0 <= score <= 4.0, "scoreがレベル範囲[1,K]の外に出ている"

        noul = NoulReadout(hidden_size)
        logit = noul(h_dec)
        p_true = torch.sigmoid(logit)
        print(f"NoulReadout: logit={float(logit):.4f}, p_true={float(p_true):.4f}")
        assert logit.shape == (), "Noulのlogitがスカラーになっていない"
        assert 0.0 <= float(p_true) <= 1.0, "p_trueが[0,1]の範囲外"

        # 選択肢数Kが変わっても同じモジュールがそのまま使えることを確認（固定サイズ出力層ではない）
        h_opts_7 = torch.randn(7, hidden_size)
        z7 = pointer(h_dec, h_opts_7)
        assert z7.shape == (7,), "Kを変えてもPointerReadoutがそのまま使えていない"

        mlp_pointer = MLPPointerReadout(hidden_size)
        z_mlp = mlp_pointer(h_dec, h_opts)
        p_mlp = torch.softmax(z_mlp, dim=-1)
        print(f"MLPPointerReadout: z.shape={tuple(z_mlp.shape)}, sum(p)={float(p_mlp.sum()):.6f}")
        assert z_mlp.shape == (4,), "MLPPointerReadoutのzの形がKと一致しない"
        assert abs(float(p_mlp.sum()) - 1.0) < 1e-5, "MLPPointerReadout softmax後の確率の合計が1にならない"
        z_mlp_7 = mlp_pointer(h_dec, h_opts_7)
        assert z_mlp_7.shape == (7,), "Kを変えてもMLPPointerReadoutがそのまま使えていない"

    print("OK: PointerReadout（K可変）・MLPPointerReadout（K可変）・NoulReadout・"
          "score_expectationがすべて期待通りの形。")

if __name__ == "__main__":
    _smoke_check()
