"""Choice/Score用のlistwise pointer readoutと、Noul用のsigmoid readoutを定義する。

PointerReadoutは`<DECISION>`と各選択肢末尾の隠れ状態から選択肢logitを計算する。
NoulReadoutは`<DECISION>`の隠れ状態を真である確率へ変換する。
"""

from __future__ import annotations

import torch
from torch import nn


class PointerReadout(nn.Module):
    """Choice / Score共有のpointer-style listwise scorer。

        z_i = (W_q h_dec) · (W_k h_opt_i) / sqrt(d)

    選択肢数Kはrequestごとに可変で、固定サイズの出力層は使わない。
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h_dec: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """`h_dec`: (d,)、`h_opts`: (K, d) -> 各選択肢のlogit `z`: (K,)。

        softmaxはここでは取らない（cross-entropy lossと組み合わせやすくするため、呼び出し側に委ねる）。
        """
        q = self.w_q(h_dec)  # (d,)
        k = self.w_k(h_opts)  # (K, d)
        return (k @ q) / (self.hidden_size**0.5)  # (K,)


class NoulReadout(nn.Module):
    """Noul（yes/no）用の`<DECISION>`隠れ状態への線形変換。

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

    print("OK: PointerReadout（K可変）・NoulReadout・score_expectationがすべて期待通りの形。")

if __name__ == "__main__":
    _smoke_check()
