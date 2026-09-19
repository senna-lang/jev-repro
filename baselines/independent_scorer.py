"""stateと各選択肢を独立にエンコードして内積で順位付けする比較用scorer。

state末尾の隠れ状態を`h_state`、各選択肢末尾の隠れ状態を`h_opt_i`とし、
`z_i = h_state · h_opt_i`を選択肢logitとして返す。各選択肢は他の選択肢を参照しないため、
選択肢を追加しても既存選択肢間のlog-oddsは変化しない。
"""

from __future__ import annotations

import torch

from model.backbone import MODEL_NAME, load_backbone


class IndependentScorer:
    """state・各選択肢を完全に独立にエンコードし、生の内積でスコアリングするbaseline。"""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.backbone, self.tokenizer = load_backbone(model_name)

    @torch.no_grad()
    def _encode_last_token(self, text: str) -> torch.Tensor:
        """テキストを独立にforwardし、最後のトークンの隠れ状態を返す（causalなので文全体を要約した表現）。"""
        inputs = self.tokenizer(text, return_tensors="pt")
        out = self.backbone(**inputs)
        return out.last_hidden_state[0, -1]  # (hidden_size,)

    @torch.no_grad()
    def logits(self, state: str, options: list[str]) -> torch.Tensor:
        """`z_i = h_state . h_opt_i`（生の内積、学習パラメータなし・スケーリングなし）を返す。"""
        h_state = self._encode_last_token(state)
        h_opts = torch.stack([self._encode_last_token(opt) for opt in options])  # (K, hidden_size)
        return h_opts @ h_state  # (K,)

    @torch.no_grad()
    def forward(self, state: str, options: list[str]) -> torch.Tensor:
        """`logits`をsoftmaxした確率 `(K,)` を返す。"""
        return torch.softmax(self.logits(state, options), dim=-1)


def _smoke_check() -> None:
    scorer = IndependentScorer()

    state = "Oil prices surged 5% today after OPEC announced production cuts."
    options = ["World", "Sports", "Business", "Sci-Tech"]

    z4 = scorer.logits(state, options)
    p4 = torch.softmax(z4, dim=-1)
    print(f"K=4: z={z4.tolist()}, p={p4.tolist()}, sum(p)={float(p4.sum()):.6f}")
    assert abs(float(p4.sum()) - 1.0) < 1e-4

    # 無関係な選択肢を1つ追加。既存2択（World, Sports）のz（ひいてはlog-odds = z_1 - z_2）が
    # 不変であるはず（h_opt_iが他の選択肢を一切見ずに独立エンコードされているため、数式上ズレようがない）。
    # 確率p経由でlog-oddsを取ると、生の内積が大きすぎてfloat32のsoftmaxが厳密に0.0/1.0へ飽和し
    # log(0/0)=nanになる（実際に観測した）。z自体（softmaxの前）で比較するのが数式的に正しく、
    # 精度の問題も回避できる。
    options_plus_one = options + ["Irrelevant Extra Option"]
    z5 = scorer.logits(state, options_plus_one)
    p5 = torch.softmax(z5, dim=-1)
    print(f"K=5: z={z5.tolist()}, p={p5.tolist()}, sum(p)={float(p5.sum()):.6f}")
    assert abs(float(p5.sum()) - 1.0) < 1e-4

    log_odds_4 = float(z4[0] - z4[1])
    log_odds_5 = float(z5[0] - z5[1])
    print(f"log-odds(World/Sports) = z_World - z_Sports: K=4 -> {log_odds_4:.6f}, K=5 -> {log_odds_5:.6f}, "
          f"diff = {abs(log_odds_4 - log_odds_5):.2e}")
    assert z4[0] == z5[0] and z4[1] == z5[1], (
        "independent scorerはh_opt_iを他の選択肢を見ずに独立にエンコードするので、"
        "無関係な選択肢を追加してもWorld/Sportsのzはビット単位で不変のはず"
    )
    assert log_odds_4 == log_odds_5, (
        "independent scorerは無関係な選択肢を追加しても既存2択のlog-oddsが不変であるはず"
        "（JevModelのpointer方式と対照的な、この設計の数式上の性質）"
    )

    # 生の内積（スケーリングなし）は大きいため、softmax後の確率は実質0.0/1.0に飽和する。
    # これはこのscorerの出力スケールによる性質であり、logitを比較に用いる。
    print(f"（観察）生の内積のスケールが大きく、確率は0/1に飽和: p4={p4.tolist()}")

    print("OK: independent scorerが動作し、'無関係な選択肢を追加しても既存2択のzとlog-oddsは"
          "ビット単位で不変'という設計上の性質をz（softmax前）で厳密に確認できた"
          "（JevModelのpointer方式との対照）。")


if __name__ == "__main__":
    _smoke_check()
