"""学習前と選定LoRAの eval 全 branch 予測（4順番平均、正準順）を {case_id: p} で保存する。

人間テストの集計（`e2.human_eval --predictions`）と paired bootstrap の入力に使う。
使い方（GPU）: python -m e2.dump_predictions checkpoints/e2/lr-1e-4/epoch-3-quarter-3 results/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from e2.evaluate_lora import option_orders, predict_request
from e2.packed_data import load_requests
from e2.train_lora import load_lora_model
from model.packer import Branch


def dump(model, tokenizer, device) -> dict[str, list[float]]:
    out = {}
    for rows, request in load_requests(Path("data/e2/teacher_branches.jsonl"), Path("data/e2/states.jsonl")):
        if rows[0]["split"] != "eval":
            continue
        branches = [Branch(b.kind, b.instructions, r["options"]) for r, b in zip(rows, request.branches)]
        orders = [option_orders(r["kind"], len(r["options"]), r["case_id"]) for r in rows]
        for row, p in zip(rows, predict_request(model, tokenizer, request.state, branches, orders, device)):
            out[row["case_id"]] = p
    return out


def main() -> None:
    adapter, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    device = torch.device("cuda")
    model, tokenizer = load_lora_model(adapter)
    model.to(device).eval()
    (out_dir / "e2_pred_selected.json").write_text(json.dumps(dump(model, tokenizer, device)))
    with model.disable_adapter():
        (out_dir / "e2_pred_baseline.json").write_text(json.dumps(dump(model, tokenizer, device)))


if __name__ == "__main__":
    main()
