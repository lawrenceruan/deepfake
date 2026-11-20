"""Encode neutral text once, then release T5 before detector training."""
import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prompt', default='')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    from wan.configs.wan_ti2v_5B import ti2v_5B as cfg
    from wan.modules.t5 import T5EncoderModel
    root = Path(a.checkpoint_dir)
    encoder = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype,
        device=torch.device(a.device), checkpoint_path=str(root/cfg.t5_checkpoint),
        tokenizer_path=str(root/cfg.t5_tokenizer))
    with torch.no_grad():
        context = encoder([a.prompt], torch.device(a.device))[0].cpu()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'context': context, 'prompt': a.prompt, 'checkpoint_dir': str(root.resolve())}, a.output)


if __name__ == '__main__':
    main()
