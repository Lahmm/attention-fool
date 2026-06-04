"""Combine quick DIM/BG and cross-ViT reports into one concise conclusion."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dim-root', required=True); parser.add_argument('--cross-root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); dim_root, cross_root = Path(args.dim_root), Path(args.cross_root)
    mechanism = json.loads((dim_root / 'dim_bg_mechanism_report.json').read_text())
    ranking = json.loads((dim_root / 'method_high_frequency_ranking.json').read_text())
    cross = json.loads((cross_root / 'final_report.json').read_text())
    ordered = sorted(mechanism['summary'].items(), key=lambda item: item[1]['mean_asr'], reverse=True)
    confirmed = [row for row in cross['candidates'] if row['confirmed']]
    lines = ['# Combined Quick-Protocol Conclusion', '', 'Protocol: `quick_protocol`, seeds `0,1`.', '', '## DIM/BG mechanism', '']
    for name, row in ordered:
        lines.append(f"- `{name}`: mean ASR {row['mean_asr']:.4f}, direction derivative {row['mean_direction_derivative']:.4f}, gradient consistency {row['gradient_consistency']:.4f}.")
    lines += ['', '## L2-matched high-frequency ranking', '']
    for index, row in enumerate(ranking['ranking'][:10], 1):
        lines.append(f"{index}. `{row['method']}`: high-frequency ratio {row['high_frequency_ratio']:.4f}.")
    lines += ['', '## Cross-ViT confirmation', '']
    if confirmed:
        for row in confirmed: lines.append(f"- Confirmed `{row['component']}` on the independent confirmation split.")
    else: lines.append('- No screened component met every quick-protocol confirmation criterion.')
    Path(args.output).write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
