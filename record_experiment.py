import sys, json, subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

def get_git_head(repo_path):
    r = subprocess.run(['git', '-C', repo_path, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip()[:8]

def parse_eval_output(output):
    results = {}
    for line in output.strip().split('\n'):
        if line.startswith('model='):
            parts = line.split()
            model = parts[0].split('=')[1]
            for p in parts:
                if p.startswith('ASR='):
                    results[model] = float(p.split('=')[1])
    return results

def main():
    repo_path = sys.argv[1]
    exp_name = sys.argv[2]
    params = sys.argv[3]
    eval_output = sys.stdin.read()
    results = parse_eval_output(eval_output)
    if not results:
        print('ERROR: no results parsed')
        sys.exit(1)
    avg_asr = sum(results.values()) / len(results)
    results['avg'] = avg_asr
    results['git_head'] = get_git_head(repo_path)
    results['exp_name'] = exp_name
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    params_dict = json.loads(params)
    results.update(params_dict)
    df = pd.DataFrame([results])
    model_cols = ['deit_base_patch16_224','beit_base_patch16_224','swin_tiny_patch4_window7_224','pvt_v2_b2','cait_s24_224','levit_256','pit_s_224','crossvit_15_240']
    meta_cols = ['exp_name','timestamp','git_head','avg']
    param_cols = [k for k in params_dict.keys() if k not in model_cols + meta_cols]
    final_cols = [c for c in (meta_cols + param_cols + model_cols) if c in df.columns]
    df = df[final_cols]
    excel_path = Path(repo_path) / 'experiments.xlsx'
    if excel_path.exists():
        existing = pd.read_excel(excel_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_excel(excel_path, index=False)
    print(f'Saved to {excel_path}')
    print(f'Avg ASR: {avg_asr:.4f}')
    for m in model_cols:
        if m in results:
            print(f'  {m}: {results[m]:.4f}')

if __name__ == '__main__':
    main()
