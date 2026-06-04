"""Causal gradient-transfer trace, intervention, MI-switch, and report CLI."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from gradient_analysis import MISwitch, attention_thirds, component_transform, direction_derivative, fft_project, parse_component, run_analyzed_attack
from main import ANNOTATIONS_PATH, IMAGE_DIR, create_attacker, parse_model_names
from nets import build_vit_model
from utils import DEVICE, load_data

MAIN_TARGETS=("deit_base_patch16_224","beit_base_patch16_224","swin_tiny_patch4_window7_224","pvt_v2_b2","cait_s24_224","levit_256","pit_s_224","crossvit_15_240")
HELD_OUT_TARGETS=("resnet50","convnext_base","efficientnet_b3","xcit_small_12_p16_224")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def bh_fdr(p_values):
    values=np.asarray(list(p_values),dtype=np.float64); order=np.argsort(values); adjusted=np.empty_like(values); running=1.0
    for rank in range(len(values),0,-1):
        idx=order[rank-1]; running=min(running,values[idx]*len(values)/rank); adjusted[idx]=running
    return adjusted.tolist()

def paired_bootstrap(values,repeats=10000,seed=0):
    """Bootstrap a paired image-by-model effect matrix over both strata."""
    if values.ndim != 2: raise ValueError("values must have shape [images, models].")
    rng=np.random.default_rng(seed); draws=np.empty(repeats)
    for idx in range(repeats):
        ii=rng.integers(0,values.shape[0],values.shape[0]); mi=rng.integers(0,values.shape[1],values.shape[1])
        draws[idx]=values[np.ix_(ii,mi)].mean()
    return {"mean":float(values.mean()),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"p_positive":float((draws<=0).mean())}

def build_baseline(num_classes):
    source=build_vit_model(num_classes=num_classes,model_name="vit_base_patch16_224")
    guides=tuple(build_vit_model(num_classes=num_classes,model_name=n) for n in ("deit_base_patch16_224","pit_s_224","cait_s24_224"))
    attacker=create_attacker(model=source,epsilon=16/255,step_size=None,steps=40,layers=(0,1,4,9,11),ti_sigma=0,
        dim=True,mi=True,mi_decay=1,normalize_grad=False,dim_resize_range=(.85,1),attention_guide_models=guides,
        attention_guide_type="qk_cls",attention_guide_build_method="patch",attention_guide_patch_size=16,
        guide_aug=True,guide_aug_area="background",guide_aug_methods=("dropout","jitter","freq"),guide_aug_copies=3,guide_aug_strength=.2)
    return source,attacker

def selected_batches(args,source,loader):
    selected=0
    for images,labels,indices in loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        with torch.inference_mode(): correct=source(images,return_attn=False).argmax(1).eq(labels)
        if not correct.any(): continue
        remaining=args.max_samples-selected
        if remaining<=0: break
        images,labels,indices=images[correct][:remaining],labels[correct][:remaining],indices[correct.cpu()][:remaining]
        selected+=images.size(0); yield images,labels,indices

def _target_normalize(model,pixels):
    cfg=getattr(model.model,"default_cfg",{})
    size=tuple(cfg.get("input_size",(3,pixels.size(-2),pixels.size(-1)))[-2:])
    if pixels.shape[-2:] != size: pixels=torch.nn.functional.interpolate(pixels,size=size,mode="bilinear",align_corners=False)
    mean=torch.tensor(cfg.get("mean",(.5,.5,.5)),device=pixels.device,dtype=pixels.dtype).view(1,3,1,1)
    std=torch.tensor(cfg.get("std",(.5,.5,.5)),device=pixels.device,dtype=pixels.dtype).view(1,3,1,1)
    return (pixels-mean)/std

def target_metrics(clean,adv,labels,names,with_gradients=False,traces=None):
    result={}
    for name in names:
        model=build_vit_model(num_classes=1000,model_name=name)
        with torch.inference_mode():
            clean_correct=model(_target_normalize(model,clean*.5+.5),return_attn=False).argmax(1).eq(labels)
            adv_success=model(_target_normalize(model,adv*.5+.5),return_attn=False).argmax(1).ne(labels)
        item={"clean_correct":clean_correct.cpu(),"adv_success":adv_success.cpu()}
        if traces is not None:
            step_success=[]; step_losses=[]
            with torch.inference_mode():
                for trace in traces:
                    normalized=_target_normalize(model,trace["x_t"].to(DEVICE))
                    logits=model(normalized,return_attn=False)
                    step_success.append(logits.argmax(1).ne(labels).cpu())
                    step_losses.append(torch.nn.functional.cross_entropy(logits,labels,reduction="none").cpu())
                final_logits=model(_target_normalize(model,adv*.5+.5),return_attn=False)
                step_success.append(final_logits.argmax(1).ne(labels).cpu())
                step_losses.append(torch.nn.functional.cross_entropy(final_logits,labels,reduction="none").cpu())
            item["step_adv_success"]=step_success; item["step_losses"]=step_losses
        if with_gradients and traces is not None:
            gradients=[]
            for trace in traces:
                pixels=trace["x_t"].to(DEVICE).requires_grad_(True)
                normalized=_target_normalize(model,pixels)
                loss=torch.nn.functional.cross_entropy(model(normalized,return_attn=False),labels)
                gradients.append(torch.autograd.grad(loss,pixels)[0].cpu())
            item["gradients"]=gradients
        result[name]=item; del model
    return result

def run_experiment(args):
    loader,num_classes=load_data(image_dir_arg=args.image_dir,annotations_path_arg=args.annotations_path,batch_size=args.batch_size,
        num_workers=args.num_workers,prefetch_factor=2,img_size=args.img_size)
    source,attacker=build_baseline(num_classes); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    projector=parse_component(args.component) if args.component else None
    transform=component_transform(projector,args.intervention,args.region) if projector else None
    switch=MISwitch(args.mi_switch,args.switch_step); seed_all(args.seed); batches=[]
    for batch_idx,(images,labels,indices) in enumerate(selected_batches(args,source,loader)):
        traces=[]; adv=run_analyzed_attack(attacker,images,labels,grad_transform=transform,mi_switch=switch,
            trace_callback=traces.append if args.mode=="trace" else None,diagnostics=args.mode=="trace" and args.gradient_decomposition)
        targets=target_metrics(images,adv,labels,args.target_models,args.target_gradients,traces) if args.evaluate_targets else {}
        path=output/f"batch_{batch_idx:04d}.pt"
        torch.save({"indices":indices.cpu(),"labels":labels.cpu(),"clean":images.cpu(),"adv":adv.cpu(),"traces":traces,"targets":targets},path)
        batches.append(path.name)
    manifest={"mode":args.mode,"seed":args.seed,"component":args.component,"region":args.region,"intervention":args.intervention,
        "mi_switch":args.mi_switch,"switch_step":args.switch_step,"batches":batches}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

def run_report(args):
    root=Path(args.input_dir); rows=[]
    for path in sorted(root.glob("batch_*.pt")):
        payload=torch.load(path,map_location="cpu")
        for model,metrics in payload.get("targets",{}).items():
            clean=metrics["clean_correct"].bool(); success=metrics["adv_success"].bool(); denominator=int(clean.sum())
            rows.append({"batch":path.name,"model":model,"clean_correct":denominator,"success":int((success&clean).sum()),
                "asr":float((success&clean).sum()/denominator) if denominator else None})
    report={"input_dir":str(root),"target_asr":rows}; output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("mode",choices=("trace","frequency-intervention","mi-switch","report"))
    p.add_argument("--output-dir",default="outputs/causal-analysis"); p.add_argument("--compare-dir",default=None); p.add_argument("--input-dir",default="outputs/causal-analysis")
    p.add_argument("--image-dir",default=IMAGE_DIR); p.add_argument("--annotations-path",default=ANNOTATIONS_PATH); p.add_argument("--img-size",type=int,default=224)
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--max-samples",type=int,default=100); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--component",default=None,help="fft:BAND[:ORIENTATION] or haar:PATH"); p.add_argument("--region",choices=("all","low","mid","high"),default="all"); p.add_argument("--intervention",choices=("keep","drop"),default="keep")
    p.add_argument("--mi-switch",choices=("always","never","on","off","reset"),default="always"); p.add_argument("--switch-step",type=int,default=1)
    p.add_argument("--evaluate-targets",action="store_true"); p.add_argument("--target-gradients",action="store_true"); p.add_argument("--gradient-decomposition",action="store_true")
    p.add_argument("--target-models",type=parse_model_names,default=MAIN_TARGETS+HELD_OUT_TARGETS); return p.parse_args()

def _summarize_observations(observations, seed):
    keys, summaries, p_values = list(observations), {}, []
    for key in keys:
        values=np.asarray(observations[key],dtype=np.float64).reshape(-1,1)
        summaries[key]=paired_bootstrap(values,seed=seed); p_values.append(summaries[key]["p_positive"])
    for key,q_value in zip(keys,bh_fdr(p_values)):
        summaries[key]["q_positive"]=q_value
    return summaries

def run_detailed_report(args):
    """Aggregate direction derivatives, MI advantage, patch maps, and paired causal ASR."""
    root=Path(args.input_dir); observations={}; heatmaps={}; asr_effects=[]
    compare=Path(args.compare_dir) if args.compare_dir else None
    for path in sorted(root.glob("batch_*.pt")):
        payload=torch.load(path,map_location="cpu"); traces=payload.get("traces",[])
        for model,metrics in payload.get("targets",{}).items():
            gradients=metrics.get("gradients",[])
            for step_idx,(trace,target_grad) in enumerate(zip(traces,gradients),start=1):
                grad=trace["gradient"]
                for band in range(8):
                    key=f"fft_band/{band}/step/{step_idx}"
                    observations.setdefault(key,[]).extend(direction_derivative(fft_project(grad,band),target_grad).tolist())
                guide=trace.get("guide_map")
                if guide is not None:
                    for region,mask in attention_thirds(guide).items():
                        key=f"region/{region}/step/{step_idx}"
                        observations.setdefault(key,[]).extend(direction_derivative(grad*mask,target_grad*mask).tolist())
                    contribution=(grad*target_grad).sum(1,keepdim=True)
                    patch=torch.nn.functional.adaptive_avg_pool2d(contribution,(14,14)).mean(0).squeeze(0)
                    heatmaps.setdefault(f"patch/model/{model}/step/{step_idx}",[]).append(patch)
                mi=direction_derivative(trace["mi_update"],target_grad)-direction_derivative(trace["raw_update"],target_grad)
                observations.setdefault(f"mi_advantage/step/{step_idx}",[]).extend(mi.tolist())
                for name,component in (trace.get("diagnostic_gradients") or {}).items():
                    observations.setdefault(f"source/{name}/step/{step_idx}",[]).extend(direction_derivative(component,target_grad).tolist())
        if compare is not None and (compare/path.name).exists():
            control=torch.load(compare/path.name,map_location="cpu"); rows=[]
            common=sorted(set(payload.get("targets",{})) & set(control.get("targets",{})))
            for model in common:
                current,base=payload["targets"][model],control["targets"][model]
                valid=current["clean_correct"].bool() & base["clean_correct"].bool()
                effect=(current["adv_success"].float()-base["adv_success"].float())
                rows.append(torch.where(valid,effect,torch.nan))
            if rows: asr_effects.append(torch.stack(rows,dim=1))
    summaries=_summarize_observations(observations,args.seed) if observations else {}
    report={"input_dir":str(root),"directional_metrics":summaries,
        "patch_heatmaps":{key:torch.stack(values).mean(0).tolist() for key,values in heatmaps.items()}}
    if asr_effects:
        effects=torch.cat(asr_effects).numpy(); complete=effects[~np.isnan(effects).any(axis=1)]
        report["paired_asr_effect"]=paired_bootstrap(complete,seed=args.seed) if len(complete) else None
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Wrote causal report with {len(summaries)} directional metrics to {output/'report.json'}")

if __name__=="__main__":
    args=parse_args()
    if args.mode=="report": run_detailed_report(args)
    else:
        if args.mode=="frequency-intervention" and not args.component: raise ValueError("frequency-intervention requires --component.")
        run_experiment(args)
