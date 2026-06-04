"""Reusable primitives for causal gradient-transfer experiments."""
from contextlib import contextmanager
from dataclasses import dataclass
import torch

FFT_BANDS = (0.0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 1.0)
FFT_ORIENTATIONS = ("all", "horizontal", "vertical", "diagonal")


def fft_mask(height, width, band, orientation="all", *, device=None, dtype=torch.float32):
    """Return a conjugate-symmetric orthogonal full-FFT mask."""
    if not 0 <= band < len(FFT_BANDS) - 1:
        raise ValueError(f"band must be in [0, {len(FFT_BANDS) - 2}], got {band}.")
    if orientation not in FFT_ORIENTATIONS:
        raise ValueError(f"orientation must be one of {FFT_ORIENTATIONS}, got {orientation!r}.")
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype).view(height, 1)
    fx = torch.fft.fftfreq(width, device=device, dtype=dtype).view(1, width)
    radius = torch.sqrt((fy / 0.5).square() + (fx / 0.5).square()) / (2.0 ** 0.5)
    lo, hi = FFT_BANDS[band], FFT_BANDS[band + 1]
    radial = (radius >= lo) & (radius <= hi) if band == 0 else (radius > lo) & (radius <= hi)
    if orientation == "all":
        return radial
    ax, ay = fx.abs(), fy.abs()
    if orientation == "horizontal":
        direction = ax > 2.0 * ay
    elif orientation == "vertical":
        direction = ay > 2.0 * ax
    else:
        direction = ~(ax > 2.0 * ay) & ~(ay > 2.0 * ax)
    return radial & direction


def fft_project(x, band, orientation="all"):
    mask = fft_mask(x.size(-2), x.size(-1), band, orientation, device=x.device)
    work = x if x.dtype == torch.float64 else x.float()
    freq = torch.fft.fft2(work, dim=(-2, -1), norm="ortho")
    return torch.fft.ifft2(freq * mask, dim=(-2, -1), norm="ortho").real.to(x.dtype)


def fft_decompose(x):
    return [fft_project(x, band) for band in range(len(FFT_BANDS) - 1)]


def _haar_split(x):
    if x.size(-2) % 2 or x.size(-1) % 2:
        raise ValueError("Haar projection requires even spatial dimensions at every selected level.")
    x00, x01 = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    x10, x11 = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return {"L": (x00+x01+x10+x11)*0.5, "H": (x00-x01+x10-x11)*0.5,
            "V": (x00+x01-x10-x11)*0.5, "D": (x00-x01-x10+x11)*0.5}


def _haar_merge(p):
    ll, lh, hl, hh = p["L"], p["H"], p["V"], p["D"]
    out = torch.empty(*ll.shape[:-2], ll.size(-2)*2, ll.size(-1)*2, device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2] = (ll+lh+hl+hh)*0.5
    out[..., 0::2, 1::2] = (ll-lh+hl-hh)*0.5
    out[..., 1::2, 0::2] = (ll+lh-hl-hh)*0.5
    out[..., 1::2, 1::2] = (ll-lh-hl+hh)*0.5
    return out


def haar_packet_project(x, path, levels=3):
    """Project onto one orthogonal Haar wavelet-packet cell, e.g. ``LLH``."""
    if len(path) != levels or any(key not in "LHVD" for key in path):
        raise ValueError(f"path must contain exactly {levels} characters from L,H,V,D.")
    def recurse(node, depth):
        parts, selected = _haar_split(node), path[depth]
        for key in parts:
            if key == selected:
                parts[key] = recurse(parts[key], depth+1) if depth+1 < levels else parts[key]
            else:
                parts[key] = torch.zeros_like(parts[key])
        return _haar_merge(parts)
    work = x if x.dtype == torch.float64 else x.float()
    return recurse(work, 0).to(x.dtype)


def haar_packet_paths(levels=3):
    paths = [""]
    for _ in range(levels):
        paths = [prefix + suffix for prefix in paths for suffix in "LHVD"]
    return paths


def attention_thirds(guide):
    flat, masks = guide.flatten(1), {}
    order = flat.argsort(dim=1)
    bounds = (0, flat.size(1)//3, 2*flat.size(1)//3, flat.size(1))
    for name, start, end in zip(("low", "mid", "high"), bounds[:-1], bounds[1:]):
        mask = torch.zeros_like(flat); mask.scatter_(1, order[:, start:end], 1.0)
        masks[name] = mask.view_as(guide)
    return masks


def direction_derivative(component, target_grad):
    return (component * target_grad).flatten(1).sum(dim=1)


@dataclass(frozen=True)
class MISwitch:
    kind: str = "always"
    step: int = 1
    def use_momentum(self, step):
        if self.kind == "always": return True
        if self.kind == "never": return False
        if self.kind == "on": return step >= self.step
        if self.kind == "off": return step <= self.step
        if self.kind == "reset": return True
        raise ValueError(f"Unsupported MI switch kind: {self.kind}")


def _rng_state():
    return (torch.random.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def _set_rng_state(state):
    torch.random.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state_all(state[1])

@contextmanager
def _attack_options(attacker, **options):
    previous = {name: getattr(attacker, name) for name in options}
    try:
        for name, value in options.items(): setattr(attacker, name, value)
        yield
    finally:
        for name, value in previous.items(): setattr(attacker, name, value)

def gradient_diagnostics(attacker, pixels, labels, guide, random_state):
    """Compute matched-randomness DIM/BG interaction, method, and leave-one-out gradients."""
    result = {}
    configs = {"plain": (False, False), "dim": (True, False), "bg": (False, True), "dim_bg": (True, True)}
    for name, (dim, bg) in configs.items():
        _set_rng_state(random_state)
        with _attack_options(attacker, input_diversity=dim, guide_aug=bg):
            result[name] = attacker._attack_grad(pixels, labels, guide).detach().cpu()
    result["interaction"] = result["dim_bg"] - result["dim"] - result["bg"] + result["plain"]
    methods = attacker.guide_aug_methods
    for method in methods:
        _set_rng_state(random_state)
        with _attack_options(attacker, guide_aug_methods=(method,)):
            result[f"method_{method}"] = attacker._attack_grad(pixels, labels, guide).detach().cpu()
        remaining = tuple(item for item in methods if item != method)
        if remaining:
            _set_rng_state(random_state)
            with _attack_options(attacker, guide_aug_methods=remaining):
                result[f"leave_out_{method}"] = attacker._attack_grad(pixels, labels, guide).detach().cpu()
    return result

def run_analyzed_attack(attacker, images, labels, *, grad_transform=None, mi_switch=None, trace_callback=None, diagnostics=False):
    """Run the existing attack algorithm with optional observation/intervention."""
    images, labels = images.to(attacker.device), labels.to(attacker.device)
    clean = attacker._denormalize(images).detach()
    needs_guide = (attacker.guide_aug and attacker.guide_aug_area != "all") or attacker.guide_grad_norm_area != "none"
    guide = attacker._build_guide_pixel_map(images, clean.size(-1)) if needs_guide else None
    adv, momentum = clean.clone().detach(), torch.zeros_like(clean)
    switch = mi_switch or MISwitch("always" if attacker.use_momentum else "never")
    for step_idx in range(attacker.steps):
        step, grad_pixels = step_idx + 1, adv.detach()
        if attacker.nesterov and step_idx > 0:
            with torch.no_grad():
                grad_pixels = grad_pixels + attacker.decay * attacker.step_size * momentum.sign()
                delta = torch.clamp(grad_pixels-clean, -attacker.epsilon, attacker.epsilon)
                grad_pixels = torch.clamp(clean+delta, 0.0, 1.0)
        grad_pixels = grad_pixels.detach().requires_grad_(True)
        before_rng = _rng_state()
        grad = attacker._attack_grad(grad_pixels, labels, guide)
        after_rng = _rng_state()
        diagnostic_grads = gradient_diagnostics(attacker, grad_pixels, labels, guide, before_rng) if diagnostics else None
        _set_rng_state(after_rng)
        if attacker.normalize_grad: grad = attacker._normalize_grad(grad)
        grad = attacker._smooth_grad(attacker._normalize_guided_grad(grad, guide))
        if grad_transform is not None: grad = grad_transform(grad, guide, step)
        previous = torch.zeros_like(momentum) if switch.kind == "reset" and step == switch.step else momentum
        momentum = attacker.decay * previous + grad
        update = momentum if switch.use_momentum(step) else grad
        if trace_callback is not None:
            trace_callback({"step": step, "x_t": adv.detach().cpu(), "gradient": grad.detach().cpu(),
                "momentum_before": previous.detach().cpu(), "momentum_after": momentum.detach().cpu(),
                "raw_update": grad.sign().detach().cpu(), "mi_update": momentum.sign().detach().cpu(),
                "guide_map": None if guide is None else guide.detach().cpu(), "diagnostic_gradients": diagnostic_grads})
        with torch.no_grad():
            adv = adv + attacker.step_size * update.sign()
            delta = torch.clamp(adv-clean, -attacker.epsilon, attacker.epsilon)
            adv = torch.clamp(clean+delta, 0.0, 1.0)
    return attacker._normalize(adv)


def component_transform(projector, intervention, region="all"):
    if region not in ("all", "low", "mid", "high"): raise ValueError("region must be all, low, mid, or high.")
    if intervention not in ("keep", "drop"): raise ValueError("intervention must be 'keep' or 'drop'.")
    def transform(grad, guide, _step):
        component = projector(grad)
        if region != "all":
            if guide is None: raise ValueError("A guide map is required for spatial intervention.")
            component = component * attention_thirds(guide)[region]
        return component if intervention == "keep" else grad-component
    return transform


def parse_component(spec):
    fields = spec.split(":")
    if fields[0] == "fft" and len(fields) in (2, 3):
        band, orientation = int(fields[1]), fields[2] if len(fields) == 3 else "all"
        return lambda x: fft_project(x, band, orientation)
    if fields[0] == "haar" and len(fields) == 2: return lambda x: haar_packet_project(x, fields[1])
    raise ValueError(f"Invalid component {spec!r}; expected fft:BAND[:ORIENTATION] or haar:PATH.")
