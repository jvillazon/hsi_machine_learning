#!/usr/bin/env python3
"""
transform_svg.py

Transform an SVG file:
  - White fill / stroke / color  →  black (#000000)
  - Black background rectangles  →  transparent (fill: none)
  - X-tick labels  →  custom labels (optional)

Usage:
    python transform_svg.py input.svg
    python transform_svg.py input.svg -o output.svg
    python transform_svg.py input.svg --cmap             # also save cmap PNG
    python transform_svg.py input.svg --labels "Label1" "Label2" "Label3"


# ── CUSTOM XTICK LABELS ──────────────────────────────────────────────────────
# Edit this list to replace x-tick labels with custom strings
# Set to None or empty list to skip label replacement
CUSTOM_XTICK_LABELS = None  # Example: ['Label1', 'Label2', 'Label3', ...]
"""



import re
import argparse
import os
import sys


# ── colour helpers ───────────────────────────────────────────────────────────

CUSTOM_XTICK_LABELS = [
        "Unknown",
        "Unknown",
        "CAR 18:2 (+H)",
        "CAR 18:1 (+H)",
        "CAR 18:0 (+H)",
        "Cer 17:2;2O/8:0;O (+Na)",
        "LPC 16:0 (+H)",
        "Unknown",
        "SM 34:1;2O (+H) or SM 18:0;2O/16:1 (+H) or SM 18:1;2O/16:0 (+H)",
        "Unknown",
        "PC 32:0 (+H)",
        "PC 34:2 (+H)",
        "Unknown (Missing)",
        "PC 16:0_18:1 (+H)",
        "PE 18:0_20:4 (+H) or PC 35:4 (+H)",
        "PC 36:5 (+H) or PC 16:0_20:5 (+H) or PC 18:2_18:3 (+H) or PC 16:1_20:4 (+H)",
        "PC 36:3 (+H)"
        "PC 18:0_18:1 (+H)",
        "PC O-38:5 (+H)",
        "PC 38:5 (+H)",
        "PC 18:0_20:4 (+H) or PC 16:0_22:4 (+H)",
        "SM 42:2;2O (+H)",
        "SM 42:1;2O (+H)",
        "SM 41:0;3O (+H)",
        "SM 42:0;3O (+H)"

    ]

_WHITE_RE = re.compile(
    r'#fff(?:fff)?(?=[^0-9a-fA-F]|$)'        # #fff  or  #ffffff
    r'|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\)'  # rgb(255,255,255)
    r'|\bwhite\b',                             # white
    re.IGNORECASE,
)

_BLACK_COLORS = frozenset({
    '#000000', '#000', 'black',
    'rgb(0,0,0)', 'rgb(0, 0, 0)',
})


def _is_black(color):
    if not color:
        return False
    return re.sub(r'\s+', '', color.strip().lower()) in _BLACK_COLORS


# ── Step 1 – replace white → black in colour-bearing attributes ──────────────

# Matches:  fill="…"  stroke="…"  color="…"  style="…"
_ATTR_RE = re.compile(
    r'((?:fill|stroke|color|style)\s*=\s*)(["\'])(.*?)\2',
    re.IGNORECASE | re.DOTALL,
)


def _replace_white_in_attrs(content):
    def _sub(m):
        prefix, quote, val = m.group(1), m.group(2), m.group(3)
        new_val = _WHITE_RE.sub('#000000', val)
        return f'{prefix}{quote}{new_val}{quote}'

    return _ATTR_RE.sub(_sub, content)


# ── Step 2 – make black/default background elements white ───────────────────

def _parse_dim(val):
    if val is None:
        return None
    try:
        return float(re.sub(r'[a-zA-Z%]', '', val))
    except ValueError:
        return None


def _attr_val(attrs, name):
    m = re.search(
        rf'\b{re.escape(name)}\s*=\s*(["\'])([^"\']*)\1', attrs, re.IGNORECASE
    )
    return m.group(2) if m else None


def _get_svg_size(content):
    m = re.search(r'<svg\b([^>]*?)>', content, re.IGNORECASE | re.DOTALL)
    if not m:
        return None, None
    attrs = m.group(1)

    w = _parse_dim(_attr_val(attrs, 'width'))
    h = _parse_dim(_attr_val(attrs, 'height'))

    if not (w and h):
        vb = _attr_val(attrs, 'viewBox')
        if vb:
            parts = vb.split()
            if len(parts) == 4:
                w = _parse_dim(parts[2])
                h = _parse_dim(parts[3])
    return w, h


def _effective_fill(attrs):
    """Return fill colour from style= or fill= attribute, or None if unset."""
    style = _attr_val(attrs, 'style')
    if style:
        fm = re.search(r'\bfill\s*:\s*([^;]+)', style, re.IGNORECASE)
        if fm:
            return fm.group(1).strip()
    return _attr_val(attrs, 'fill')


def _fill_is_dark_or_unset(fill):
    """True when fill is black/unset (SVG default fill is black)."""
    return fill is None or _is_black(fill)


# ── rect helpers ─────────────────────────────────────────────────────────────

def _is_bg_rect(attrs, svg_w, svg_h):
    """True when the rect sits at (0,0) and covers ≥90% of the SVG canvas."""
    x = _parse_dim(_attr_val(attrs, 'x') or '0')
    y = _parse_dim(_attr_val(attrs, 'y') or '0')
    if x != 0 or y != 0:
        return False
    w = _parse_dim(_attr_val(attrs, 'width'))
    h = _parse_dim(_attr_val(attrs, 'height'))
    if not (w and h):
        return False
    if svg_w and svg_h:
        return (w / svg_w) >= 0.9 and (h / svg_h) >= 0.9
    return True


def _set_fill_white(element_str, tag):
    """Set fill to #ffffff on a <rect> or <path> element string."""

    def _fix_style(m):
        quote, style = m.group(1), m.group(2)
        new_style = re.sub(
            r'\bfill\s*:\s*[^;]+', 'fill: none', style, flags=re.IGNORECASE
        )
        if new_style == style:
            new_style = new_style.rstrip('; ') + '; fill: none'
        return f'style={quote}{new_style}{quote}'

    # 1) Modify fill inside an existing style="…"
    new = re.sub(
        r'style=(["\'])([^"\']*)\1', _fix_style, element_str, flags=re.IGNORECASE
    )
    if new != element_str:
        return new

    # 2) Replace a standalone fill= attribute
    new = re.sub(
        r'\bfill\s*=\s*(["\'])[^"\']*\1', 'fill="none"',
        element_str, flags=re.IGNORECASE
    )
    if new != element_str:
        return new

    # 3) No fill at all – insert one right after the opening tag name
    return element_str.replace(f'<{tag}', f'<{tag} fill="none"', 1)


# ── path helpers ─────────────────────────────────────────────────────────────

def _parse_rect_path(d):
    """
    Return (x_min, y_min, width, height) if `d` describes a 4-corner rectangle
    (M … L … L … L … z), otherwise return None.
    """
    # Normalise whitespace and strip the closing z/Z
    d = re.sub(r'\s+', ' ', d.strip())
    d_clean = re.sub(r'\s*[zZ]\s*$', '', d).strip()

    # Accept only M + exactly 3 L commands (4 corners)
    token_re = re.compile(r'([MLml])\s*([-\d.e]+)\s+([-\d.e]+)', re.IGNORECASE)
    tokens = token_re.findall(d_clean)
    if len(tokens) != 4:
        return None
    if tokens[0][0].upper() != 'M':
        return None
    if not all(t[0].upper() == 'L' for t in tokens[1:]):
        return None

    try:
        pts = [(float(t[1]), float(t[2])) for t in tokens]
    except ValueError:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def _is_bg_path(attrs, svg_w, svg_h):
    """True when the path forms a rectangle whose area is ≥5% of the SVG canvas.

    The threshold is intentionally low so it catches both the full figure
    background (patch_1, origin at 0,0) and sub-area axes backgrounds
    (patch_2, origin elsewhere) while ignoring tiny decorative shapes.
    """
    d = _attr_val(attrs, 'd')
    if not d:
        return False
    result = _parse_rect_path(d)
    if result is None:
        return False
    _x, _y, w, h = result
    if svg_w and svg_h:
        canvas_area = svg_w * svg_h
        if canvas_area == 0:
            return False
        return (w * h / canvas_area) >= 0.05
    # If we can't determine canvas size, accept any reasonable rectangle
    return w > 0 and h > 0


# ── combined scanner ─────────────────────────────────────────────────────────

_RECT_RE = re.compile(
    r'<rect\b([^>]*?)(?:/>|>.*?</rect\s*>)',
    re.IGNORECASE | re.DOTALL,
)

_PATH_RE = re.compile(
    r'<path\b([^>]*?)(?:/>|>.*?</path\s*>)',
    re.IGNORECASE | re.DOTALL,
)


def _make_backgrounds_white(content):
    svg_w, svg_h = _get_svg_size(content)
    replacements = []

    for m in _RECT_RE.finditer(content):
        attrs = m.group(1)
        fill = _effective_fill(attrs)
        if _fill_is_dark_or_unset(fill) and _is_bg_rect(attrs, svg_w, svg_h):
            replacements.append(
                (m.start(), m.end(), _set_fill_white(m.group(0), 'rect'))
            )

    for m in _PATH_RE.finditer(content):
        attrs = m.group(1)
        fill = _effective_fill(attrs)
        if _fill_is_dark_or_unset(fill) and _is_bg_path(attrs, svg_w, svg_h):
            replacements.append(
                (m.start(), m.end(), _set_fill_white(m.group(0), 'path'))
            )

    # Apply in reverse order so earlier offsets stay valid
    replacements.sort(key=lambda r: r[0])
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + content[end:]

    return content, len(replacements)


# ── Step 3 – extract colormap from QuadMesh / PathCollection ────────────────

_HEX6_RE = re.compile(r'#[0-9A-Fa-f]{6}')
_HEX3_RE = re.compile(r'#[0-9A-Fa-f]{3}(?![0-9A-Fa-f])')


def _hex_to_rgb01(h):
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r / 255.0, g / 255.0, b / 255.0


def _group_content(content, group_id):
    """Return the inner content of <g id="group_id">…</g>, or empty string."""
    pattern = re.compile(
        rf'<g\s[^>]*id=["\']{{0,1}}{re.escape(group_id)}["\']{{0,1}}[^>]*>(.*?)</g>',
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(content)
    return m.group(1) if m else ''


def extract_cmap_colors(content):
    """
    Extract colormap colors in order from the SVG.

    Strategy (in priority order):
      1. QuadMesh_1 group  – Matplotlib colorbar cells, already in cmap order.
      2. PathCollection groups – bubble fill colors (deduplicated, sorted by
         first appearance which correlates with data value order).

    Returns a list of hex color strings in colormap order.
    """
    # 1) Try QuadMesh (colorbar) – most reliable source
    qm = _group_content(content, 'QuadMesh_1')
    if qm:
        colors = []
        for m in re.finditer(
            r'<path\b[^>]*style=["\'][^"\']*(fill:\s*(#[0-9A-Fa-f]{3,6}))', qm,
            re.IGNORECASE,
        ):
            c = m.group(2)
            if c not in colors:
                colors.append(c)
        if colors:
            return colors

    # 2) Fall back to PathCollection bubble fills (unique, document order)
    colors = []
    pc_re = re.compile(r'<g\s[^>]*id=["\']PathCollection_\d+["\'][^>]*>(.*?)</g>',
                       re.DOTALL | re.IGNORECASE)
    for pc_m in pc_re.finditer(content):
        for path_m in re.finditer(
            r'<path\b[^>]*style=["\'][^"\']*(fill:\s*(#[0-9A-Fa-f]{3,6}))',
            pc_m.group(1), re.IGNORECASE,
        ):
            c = path_m.group(2)
            if c not in colors:
                colors.append(c)

    return colors


# ── Step 4 – replace x-tick labels ──────────────────────────────────────────

def replace_xtick_labels(content, new_labels):
    """
    Replace x-tick labels in the SVG with a list of new labels.
    
    Looks for text elements within XAxis/xtick groups (Matplotlib-specific)
    and replaces them with provided labels in order.
    
    Returns:
        Tuple of (modified_content, number_of_replacements)
    """
    if not new_labels or len(new_labels) == 0:
        return content, 0
    
    # Strategy 1: Find text in XAxis group (Matplotlib standard)
    xaxis_pattern = re.compile(
        r'<g\s[^>]*id=["\']XAxis["\'][^>]*>(.*?)</g>',
        re.DOTALL | re.IGNORECASE,
    )
    xaxis_match = xaxis_pattern.search(content)
    
    if xaxis_match:
        xaxis_content = xaxis_match.group(1)
        text_pattern = re.compile(
            r'(<text\b[^>]*>)([^<]*)(</text>)',
            re.IGNORECASE | re.DOTALL,
        )
        
        replacements = []
        label_idx = 0
        offset_adjustment = 0
        
        for m in text_pattern.finditer(xaxis_content):
            if label_idx >= len(new_labels):
                break
            
            opening_tag = m.group(1)
            text_content = m.group(2).strip()
            closing_tag = m.group(3)
            
            # Replace non-empty text with new label
            if text_content:
                new_text = new_labels[label_idx]
                old_match_str = m.group(0)
                new_match_str = f'{opening_tag}{new_text}{closing_tag}'
                xaxis_content = xaxis_content[:m.start() + offset_adjustment] + new_match_str + xaxis_content[m.end() + offset_adjustment:]
                offset_adjustment += len(new_match_str) - len(old_match_str)
                label_idx += 1
        
        # Replace the XAxis group content in the original content
        if label_idx > 0:
            new_xaxis_group = f'<g id="XAxis"{xaxis_match.group(0)[18:-4]}>{xaxis_content}</g>'
            content = content[:xaxis_match.start()] + new_xaxis_group + content[xaxis_match.end():]
            return content, label_idx
    
    # Strategy 2: Fall back to finding all short text elements (last resort)
    # This searches for text that looks like tick labels (short, numeric-like)
    text_pattern = re.compile(
        r'(<text\b[^>]*>)([^<]*)(</text>)',
        re.IGNORECASE | re.DOTALL,
    )
    
    matches = list(text_pattern.finditer(content))
    replacements = []
    label_idx = 0
    
    # Filter to likely xtick labels: short text, typically in the lower portion
    # Look for numeric or short alphanumeric content
    tick_like_texts = []
    for i, m in enumerate(matches):
        text_content = m.group(2).strip()
        # Tick labels: short (< 50 chars), non-empty, and not obviously titles/labels
        if text_content and len(text_content) < 50 and not any(x in text_content.lower() for x in ['unit', 'molecule', 'ratio']):
            tick_like_texts.append((m, i))
    
    # Use the last N matches if we have enough (xticks are usually at the end of SVG)
    if len(tick_like_texts) >= len(new_labels):
        tick_like_texts = tick_like_texts[-len(new_labels):]
    
    # Apply replacements in reverse order to preserve offsets
    for m, _ in reversed(tick_like_texts[:len(new_labels)]):
        opening_tag = m.group(1)
        closing_tag = m.group(3)
        new_text = new_labels[len(new_labels) - 1 - label_idx]
        replacement = f'{opening_tag}{new_text}{closing_tag}'
        content = content[:m.start()] + replacement + content[m.end():]
        label_idx += 1
    
    return content, label_idx


def print_cmap_code(colors, input_path):
    """Print Python code that recreates the extracted colormap."""
    if not colors:
        print("No cmap colors found; no code to print.", file=sys.stderr)
        return

    base = os.path.splitext(os.path.basename(input_path))[0]
    safe_name = re.sub(r'[^0-9a-zA-Z_]', '_', base)[:40]

    color_list = '[' + ', '.join(f"'{c}'" for c in colors) + ']'
    print()
    print('# ── Colormap extracted from:', os.path.basename(input_path))
    print(f'# Add this to your script (e.g. hsi_visualizer.py):')
    print()
    print('from matplotlib.colors import LinearSegmentedColormap')
    print()
    print(f'_{safe_name}_COLORS = {color_list}')
    print(f'{safe_name}_cmap = LinearSegmentedColormap.from_list(')
    print(f"    '{safe_name}', _{safe_name}_COLORS, N=256")
    print(')')
    print()
    print(f'# Then pass  cmap={safe_name}_cmap  to generate_unit_heatmaps /')
    print('#            generate_unit_bubble_chart / generate_unit_bubble_charts')


# ── main ─────────────────────────────────────────────────────────────────────

def transform_svg(input_path, output_path, cmap_output=None, xtick_labels=None):
    with open(input_path, 'r', encoding='utf-8') as fh:
        content = fh.read()

    # Step 3 (optional): print cmap code BEFORE color transforms
    if cmap_output is not None:
        colors = extract_cmap_colors(content)
        print_cmap_code(colors, input_path)

    # Step 1: white → black
    content = _replace_white_in_attrs(content)

    # Step 2: black/default background elements → white
    content, n_bg = _make_backgrounds_white(content)
    
    # Step 4 (optional): replace x-tick labels
    n_labels = 0
    if xtick_labels:
        content, n_labels = replace_xtick_labels(content, xtick_labels)

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(content)

    print(f"White \u2192 black conversions applied.")
    print(f"Background element(s) made transparent: {n_bg}")
    if n_labels > 0:
        print(f"X-tick labels replaced: {n_labels}")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Transform SVG: white fill/stroke/color \u2192 black, '
            'black background rects \u2192 transparent, '
            'optionally replace x-tick labels.'
        )
    )
    parser.add_argument('input', help='Input .svg file path')
    parser.add_argument(
        '-o', '--output',
        help='Output .svg file path (default: <input>_transformed.svg)',
    )
    parser.add_argument(
        '--cmap',
        action='store_true',
        default=False,
        help='Also extract and print Python code to recreate the colormap.',
    )

    parser.add_argument(
        '--labels',
        nargs='*',
        default=None,
        help='Replace x-tick labels with these custom labels (space-separated).',
    )
    args = parser.parse_args()
    
    # Use command-line labels if provided, otherwise use the global CUSTOM_XTICK_LABELS
    xtick_labels = args.labels if args.labels else CUSTOM_XTICK_LABELS

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output = f'{base}_transformed{ext}'

    cmap_output = True if args.cmap else None
    transform_svg(args.input, output, cmap_output=cmap_output, xtick_labels=xtick_labels)


if __name__ == '__main__':
    main()
