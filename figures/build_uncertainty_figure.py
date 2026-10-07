"""Portable renderer of supplied display values: stdlib SVG/LaTeX; Pillow PNG.

Reads only waterdrop_residual_display.json. No NumPy, model, evaluator, source
data, statistical fitting or scientific audit is needed to reproduce this figure.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get('REPRODUCTION_OUTPUT', ROOT / 'generated'))
DATA = ROOT / 'results/controls/waterdrop_residual_display.json'
STEM = 'waterdrop_residual_times'
WIDTH, HEIGHT = 470, 390
INK, MUTED, GRID, ZERO = '#263544', '#62717F', '#B9C5CC', '#BFC4CA'
ANCHORS = ['#440154', '#472D7B', '#3B528B', '#2C728E', '#21918C',
           '#27AD81', '#5CC863', '#AADC32', '#FDE725']


def palette(index):
    fraction = index / 255 * (len(ANCHORS) - 1)
    left = min(int(fraction), len(ANCHORS) - 2)
    amount = fraction - left
    a, b = ANCHORS[left], ANCHORS[left + 1]
    rgb = [round(int(a[i:i + 2], 16) * (1 - amount) + int(b[i:i + 2], 16) * amount)
           for i in (1, 3, 5)]
    return '#' + ''.join(f'{v:02X}' for v in rgb)


PALETTE = [palette(i) for i in range(256)]


class Drawing:
    def __init__(self):
        self.items = []

    def text(self, x, y, text, size=8, align='left', bold=False, color=INK, tex=None):
        self.items.append(dict(kind='text', x=x, y=y, text=text, size=size,
                               align=align, bold=bold, color=color, tex=tex or text))

    def circle(self, x, y, r, color, **metadata):
        self.items.append(dict(kind='circle', x=x, y=y, r=r, color=color, **metadata))

    def line(self, x1, y1, x2, y2, color=GRID, width=.4):
        self.items.append(dict(kind='line', x1=x1, y1=y1, x2=x2, y2=y2,
                               color=color, width=width))

    def rect(self, x, y, w, h, color):
        self.items.append(dict(kind='rect', x=x, y=y, w=w, h=h, color=color))


def build(data):
    assert [f['target_frame'] for f in data['frames']] == [7, 502, 1000]
    assert data['particles'] == 997
    rows = [('Predicted', 'residual score', 'predicted_q', 196),
            ('Realized', 'squared residual', 'actual_coordinate_squared_residual', 59)]
    values = [v for f in data['frames'] for _, _, field, _ in rows for v in f[field]]
    assert all(math.isfinite(v) and v >= 0 for v in values)
    positives = [v for v in values if v > 0]
    low, high = math.floor(math.log10(min(positives))), math.ceil(math.log10(max(positives)))
    if low == high:
        high += 1
    # These are display extents, not reported scientific summary statistics.
    coords = [xy for f in data['frames'] for xy in f['observed_positions']]
    bounds = data['metadata_bounds']
    lo = min(bounds[0][0], bounds[1][0], *(v for xy in coords for v in xy))
    hi = max(bounds[0][1], bounds[1][1], *(v for xy in coords for v in xy))
    padding = .017 * (hi - lo)
    lo, hi = lo - padding, hi + padding

    def color(value):
        if value == 0:
            return ZERO
        unit = (math.log10(value) - low) / (high - low)
        assert -1e-12 <= unit <= 1 + 1e-12
        return PALETTE[round(min(1, max(0, unit)) * 255)]

    p = Drawing()
    p.text(8, 376, 'Residual scores and errors across observed motion', 12, bold=True)
    p.text(8, 361, 'WaterDrop | faithful 100k | seed 0, source 3 | all 997 particles', 8.2, color=MUTED)
    panel_width = 124
    lefts = [72, 205, 338]
    for f, x in zip(data['frames'], lefts):
        p.text(x + panel_width / 2, 343, f"Target frame {f['target_frame']}", 9.2, 'center', True)
        p.text(x + panel_width / 2, 330, f"last input: {f['last_observed_input_frame']}", 7.3, 'center', color=MUTED)
    p.text(8, 264, 'Predicted', 8.5, bold=True)
    p.text(8, 252, 'residual score', 7.6)
    p.text(8, 238, 'q', 11, tex=r'$q_i$')
    p.text(8, 127, 'Realized', 8.5, bold=True)
    p.text(8, 115, 'squared', 7.6)
    p.text(8, 104, 'residual', 7.6)
    p.text(8, 90, 'SE / 2', 9, tex=r'$\mathrm{SE}_i/2$')
    for row_index, (_, _, field, y) in enumerate(rows):
        for column, (f, x) in enumerate(zip(data['frames'], lefts)):
            positions, scalar = f['observed_positions'], f[field]
            assert len(positions) == len(scalar) == 997
            tx = lambda value: x + (value - lo) / (hi - lo) * panel_width
            ty = lambda value: y + (value - lo) / (hi - lo) * panel_width
            p.rect(x, y, panel_width, panel_width, '#FAFBFC')
            for b in (bounds[0][0], bounds[0][1]):
                p.line(tx(b), ty(bounds[1][0]), tx(b), ty(bounds[1][1]))
            for b in (bounds[1][0], bounds[1][1]):
                p.line(tx(bounds[0][0]), ty(b), tx(bounds[0][1]), ty(b))
            for particle, (xy, value) in enumerate(zip(positions, scalar)):
                p.circle(tx(xy[0]), ty(xy[1]), .58, color(value),
                         particle=particle, panel=[row_index, column])
            if row_index == 1:
                for b in (bounds[0][0], bounds[0][1]):
                    p.text(tx(b), y - 8, f'{b:.1f}', 6.8, 'center', color=MUTED)
            if column == 0:
                for b in (bounds[1][0], bounds[1][1]):
                    p.text(x - 5, ty(b), f'{b:.1f}', 6.8, 'right', color=MUTED)
    bar_x, bar_y, bar_w, bar_h = 165, 26, 281, 8
    p.text(8, 31, 'Shared logarithmic scale', 7.7, bold=True)
    p.text(8, 19, 'normalized error per coordinate', 7.1, color=MUTED)
    for i in range(256):
        p.rect(bar_x + i / 256 * bar_w, bar_y, bar_w / 256 + .02, bar_h, PALETTE[i])
    for exponent in range(low, high + 1):
        x = bar_x + (exponent - low) / (high - low) * bar_w
        p.line(x, bar_y - 2, x, bar_y, color=MUTED)
        p.text(x, bar_y - 10, '1e' + str(exponent), 7, 'center',
               tex=rf'$10^{{{exponent}}}$')
    p.text(8, 7, 'Same positions and scale in both rows; colors are not rescaled within a panel.', 7, color=MUTED)
    return p, {'log10_color_limits': [low, high], 'shared_spatial_limits': [lo, hi],
               'zero_color': ZERO, 'zero_values': values.count(0),
               'color_quantization': '256 colors interpolated across fixed sequential anchors',
               'particle_glyphs': sum(i['kind'] == 'circle' for i in p.items)}


def write_svg(p):
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}pt" height="{HEIGHT}pt" viewBox="0 0 {WIDTH} {HEIGHT}">',
              '<rect width="100%" height="100%" fill="white"/>']
    for item in p.items:
        c, kind = item['color'], item['kind']
        if kind == 'circle':
            output.append(f'<circle cx="{item["x"]:.6f}" cy="{HEIGHT-item["y"]:.6f}" r="{item["r"]}" fill="{c}"/>')
        elif kind == 'line':
            output.append(f'<line x1="{item["x1"]:.6f}" y1="{HEIGHT-item["y1"]:.6f}" x2="{item["x2"]:.6f}" y2="{HEIGHT-item["y2"]:.6f}" stroke="{c}" stroke-width="{item["width"]}"/>')
        elif kind == 'rect':
            output.append(f'<rect x="{item["x"]:.6f}" y="{HEIGHT-item["y"]-item["h"]:.6f}" width="{item["w"]:.6f}" height="{item["h"]:.6f}" fill="{c}"/>')
        else:
            anchor = {'left': 'start', 'right': 'end', 'center': 'middle'}[item['align']]
            weight = 'bold' if item['bold'] else 'normal'
            output.append(f'<text x="{item["x"]:.6f}" y="{HEIGHT-item["y"]:.6f}" fill="{c}" font-family="Arial,Helvetica,sans-serif" font-size="{item["size"]}" font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="central">{html.escape(item["text"])}</text>')
    return '\n'.join(output + ['</svg>', ''])


def write_native(p):
    colors = {c: f'urc{index}' for index, c in enumerate(dict.fromkeys(i['color'] for i in p.items))}
    out = [r'\begingroup', r'\sffamily', r'\setlength{\unitlength}{1pt}']
    out += [rf'\definecolor{{{name}}}{{HTML}}{{{color[1:]}}}' for color, name in colors.items()]
    out.append(rf'\begin{{picture}}({WIDTH},{HEIGHT})')
    last_color = None
    for item in p.items:
        if item['color'] != last_color:
            out.append(r'\color{' + colors[item['color']] + '}')
            last_color = item['color']
        kind = item['kind']
        if kind == 'circle':
            out.append(rf'\put({item["x"]:.6f},{item["y"]:.6f})' + r'{\circle*{' + str(2*item['r']) + '}}')
        elif kind == 'rect':
            out.append(rf'\put({item["x"]:.6f},{item["y"]:.6f})' + r'{\rule{' + f'{item["w"]:.6f}pt' + '}{' + f'{item["h"]:.6f}pt' + '}}')
        elif kind == 'line':
            x1, y1, x2, y2 = (item[k] for k in ('x1', 'y1', 'x2', 'y2'))
            out += [rf'\linethickness{{{item["width"]}pt}}', rf'\qbezier({x1:.6f},{y1:.6f})({(x1+x2)/2:.6f},{(y1+y2)/2:.6f})({x2:.6f},{y2:.6f})']
        else:
            align = {'left': '[l]', 'right': '[r]', 'center': ''}[item['align']]
            bold = r'\bfseries' if item['bold'] else r'\mdseries'
            font = r'\fontsize{' + str(item['size']) + '}{' + str(item['size']+1) + r'}\selectfont' + bold
            out.append(rf'\put({item["x"]:.6f},{item["y"]:.6f})' + r'{\makebox(0,0)' + align + '{' + font + ' ' + item['tex'] + '}}')
    return '\n'.join(out + [r'\end{picture}', r'\endgroup', ''])


def write_png(p):
    from PIL import Image, ImageDraw, ImageFont
    scale = 4
    image = Image.new('RGB', (WIDTH * scale, HEIGHT * scale), 'white')
    draw = ImageDraw.Draw(image)
    fonts = {}

    def font(size, bold):
        key = size, bold
        if key not in fonts:
            candidates = [('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf'),
                          ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
            for candidate in candidates:
                try:
                    fonts[key] = ImageFont.truetype(candidate, round(size * scale))
                    break
                except OSError:
                    continue
            else:
                fonts[key] = ImageFont.load_default(size=round(size * scale))
        return fonts[key]

    text_bounds = []
    for item in p.items:
        kind, color = item['kind'], item['color']
        if kind == 'circle':
            x, y, r = item['x'] * scale, (HEIGHT-item['y']) * scale, item['r'] * scale
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        elif kind == 'rect':
            x, y, w, h = item['x'] * scale, (HEIGHT-item['y']) * scale, item['w'] * scale, item['h'] * scale
            draw.rectangle((x, y-h, x+w, y), fill=color)
        elif kind == 'line':
            draw.line((item['x1']*scale, (HEIGHT-item['y1'])*scale,
                       item['x2']*scale, (HEIGHT-item['y2'])*scale), fill=color,
                      width=max(1, round(item['width']*scale)))
        else:
            f = font(item['size'], item['bold'])
            anchor = {'left': 'lm', 'right': 'rm', 'center': 'mm'}[item['align']]
            point = item['x']*scale, (HEIGHT-item['y'])*scale
            box = draw.textbbox(point, item['text'], font=f, anchor=anchor)
            assert box[0] >= 0 and box[1] >= 0 and box[2] <= WIDTH*scale and box[3] <= HEIGHT*scale, (item['text'], box)
            draw.text(point, item['text'], font=f, fill=color, anchor=anchor)
            text_bounds.append({'text': item['text'], 'bounds_pixels': box})
    image.save(HERE / (STEM + '.png'))
    return text_bounds


def main():
    global HERE, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA, help='Portable saved display JSON')
    parser.add_argument('--output-dir', type=Path, default=HERE, help='Directory for generated figures')
    parser.add_argument('--skip-png', action='store_true', help='Write SVG and native LaTeX with the Python standard library only')
    args = parser.parse_args()
    DATA, HERE = args.data.resolve(), args.output_dir.resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA.read_text())
    p, display = build(data)
    assert display['particle_glyphs'] == 6 * 997
    native = write_native(p)
    caption = (r'\textbf{Residual uncertainty over observed motion.} '
               r'Columns use the first, middle and last scheduled diagnostic targets; both rows place particles '
               r'at the last observed input frame (all indices are zero-based). Top: current-base $q_i$. '
               r'Bottom: realized vector squared residual divided by two, in the same checkpoint-normalized, '
               r'per-coordinate units. All 997 particles share one logarithmic color scale and spatial scale. '
               r'This fixed example is faithful 100k seed 0, first evaluated source 3, under the no-self-loop '
               r'objective-control convention; it is separate from the 110k continuation and autonomous rollouts. '
               r'The maps reveal spatial structure in a learned error signal; they do not establish calibration or beneficial allocation.')
    (HERE / (STEM + '.svg')).write_text(write_svg(p))
    (HERE / (STEM + '_picture.tex')).write_text(native)
    (HERE / (STEM + '.tex')).write_text('\n'.join([r'\begin{figure*}[!t]', r'\centering', native,
                                                r'\caption{' + caption + '}',
                                                r'\label{fig:residual-times}', r'\end{figure*}', '']))
    bounds = [] if args.skip_png else write_png(p)
    extensions = ('.svg', '.tex', '_picture.tex') if args.skip_png else ('.svg', '.png', '.tex', '_picture.tex')
    output_paths = [HERE / (STEM + ext) for ext in extensions]
    report = {'dimensions_pt': [WIDTH, HEIGHT], **display,
              'display_data_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'figure_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_paths},
              'value_transforms': ['One shared log10 color mapping; 256 sequential colors',
                                   'Same isotropic spatial mapping across panels; no coordinate clipping',
                                   'All particles retained in their original order']}
    (HERE / 'waterdrop_residual_times_metadata.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('WaterDrop residual figure: six panels, 5,982 particle glyphs')



if __name__ == '__main__':
    main()
