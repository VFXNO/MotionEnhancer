"""Check the actual off-screen logo animation; optionally refresh README media.

Requires Pillow, like make_readme_art.py. No screen capture or installation.
Usage: python docs/tools/check_logo_animation.py --output docs/assets
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[2]


def changed(first, second, region, threshold):
    a, b = first.convert('RGB').load(), second.convert('RGB').load()
    return sum(1 for y in range(first.height) for x in range(first.width)
               if region(x, y) and max(abs(a[x, y][k] - b[x, y][k]) for k in range(3)) > threshold)


def check(executable, output=None):
    work = Path(tempfile.mkdtemp(prefix='logo-animation-', dir=Path(os.environ['LOCALAPPDATA']) / 'Temp/opencode'))
    env = {k: v for k, v in os.environ.items() if not k.startswith('MOTION_ENHANCER_')}
    try:
        def run():
            subprocess.run([str(executable), '--gui'], env=env, cwd=work, check=True, timeout=60)

        env['MOTION_ENHANCER_LOGO_FRAMES'] = str(work / 'loop')
        run()
        env.pop('MOTION_ENHANCER_LOGO_FRAMES')

        def load_frames(pattern):
            frames = []
            for path in sorted(work.glob(pattern)):
                with Image.open(path) as image: frames.append(image.convert('RGB'))
            return frames

        loop = load_frames('loop_[0-9][0-9][0-9][0-9].png')
        sidebar = load_frames('loop_sidebar_[0-9][0-9][0-9][0-9].png')
        assert len(loop) == len(sidebar) == 361, 'Expected a complete 12-second, 30-fps native export'
        assert loop[0].tobytes() == loop[-1].tobytes(), 'Logo jumps at the loop boundary'
        assert sidebar[0].tobytes() == sidebar[-1].tobytes(), 'Native sidebar does not loop seamlessly'

        def delta(a, b):
            return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3

        seam = delta(loop[-2], loop[0])
        adjacent = max(delta(loop[-3], loop[-2]), delta(loop[0], loop[1]))
        assert 0 < seam <= adjacent * 1.5 + 0.05, 'Loop seam has an abrupt motion/brightness change'
        with Image.open(work / 'loop_intro.png') as intro, Image.open(work / 'loop_handoff.png') as ambient:
            assert intro.tobytes() == ambient.tobytes(), 'Splash-to-ambient phase resets at hand-off'
        print('12-second loop, smooth seam and phase-continuous splash hand-off: PASS', flush=True)

        def annulus(inner, outer):
            return lambda x, y: inner ** 2 <= (x - 116) ** 2 + (y - 84) ** 2 <= outer ** 2

        # Restrict checks to the logo, not the separately animated sidebar lines.
        checks = (
            ('orbiting nodes', annulus(42, 53), 20, 10),
            ('circling spark', annulus(27, 35), 20, 10),
            ('motion trails', lambda x, y: 89 <= x <= 112 and 74 <= y <= 94, 3, 5),
            ('breathing glow', annulus(50, 65), 0, 50),
        )
        for name, region, threshold, minimum in checks:
            count = changed(sidebar[0], sidebar[42], region, threshold)
            assert count >= minimum, f'{name} is not animated: only {count} changed pixels'
            print(f'{name}: PASS ({count} changed logo pixels)', flush=True)

        env['MOTION_ENHANCER_SPLASH_FRAMES'] = str(work / 'splash')
        run()
        splash = []
        for path in sorted(work.glob('splash_*.png')):
            with Image.open(path) as image: splash.append(image.convert('RGB'))
        assert len(splash) == 78, 'Expected the full 2.6-second, 30-fps splash'
        assert changed(splash[9], splash[42], lambda x, y: 150 <= x < 310 and 60 <= y < 220, 20) > 100
        print('staged splash build-up: PASS', flush=True)

        if output:
            output.mkdir(parents=True, exist_ok=True)
            # One palette avoids frame-to-frame color shimmer. GIF timing uses
            # centiseconds: 30/30/40 ms preserves the actual 30-fps loop duration.
            def save_gif(name, frames, durations):
                # Include the whole sequence: the splash starts blank, so its
                # first frame alone cannot represent the later white artwork.
                samples = min(16, len(frames))
                atlas = Image.new('RGB', (128, 128 * samples))
                for i in range(samples):
                    frame = frames[i * (len(frames) - 1) // (samples - 1)]
                    atlas.paste(frame.resize((128, 128), Image.Resampling.NEAREST), (0, 128 * i))
                palette = atlas.quantize(colors=256)
                indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
                path = output / name
                indexed[0].save(path, save_all=True, append_images=indexed[1:],
                                duration=durations, loop=0, optimize=False)
                with Image.open(path) as gif:
                    assert gif.n_frames == len(frames), f'{name}: animation frames collapsed'
                    duration, brightest = 0, 0
                    for i in range(gif.n_frames):
                        gif.seek(i)
                        duration += gif.info['duration']
                        brightest = max(brightest, max(hi for lo, hi in gif.convert('RGB').getextrema()))
                    assert duration == sum(durations), f'{name}: incorrect playback speed'
                    assert brightest >= 240, f'{name}: palette lost the bright logo/text'
            timings = [30, 30, 40] * 120
            save_gif('logo-motion.gif', loop[:-1], timings)
            save_gif('sidebar.gif', sidebar[:-1], timings)
            save_gif('splash.gif', splash, [30, 30, 40] * 26)
            env.pop('MOTION_ENHANCER_SPLASH_FRAMES')
            env['MOTION_ENHANCER_GUI_SNAPSHOT'] = str(work / 'gui.png')
            env['MOTION_ENHANCER_GUI_TIME'] = '1.2'
            run()
            shutil.copyfile(work / 'gui.png', output / 'gui-live.png')
            env['MOTION_ENHANCER_GUI_PAGE'] = '1'
            run()
            shutil.copyfile(work / 'gui.png', output / 'gui-offline.png')
            print(f'Updated actual-app animations and screenshots in {output}', flush=True)
    finally:
        shutil.rmtree(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build-project/Release/motion_enhancer.exe')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    check(args.executable.resolve(), args.output)
