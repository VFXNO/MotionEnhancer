# Label UI: paint the HUD

**AI Trainer → Label UI** makes HUD labels from your gameplay videos for the
optional UI model ([UI_TRAINING.md](UI_TRAINING.md)). Each frame opens with the
HUD already painted; you fix what it got wrong and save.

## Steps

1. **Project → New…** and choose an empty folder (one per game is easiest).
   **Open…** reopens one later; the page also reopens the last project by itself.
2. **Video → Choose video…**, pick its **Split** (Training or Validation),
   how many **Frames** to label, and the size samples are saved at
   (the installed AI model's size, or the original). Press **Load frames**:
   the frames are spread evenly over the video.
3. Each frame opens pre-painted in green. Fix it:
   - **HUD** (B): paint HUD pixels.
   - **Erase** (E): mark pixels as not HUD.
   - **Unsure** (U, orange): pixels you cannot decide; training ignores them.
   - **[** and **]** change the brush size, **H** hides the paint, **F** fits
     the frame. The mouse wheel zooms and the right button pans. **Ctrl+Z**
     undoes a stroke, **Suggest** paints the suggestion again, **Clear** removes
     all paint.
4. **Save & next** (S) stores the frame and opens the next one; **Skip** (N)
   moves on without saving. Saved frames get a ✓ in the list; reopening one
   shows its saved label.
5. Label a second video as **Validation**: training needs both splits, and a
   video always stays in the split it was first saved in.
6. **Use for training** puts the project's `labels.json` into the Train page's
   UI label field.

## What to paint

Paint what stays fixed on the screen: text, icons, bars, the minimap frame,
the crosshair. Leave out markers that move with the world (quest diamonds,
enemy health bars over heads) - the motion handles those. For a see-through
panel, paint its text and icons; mark the panel itself Unsure if in doubt.

## How the suggestion works

The same idea as the program's HUD persistence (`shaders/UiPersistCS.hlsl`):
over the 24 frames before the labeled one, it finds pixels that stay fixed -
edges that keep their gradient, or pixels that keep their colour and stand out
from their surroundings - while the scene around them moves. Enclosed holes are
filled (a bar's inside), and only pixels that stayed the same over the last
frames are kept. Nothing is learned where the camera does not move, so pick
frames with motion.

On lossless synthetic HUDs over moving 1440p gameplay it covers 99-100% of the
opaque HUD; 60-85% of what it paints is HUD (the rest is mostly scenery that
happened to stay still), so check every frame before saving. It takes about
2.5 s per 1440p frame and runs in the background while you paint the previous
one.

## Files

`labels.json` is an `me-ui-labels-1` manifest (see `mefi/ui_data.py`). Each
sample is `samples/<video>-<frame>/` with `frame0-2.png` (the three frames
ending at the labeled one), `coverage.png` (255 = HUD) and `valid.png`
(255 = known, 0 = unsure), each listed with its SHA256. Frames from the locked
test capture are always saved as Test. The page's code is
`mefi/hud_label_gui.py` and `mefi/hud_label.py`; tests: `test_hud_label.py`.
