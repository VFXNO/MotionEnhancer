# Label UI: automatic training labels, video sampling and optional AI guidance

Open **AI Trainer → Label UI**, or **Train → Create / edit UI labels…**.
You do **not** need to write JSON or calculate image hashes. This editor prepares
the PNGs and manifest for you: `me-ui-labels-1` for reviewed data, or the separately
opted-in `me-ui-labels-2` when including machine-generated training labels.

EfficientSAM-Ti is now supported for smart selection and automatic mask
proposals. Its pinned labeling-only model was downloaded with separate approval
and tested on generated shapes, not user recordings. No training started and the
existing game runtime/icon detector is unchanged.

## Batch auto-labeling from one HUD example (no per-window Save)

This is different from the older **Reuse template when opening unreviewed
windows** draft tool. The batch tool labels and saves **all compatible prepared
training windows** in a subprocess, including windows you never open. Eligible
machine labels can be exported and trained without individually marking them
reviewed. They remain explicitly **machine-generated**, not verified truth.

1. Restart an older AI Trainer after any running job finishes. Use a separate
   editable project per game/HUD layout, and import windows across your videos
   with **Sample many windows from video**. Three consecutive frames are one
   training input; only the current-frame mask is generated. This processes
   requested prepared samples, not every frame of a 19,000-frame recording.
2. On **one representative training example**, mark actual HUD foreground and
   some **checked background near those shapes**. EfficientSAM clicks/boxes can
   help select shapes. Do not label an entire bounding rectangle as text/UI.
   Review/save this one example, then **Use saved reviewed sample as HUD template**.
   At least four fully known UI and sixteen background 2×2 cells are required.
   A template from validation is deliberately forbidden for batch generation.
3. To use that example for different training recordings, explicitly check
   **Other training recordings use this same game, HUD layout and prepared size**.
   Otherwise only its original recording is eligible. Do not mix HUD scales,
   games, menus or layout variants under this confirmation; pin another example
   or use another project for them.
4. Click **Auto-label all compatible training windows…**, then confirm the scope.
   It generates/saves masks in one operation, not a Save/review dialog per window.
   Reviewed samples, manually saved drafts and validation are never replaced.
   Existing purely machine-generated training revisions can be regenerated.
   If the reviewed template is subsequently saved as a new revision, re-pin it
   and regenerate machine labels before exporting; stale teachers are refused.
5. The result reports **eligible training masks**, **needs review**, and up to
   **eight recommended spot-checks**. **Next uncertain / recommended spot-check**
   navigates those cases. The eight audits are a deterministic coverage aid,
   not proof of precision; uncertain cases may be numerous if the seed is poor.
   Inspect/correct audits and failures, not thousands of unchanged examples.
   Saving an edited sample as an unreviewed draft removes its automatic-export
   eligibility; explicitly reviewing/saving promotes it to reviewed human data.
6. Review a small, varied set from a **separate validation recording**, including
   actual UI **and** background. Validation is never automatically labeled or
   used as the teacher. Use its precision/recall/IoU to measure the trained result;
   automatic-label eligibility or low training loss does not establish accuracy.
7. Enable **Include eligible automatic TRAINING labels**, then **Export training
   dataset…** to a new folder. Without that opt-in, export stays reviewed-only.
8. On **Train**, explicitly enable **Allow machine-generated UI training labels**
   for this new experiment. CLI equivalent: `--allow-auto-ui-labels`. Selecting
   the manifest alone does not grant permission or start training. **Continue**
   restores its saved permission/data; do not replace the dataset of an old run.

### What the automatic labeler actually checks

It is a **local color/temporal propagation heuristic**, not an OCR model, generic
semantic HUD detector, SAM video tracker, or new trained model. It needs no new
download or cloud upload. The reviewed seed gives local foreground/background
color prototypes in small neighborhoods around its actual HUD shapes. Default
**HUD neighborhood** padding is 16 pixels (0–64). Those neighborhoods constrain
search; their rectangles are **never directly filled as labels**.

Each region needs at least 60% of its original foreground **and background**
anchors color-matched in all three target frames. Within an active region, all
three frames must agree with the foreground or background color class, within
the RGB tolerance (default 12), and stay at least eight additional color levels
away from the opposite class. This can find changing glyph/icon shapes with
familiar colors inside the same HUD neighborhood, not just copy old glyph pixels.
Exactly matching reviewed foreground edge pixels retain their saved coverage;
new color-segmented shapes are binary pseudo-labels, not estimated edge opacity.

Outside these neighborhoods, unrecognized colors, ambiguous classes and
temporally disagreeing pixels stay **unknown**; they are not declared background.
An eligible mask needs at least four fully known positive and sixteen negative
2×2 head cells. This is a plumbing/abstention gate, **not calibrated confidence**.
Transparent panels over changing scenery, low-contrast/thin text, different HUD
colors, animations, disappearing HUD and matching stationary scenery remain
failure modes. No real-game accuracy guarantee has been established. If most
windows fail, improve the seed/layout coverage; do not force unknown pixels to
background or bulk-call them reviewed merely to increase the sample count.

Limits: up to **5,000 prepared training windows** per batch, **512 MiB encoded
mask staging**, and **Auto-label limit (minutes)** (30 by default, 5–120). Memory
is bounded per sample, not a stack of thousands of RGB frames. Cancel, timeout,
changed source/project metadata, failed staging or invalid results accepts no
new label revisions. Existing frame/annotation files are never overwritten;
successful batches publish fresh revisions in one project update.

## Faster workflow for many examples per video

**Three frames are one training sample, not a limit of three frames per video.**
Only the last frame in each triplet needs labels. You can now prepare many
samples in one operation, and reuse unchanged HUD shapes while reviewing them.
Restart an older AI Trainer to load these controls **after any running job has
finished or you have explicitly stopped it**. This update does not stop a job.

1. Create/open your editable labeling project and choose a video, recording ID
   and split. Related footage stays in one split; validation needs a different
   original recording.
2. Set **Current frame** to the first desired current frame (usually **2**),
   **Windows across video** to **20** to start, and leave **Last current frame**
   at **End of video (metadata)**. Click **Sample many windows from video**.
   This prepares evenly spaced three-frame windows, not just the first triplet.
   You can instead enter an explicit last ordinal for a shorter section; it is
   inclusive and must be at least Current frame. Each triplet is consecutive;
   samples within this operation do not overlap. Short clips produce fewer.
   **Video time limit (minutes)** defaults to **30**, adjustable from **5 to 120**
   for long recordings. It applies to both video-import buttons, not AI inference.
3. Review and save one representative sample, using painting or EfficientSAM.
   Label actual text/icon shapes, **and checked background**, not only UI.
   Click **Use saved reviewed sample as HUD template**. Nothing else is changed.
4. Enable **Reuse template when opening unreviewed windows**, then select another
   window from the same recording at the same prepared size. Or leave automatic
   reuse off and click **Apply HUD template to this window** when wanted.
   - Only saved, reviewed **UI foreground** is reused, only into unknown pixels.
     Existing labels win. Background, unknown pixels and the review flag are
     **never** transferred. Already-reviewed windows are not modified.
   - A pixel is copied only if its RGB values match the template's current-frame
     pixel within **RGB change tolerance** in **all three target frames**. The
     default is 12 levels per channel (range 0–32); 0 requires an exact match.
     This is a conservative editing filter, **not a UI classifier or tracking**.
     It may omit changing text, transparent edges, animations or displaced HUD.
     Coincidentally matching scenery can still be wrong; inspect every result.
   - Reuse produces an **unsaved, undoable draft** and clears review. It does not
     silently write labels for unopened windows. Correct omissions/mistakes,
     review useful background, and leave uncertain regions unknown. Mark all
     remaining pixels as background only after actually checking them for UI.
5. Check **I reviewed the current-frame labels…**, then click **Save reviewed and
   next unreviewed**. This saves only that sample and opens the next unreviewed
   window (wrapping through the list); automatic template reuse prepares its
   draft if compatible. It never checks the review box on your behalf.
6. Repeat with varied scenes, and with separate validation recordings. Choose
   a new reviewed template for a different HUD/recording. Templates are kept in
   memory for this session and cleared when changing projects; saved drafts
   retain template revision/hash provenance. Export to a new snapshot as usual.

Batch limits are **1–100 windows**, **512 MiB** of retained raw/encoded prepared
frames, and the selected **Video time limit (minutes)** (default **30**; **5–120**).
AI suggestions, image imports and model downloads retain their **five-minute**
deadline. Video imports no longer retrieve every intervening frame as a color
image: only selected triplets are retrieved/converted. Sequential decoding still
visits intervening frames to keep exact ordinal IDs; this is not approximate
keyframe seeking and may still take time for a long/high-resolution recording.
Resizing prepared copies reduces storage/annotation cost, not the source codec's
decode resolution. Increase the video time limit or narrow **Last current frame**
if decoding exceeds it; use fewer windows or smaller prepared copies for the
size limit. Cancel remains available. No source file is modified, no folder is
scanned, and no teacher, AI scan, download or training starts as part of bulk import. Sampling many
windows from one recording does not replace collecting different recordings.

Cancellation, timeout or a failed worker accepts **no** results. If the decoder
ends early or the size limit is reached, fully prepared windows may be accepted
with a visible warning. Frame-count metadata is only an estimate; files without
usable metadata require an explicit last ordinal. Before accepting a completed
batch, the editor validates all staged references/split guards; one invalid
window refuses the entire import. New windows are published in one project
update and begin unknown/unreviewed. Existing recording/frame triplets are
skipped, not replaced, even if a different prepared size was requested.

## Manual workflow

1. **New project:** choose a separate **new or empty folder**. Existing datasets,
   runs, models and source footage are not converted or overwritten. **Open
   project** resumes an existing editor project containing `project.json`.
2. **Choose your recording:** select a video, a **current frame number** (zero
   based, at least 2), and a **Training** or **Validation** recording split. Click
   **Read video window**. The tool copies that frame and its two preceding frames
   into the project. It does not search folders or run a flow teacher. Decoding
   starts at the beginning for reliable ordinal frame IDs; late windows in a
   long video can take time. Cancel stops the disposable operation subprocess.
3. Alternatively, **Choose 3 images…** and select each frame in chronological
   order. Confirm that they are genuinely consecutive and from one recording.
   Set the current frame number accordingly. The original recording ID must be
   the same for every related window, renamed copy and derived image sequence.
   For extracted frames, replace the directory-based default ID with the
   original video ID where appropriate; an image folder cannot prove provenance.
4. Keep native dimensions if they are multiples of 16 and at least 64. Otherwise
   explicitly enable **Resize prepared copies** and choose the training/network
   size, e.g. 832×480. This affects only new project copies, not originals. All
   labels are drawn in these prepared coordinates. Resizing may lose fine text;
   inspect the result rather than assuming it preserves glyph detail. Images
   are limited to 4096 pixels per dimension.
5. **Paint the current frame** (choose **Paint labels** above the image):
   - **UI pixels / green:** actual text strokes, icons or panel shapes. Do not
     fill a rectangle surrounding text and its scenery.
   - **Checked background / cyan:** scenery you checked and know is not UI.
   - **Unknown / blue:** unreviewed or ambiguous pixels; training ignores them.
   - Zoom with **+ / − / Fit / 1:1**, and scroll to pan. Radius **0** edits one
     image pixel; larger radii paint continuous strokes. Coverage 255 means a
     fully covered UI pixel; smaller values can describe antialiased edges, not
     uncertainty or panel opacity. Use Unknown for uncertainty.
   - **Undo / Redo** restore both UI coverage and known/unknown status. Inspect
     the previous frames with the view selector; they are deliberately read-only.
   - After locating every UI element, **Mark remaining unknown as checked
     background…** can finish a fully reviewed image. It asks for confirmation:
     missed UI would otherwise become an incorrect background label. Reset to
     Unknown is also confirmed and undoable.
6. **Save annotation** saves a draft. Check **I reviewed the current-frame
   labels…** before saving only when the labels are actually checked. Editing,
   undoing or applying an AI draft clears that review checkbox. Unreviewed
   samples are not exported. Save/discard/cancel prompts protect unsaved edits
   when switching samples, projects or closing the window.
7. Repeat for a **different original recording** assigned to Validation. Keep
   all related footage in one split. Exact frame-copy and renamed identical-video
   leakage across splits are rejected; re-encoded/related footage still needs
   your provenance review. This tool never opens the locked Arabic regression
   capture as labeling input and offers no editable test split.
8. **Export training dataset…:** choose another **new/empty folder**.
   By default it creates independent copies of reviewed frames/labels, `labels.json` with
   hashes and recording splits, and `provenance.json`. At least one reviewed
   sample in both Train and Validation is required. This minimum only verifies
   plumbing; it is not enough to establish useful UI accuracy.
9. **Use exported manifest on Train page** fills the manifest field. It does
   not choose your training mode, launch training, extend a budget or activate
   anything. Follow [UI training](UI_TRAINING.md) to configure a separate new run.

Unknown pixels are safe to leave unreviewed, but a training crop with no fully
known 2×2 cells will fail. Label useful background as well as UI and choose crops
that retain reviewed pixels. Train/validation recordings must also be disjoint
from **all flow recordings** for joint Flow + UI training. For video imports the
default recording ID matches the original flow builder's path identity, enabling
its overlap guard. Related or renamed footage still needs consistent IDs.

## EfficientSAM smart selection and automatic candidates

**You do not have to find an ONNX file or configure normalization.** This adapter
uses the pinned OpenCV EfficientSAM-Ti model with the correct inputs/settings.
Manual labeling still works without it. No model is embedded in the application
package and no download runs merely by opening a page.

On this development machine the approved model is ready in:
`%LOCALAPPDATA%\MotionEnhancer\label-models\efficientsam-ti\`.
The **Label UI → EfficientSAM-Ti available** message means it was found. Other
machines show **Download EfficientSAM-Ti (48.3 MB)…**; that button asks before
downloading, verifies size/SHA256, and retains the license/source metadata.
There is no package installation, account/API key or cloud image upload.
**Verify cached EfficientSAM-Ti…** rechecks existing files without redownloading
valid copies. If a cache file differs from the pin, the tool refuses to overwrite
it; move that file to a backup location and retry verification/download.

### Smart click/box workflow

1. Open your label project and import/select a three-frame window as above.
2. Choose **AI clicks: left + / right −** above the image. Left-click inside one
   UI shape and optionally right-click scenery to exclude it. Use at most six
   clicks. **Clear clicks / box** resets prompts without changing labels.
3. Or choose **AI box: drag around a shape** and draw a tight box around one UI
   element. This is only a segmentation prompt; **the box is never filled into
   the label**. Use separate selections for unrelated UI elements.
4. Click **Generate masks for clicks / box**. The generated mask alternatives
   appear below the image. Click a row to preview it in magenta. No pixels have
   been accepted yet and every candidate checkbox starts unchecked.
5. Check the masks that actually contain UI, then **Add checked masks as UI**.
   Only unknown pixels **inside selected shapes** become draft UI. Existing
   manual labels are preserved and **everything outside stays unknown**. A
   shape score is predicted mask quality, not HUD probability or label accuracy.
6. Switch to **Paint labels** to correct scenery, missed strokes or text edges.
   Changing brush controls also returns to painting. Review and save as above.
   Undo restores both label values and known/unknown status.

### Automatic proposals for the current frame

Choose **Quick scan — 3 × 3 points** or **Denser scan — 5 × 5 points**, then
**Scan current frame for candidate masks**. This scans **only the selected
current frame**, not a folder/video/dataset, and does not use any prior box.
It tries 9 or 25 evenly spaced positive prompts, filters empty, very large and
low predicted-quality shapes, and removes near-duplicate pixel masks. Up to 48
candidates are retained, with a 64 MiB retained-mask budget reducing that count
for large prepared frames. No candidate is automatically checked or accepted.

Preview/check actual UI and add it as above; discard scenery or the entire set
with **Discard candidate masks**. A coarse grid can miss tiny text or produce
background shapes; a denser scan is not a guarantee. Use a tight click/box or
manual painting for missed elements. There is no automatic background label for
all pixels outside proposed UI.

CPU inference is offline work, not a 4 ms real-time path. On generated 192×128
shapes here, GUI-to-worker smart selection took about 3 seconds and a 3×3 scan
about 22 seconds; your CPU/images may differ. Cancel stops the subprocess and
discards that operation's results. The existing five-minute operation limit
also applies. Nothing is trained or activated by making/accepting suggestions.

### Model source and license

- [OpenCV Zoo EfficientSAM](https://github.com/opencv/opencv_zoo/tree/main/models/image_segmentation_efficientsam),
  [pinned OpenCV model repository](https://huggingface.co/opencv/image_segmentation_efficientsam/tree/1e791988b34b2098736003057da0f3e6bc826259).
- `image_segmentation_efficientsam_ti_2025april.onnx`: **48,312,857 bytes**,
  **Apache-2.0**, SHA256
  `4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77`.
- `LICENSE` and `source.json` are retained beside it, outside application builds,
  training runs and installed runtime models. Existing mismatched cache files
  are refused, not overwritten. Downloads use pinned HTTPS URLs with size/hash
  verification and exclusive publication; interrupted subprocess downloads may
  leave a uniquely named `.partial` file that is never treated as a model.
- Only the checksum-pinned embedded-weight graph is supported by this adapter:
  RGB FP32 `[1,3,1024,1024]`, six prompt coordinates and labels, three mask-logit
  alternatives plus predicted IoUs. The image copy is stretched to model size;
  zero-threshold binary masks are restored with nearest-neighbor interpolation.
  Negative clicks use label 0, padding uses -1, box corners use 2/3. No upstream
  Python code is downloaded or executed; inference uses installed OpenCV DNN CPU.
- Generic object segmentation **does not know which objects are game HUD**.
  Especially for thin Arabic/multilingual text, missed/smeared shapes are
  possible and no real HUD accuracy has been established. Human correction and
  explicit review remain mandatory; unsure pixels should remain unknown.

## Other dense AI models / imported probability maps

Expand **Other dense model / external probability PNG…** to use the original
generic adapter. This is separate from EfficientSAM and its foreground-only
candidate acceptance. It supports either:

- **Model… → Suggest current frame with local AI:** run an explicitly selected
  local ONNX segmenter in a separate CPU subprocess. Its SHA256 is pinned on
  selection and checked during use. Select the model's documented normalization
  (**RGB 0..1**, **RGB −1..1**, or **RGB ImageNet**) and whether its output is
  **probabilities** or **logits**. Choosing the wrong settings can produce wrong
  suggestions even if inference succeeds.
- **Import external AI probability PNG…:** use a dense map produced by another
  tool/model. It must be an 8-bit single-channel PNG, 0–255 probability, matching
  the prepared current frame exactly. The editor does not run that external tool
  or upload anything. A mask can be imported as 0/255, but is still only a draft.

Magenta shows **unaccepted suggestions**, separate from saved label pixels.
**Apply AI draft only to unknown pixels** marks predictions ≥0.85 as draft UI,
≤0.15 as draft background, and leaves uncertain predictions unknown. It never
overwrites already-known manual pixels. These confidence bands are fixed editing
safeguards, not validated accuracy thresholds and not changes to the game
detector. Confidence becomes a binary draft label, **not** antialiased coverage
or opacity. Correct text edges manually as needed.

You must inspect/correct the result and explicitly review the sample before
export. A text-only segmenter will not find all icons/panels; a generic object
segmenter does not automatically distinguish HUD from scenery or world-space
text. Do not accept all predictions simply because they are confident. Model
identity, input/output settings, draft thresholds and original source provenance
are retained with the annotation/export. No machine suggestion is silently
treated as verified ground truth.

### Supported generic dense-model contract

- One fixed **FP32 RGB NCHW** input `[1,3,H,W]` (16–4096 pixels per dimension).
- One fixed **FP32 binary dense output** `[1,1,H,W]`, UI/text candidate
  probabilities in `[0,1]` or logits. Output can be lower resolution.
- Self-contained ONNX, at most 512 MiB, standard operators supported by installed
  OpenCV DNN. External tensor files, custom operator domains/local ONNX functions,
  box detectors, multiclass outputs and prompt/multiple-input models are rejected.
  Consequently, arbitrary YOLO/OCR/SAM exports are **not** plug-and-play; use an
  appropriate separately implemented adapter or import their dense output PNG.
- Inference stretches an image copy to the declared model size and restores its
  dense output with bilinear interpolation. A model requiring letterboxing,
  different color order, extra inputs or other preprocessing needs an adapter;
  do not assume that a shape-compatible model is semantically compatible.
- OpenCV DNN CPU backend, no new dependency installation or GPU placement claim.
  Runtime/model incompatibilities fail without accepting a result. Cancel or a
  five-minute operation timeout kills the operation subprocess; no automatic
  retries or dataset-wide inference occur.

AI selection/inference processes only the current explicitly selected frame per
request. Bulk video import prepares the explicitly requested windows without AI.
This is offline labeling, not a real-time inference path or a 4 ms latency claim.

## Files and safety

An editable project contains `project.json`, immutable prepared frame copies and
separately versioned annotation PNGs. Existing annotation revisions are retained;
unsaved edits live only in memory. External project edits are detected before
metadata is replaced; reopen rather than overwrite another editor's changes.

Each exported dataset is a separate snapshot. Editing a project afterward does
not change earlier exported files or their hashes; choose a fresh export folder
for a revision. A failed export may leave an incomplete new folder, but it never
replaces an old dataset, and `labels.json` is published last. No build, packaging,
runtime model staging or installed-application modification happens here.

The repeatable unit suites use generated/mock-only inputs, with a tiny generated
ONNX color-channel selector for the generic adapter. They require no network,
real footage, downloaded model or training optimizer update. Separately approved
real EfficientSAM smoke checks used generated shapes only; none establishes HUD
accuracy:

```powershell
# From the repository root, using the already installed environment:
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = "$env:LOCALAPPDATA\Temp\opencode"
$env:TMP = $env:TEMP
$env:QT_QPA_PLATFORM = 'offscreen'
& tools/ai-train/.venv/Scripts/python.exe tools/ai-train/test_ui_labeling.py -v
& tools/ai-train/.venv/Scripts/python.exe tools/ai-train/test_ui_sam.py -v
& tools/ai-train/.venv/Scripts/python.exe tools/ai-train/test_ui_auto_label.py -v
```
