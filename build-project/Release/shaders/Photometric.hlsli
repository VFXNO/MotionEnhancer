// The brightness mapping that matches a frame pair before motion is
// measured, shared by every estimator that compares luma (PackLumaCS for
// SAD, DisPatchCS and DisDensifyCS for DIS) so they cannot disagree about
// it. `own` and `other` are the two frames' whole-frame luma mean and
// deviation (FrameStatsCS); each frame is mapped onto their average with
//     luma' = gain * luma + offset.
//
// Raw luma fails completely when the brightness changes between the frames
// - an exposure change, a fade, a flash; 12% gain plus 10 levels measured
// 69 px of error at SAD's finest level and was wrong at every level -
// because every candidate is off by the change. Mapping both frames onto
// common statistics cancels any global gain and offset exactly.
//
// Only a real change is compensated. Motion changes the statistics too -
// content entering the frame on a 47 px pan moved the mean by 1.8 levels and
// the deviation by 6% - and on smooth content a correction that small is
// not harmless: a ramp of half a level per pixel matched one level off
// lands two pixels away, and compensating every pair cost a pan 4.8 dB. The
// measured brightness change moved the mean by 25 levels. So the mapping
// fades in between kShiftStart and kShiftFull of mean difference and is
// exactly the identity below it: ordinary content is matched on exactly the
// luma it always was.

// A deviation ratio outside this range is not an exposure change but a
// different picture (a cut the capture did not flag, a menu): match it on
// its own luma rather than stretch one frame by an arbitrary factor.
static const float kPhotoMaxGain = 2.0f;
static const float kPhotoMinDeviation = 2.0f;
static const float kPhotoShiftStart = 6.0f;    // levels of mean difference: over 3x what motion produced
static const float kPhotoShiftFull = 12.0f;

void PairMapping(float2 own, float2 other, out float gain, out float offset)
{
    const float targetMean = 0.5f * (own.x + other.x);
    const float targetDeviation = 0.5f * (max(own.y, kPhotoMinDeviation) + max(other.y, kPhotoMinDeviation));
    gain = targetDeviation / max(own.y, kPhotoMinDeviation);
    offset = targetMean - gain * own.x;
    float strength = saturate((abs(own.x - other.x) - kPhotoShiftStart) / (kPhotoShiftFull - kPhotoShiftStart));
    if (gain > kPhotoMaxGain || gain < 1.0f / kPhotoMaxGain) strength = 0.0f;
    gain = lerp(1.0f, gain, strength);
    offset = lerp(0.0f, offset, strength);
}
