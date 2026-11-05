#include "Common.hlsli"

// Fills the components of the flow field that the matcher could not
// measure, from the neighbours that could. Per axis, independently.
//
// Why this pass exists. A block only constrains the component of its vector
// along which its own content actually changes: a flat block constrains
// neither, and a vertical line constrains x but says nothing whatever about
// y. On a noise-like background that is a rounding error — 89% of blocks
// measure both axes exactly. On the content this overlay is actually
// pointed at, a page of charts and anti-aliased text on a flat background,
// **77% of blocks measure nothing at all** and many of the rest measure one
// axis. Those blocks previously emitted whatever the inheritance chain
// handed them, which began at zero and, in a column with no horizontal
// structure anywhere, stayed zero. Measured on a 1080p text page: 9.7% of
// blocks correct, 57% wrong by more than 8 px, and the wrong ones were
// overwhelmingly (x, 0) — x found, y collapsed.
//
// Two visible consequences, both reported from live capture:
//   * a flat block beside text emitted ~0 while the text beside it emitted
//     the true vector, and the warp blends the two bilinearly across 8 px,
//     which drags the background and smears the glyph edge. "Artifacts
//     around text".
//   * the larger the motion, the larger that disagreement, so the same
//     defect scales with it. "Too many artifacts on large motion".
//
// This is not smoothing, suppression or regularization. A component that
// was measured is passed through untouched at any confidence — the pass can
// only ever write a component that carried no information, and it writes it
// from neighbours that did. Nothing is penalised, vetoed or zeroed.
//
// Running it at every pyramid level is what gives it reach: a filled coarse
// level becomes the next level's predictor, so a flat region far larger
// than this kernel's radius is still covered, and any finer block that does
// have structure simply re-measures and overrides the fill.
Texture2D<float4> InputFlow : register(t0);
RWTexture2D<float4> OutputFlow : register(u0);

// +-3 blocks. At the finest level that is +-24 px, and coarser levels carry
// the long range; a wider kernel here costs quadratically for reach the
// pyramid already provides more cheaply.
static const int kFillRadius = 3;
// Confidence at which a component counts as fully measured. Below it the
// component is blended toward the neighbourhood in proportion to how little
// it knows, so there is no cliff between "measured" and "filled".
//
// Above one on purpose, which leaves even a fully confident block taking
// about a sixth of its (agreeing, confidence-weighted) neighbourhood. The
// gradient-based confidence measures whether a block *could* resolve an
// axis, not whether it did: a block sitting on a glyph stem has real
// vertical structure and still matches the wrong text row. On a page
// scrolling by a fractional (5.5, 2.5) that overstatement cost 10 dB -
// 19.7 against 30.3 with this floor - because those blocks kept a y they
// had no business keeping.
//
// Raising it further keeps paying (2.0 scores higher on most cases) and is
// deliberately not taken: at 2.0 every block, however well measured,
// surrenders half its vector to its neighbours, which is unconditional
// regularization rather than filling in what was never measured, and the
// moving-object cases show the cost. The honest way to go further is a
// confidence that reflects how sharply the *match* was resolved rather than
// how much gradient the block contains.
static const float kMeasured = 1.2f;
// How far a neighbour's vector may differ from this block's own before it
// stops counting as evidence about this block, in level pixels.
//
// Without this the fill is a plain neighbourhood average and a small object
// loses: the flat interior of a moving ellipse is surrounded by far more
// confident *background* than confident outline, so it gets filled with the
// background's motion. Measured: the two moving-object cases lost 1.8 and
// 3.3 dB to exactly that. Eight pixels is deliberately generous — a
// neighbour disagreeing by a pixel or two still counts fully, and only a
// neighbour on the other side of a genuine motion boundary is discounted.
static const float kFlowSigma = 2.0f;

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const uint2 grid = FlowGridSize(LevelWidth, LevelHeight, (uint)BlockSize);
    if (id.x >= grid.x || id.y >= grid.y) return;
    const int2 gridMax = int2(grid) - 1;

    const float4 self = InputFlow.Load(int3(id.xy, 0));
    const float2 confidence = saturate(self.zw);

    // How much standing this block's own vector has, per axis.
    const float2 known = saturate(confidence / kMeasured);

    // Confidence-weighted neighbourhood mean, per axis. A neighbour counts
    // in proportion to how well *it* measured that axis, how close it is,
    // and how far its vector is from this block's own.
    //
    // That last term is **gated on this block's own confidence**, and the
    // gate is the whole point. Keeping a motion boundary from being averaged
    // across is worth having, but a block that measured nothing has no
    // standing to reject neighbours for disagreeing with it - and ungated it
    // did exactly that, which made the fill unable to correct a wrong value
    // *because* it was wrong. The case that exposed it: when a scroll stops,
    // an untextured block still held the previous frame's vector (the
    // temporal predictor, since removed, was the tie-break reference and
    // nothing in a featureless block outvoted it), its correct neighbours
    // now read zero, and at a 12 px disagreement the ungated term weighted
    // them 0.027. The
    // block kept warping static content, and the frames right after motion
    // stopped lost 15 dB.
    float2 weight = float2(0.0f, 0.0f);
    float2 sum = float2(0.0f, 0.0f);
    [unroll]
    for (int dy = -kFillRadius; dy <= kFillRadius; ++dy)
        [unroll]
        for (int dx = -kFillRadius; dx <= kFillRadius; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            const int2 p = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), gridMax);
            const float4 other = InputFlow.Load(int3(p, 0));
            const float2 difference = other.xy - self.xy;
            const float similar = rcp(1.0f + dot(difference, difference) /
                                             (kFlowSigma * kFlowSigma));
            const float inverse = rcp((float)(dx * dx + dy * dy));
            const float2 w = saturate(other.zw) * lerp(1.0f, similar, known) * inverse;
            weight += w;
            sum += w * other.xy;
        }

    const float2 neighbour = sum / max(weight, 1e-6f);
    // How much of the neighbourhood to take: all of it where this block
    // knows nothing and the neighbourhood knows something, none of it where
    // the block measured the axis itself.
    const float2 available = saturate(weight);
    const float2 take = (1.0f - known) * available;
    const float2 filled = lerp(self.xy, neighbour, take);

    // The component is now as good as what filled it, so a later level
    // inherits a predictor that reflects what is actually known about it.
    OutputFlow[id.xy] = float4(filled, max(confidence, take * available));
}
