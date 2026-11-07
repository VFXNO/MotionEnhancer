#pragma once

#include "FrameTimeline.h"

// Work ordering only, not a pacing policy: deadlines and k/N positions are
// still owned by FrameTimeline. Real frames must reach Present before FSR
// for their pair is queued on the same GPU queue.
struct FrameOutputPlan {
    bool fsrBeforePresent = false;
    bool fsrAfterPresent = false;
    float sourceDeltaMs = 0.0f;

    static FrameOutputPlan make(const FrameTimeline::Selection& sel, uint32_t multiplier,
                                float window, int64_t sourcePeriod) {
        FrameOutputPlan plan;
        const int64_t delta = sel.f1.ts - sel.f0.ts;
        plan.sourceDeltaMs = static_cast<float>(delta > 0 ? delta : sourcePeriod) / 10000.0f;
        // Odd fixed multipliers never request t=0.5. Do not spend GPU time
        // making a midpoint that cannot be shown (even with a wide window).
        const bool usesMidpoints = multiplier == 0 || multiplier % 2 == 0;
        if (!sel.valid || sel.f0.index == sel.f1.index || sel.f1.sceneCut ||
            sel.f1.flowEntry < 0 || !usesMidpoints) return plan;
        plan.fsrBeforePresent = sel.canUseMidpoint(window);
        plan.fsrAfterPresent = !plan.fsrBeforePresent;
        return plan;
    }
};
