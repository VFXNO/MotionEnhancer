#include "FrameTimeline.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr int64_t kSecond = 10000000;
constexpr int64_t kMs = 10000;
// Fraction of the source period within which an arrival counts as "on the
// cadence grid" (or, measured from the previous accepted frame, as an
// update to that same frame).
constexpr double kGridTolerance = 0.3;
// A frame held for deferral is considered real (accepted) unless a newer
// frame supersedes it within this fraction of a period.
constexpr double kDeferWindow = 0.7;
// Steadiness is judged on the newest kCadenceWindow intervals; the rate is
// the mean over kRateWindow (see updateCadence).
constexpr size_t kCadenceWindow = 16;
constexpr size_t kRateWindow = 96;
constexpr size_t kCadenceMinSamples = 8;
// A steady cadence's typical interval must agree with the mean rate to
// within this; otherwise the arrivals are quantized to the refresh grid.
constexpr double kQuantizedSpread = 0.06;
// The newest intervals' mean this far from the long mean: the rate changed.
constexpr double kRateChange = 0.15;
// Unsteady cadence (see regularize): the smoothed clock resyncs to the raw
// timestamp when it is this many periods off, or after a gap this long.
constexpr double kFreeTolerance = 0.75;
constexpr double kFreeGap = 4.0;
// A pair longer than this many source periods is a pause, never stretched
// into interpolated output. Two periods is one missing frame (WGC delivers
// only the newest frame of a composition: 60 fps on 60 Hz with +-1 ms
// present jitter loses frames whenever its phase sits near the vblank), not
// a pause. The old test (gap > 2 periods) sat exactly on that common gap:
// regularization noise flipped such a pair between double-speed motion and
// a "pause" that skipped the real frame and re-anchored the clock.
constexpr uint32_t kMaxPairPeriods = 2;
// Accepted source periods the fixed-N clock measures before its first
// output and before it trusts its own window (see nextOutput, clockPeriod).
// About 0.5 s at 60 fps, 1.3 s at 24 fps.
constexpr uint64_t kClockMinPeriods = 24;

// Mean of the intervals that are neither pauses (over 3x the median) nor
// fragments of one (repaints, under 0.35x): the source's true rate even when
// its frames reach the screen on a refresh grid its rate does not divide
// (1, 1, 1, 1, 2 refreshes for 120 fps on 144 Hz).
int64_t robustMean(std::vector<int64_t> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const int64_t median = values[values.size() / 2];
    double sum = 0.0;
    size_t count = 0;
    for (int64_t v : values) {
        if (v * 100 < median * 35 || v > 3 * median) continue;
        sum += static_cast<double>(v);
        ++count;
    }
    return count ? static_cast<int64_t>(sum / static_cast<double>(count) + 0.5) : median;
}
}

void FrameTimeline::configure(uint32_t sourceFpsOverride, double refreshHz, uint32_t outputMultiplier) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_sourceFpsOverride = sourceFpsOverride;
    m_outputMultiplier = outputMultiplier;
    m_maxOutputPeriod = static_cast<double>(kSecond) / std::max(1.0, refreshHz);
    m_captureTimestampQuantum = static_cast<int64_t>(m_maxOutputPeriod + 0.5);
    if (sourceFpsOverride > 0) {
        m_period = kSecond / static_cast<int64_t>(sourceFpsOverride);
        m_periodMeasured = true;
        m_locked = true;
        updateRefreshGrid(static_cast<double>(kSecond) / sourceFpsOverride, true);
    }
    // Buffer source frames and flow, not display refreshes. This budget is
    // fixed for the source rate; GPU stalls never stretch the output clock.
    applyBudgetLimits();
}

void FrameTimeline::setLatencyLimits(double minExtraMs, double maxMs) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_minLatencyExtra = static_cast<int64_t>(std::max(0.0, minExtraMs) * static_cast<double>(kMs));
    m_maxLatencyOverride = maxMs > 0.0 ? static_cast<int64_t>(maxMs * static_cast<double>(kMs)) : 0;
}

FrameTimeline::Selection FrameTimeline::nextOutput(int64_t now) {
    std::lock_guard<std::mutex> lock(m_mutex);
    Selection sel;
    sel.presentTime = now + kMs; // missing source data: sleep, never fabricate a repeat
    if (m_frames.empty() || !m_captureOffsetValid) return sel;
    // The fixed clock's step and the budget come from the source period.
    // Until it is measured (8 intervals, ~130 ms at 60 fps) m_period is the
    // 30 fps placeholder: the old code anchored and ran the 60 fps clock at
    // 30 fps meanwhile, and the frames that piled up were never given back.
    // Offline (60 fps 2x, every monitor): 128 ms latency instead of 53 ms
    // and all 8 capture slots held, so any extra delay dropped frames; at
    // 24 fps the clock ran 25% fast instead and starved for good. The clock
    // also waits for kClockMinPeriods accepted periods: an estimate over
    // n pairs weighs one miscounted pair or the first timestamp's error by
    // 1/n, and the clock integrates that while the window grows to 96 (from
    // 8 pairs a single late frame drained 1.5 periods, 50 ms at 30 fps).
    // (+8: the window ends at the consumed pair, a few behind the newest.)
    // The first output waits ~0.5 s at 60 fps; the desktop shows meanwhile.
    if (m_outputMultiplier > 0 && !m_clockStarted &&
        (!m_periodMeasured || (m_sourceFpsOverride == 0 && m_clockPeriodSum < kClockMinPeriods + 8))) {
        sel.retireBeforeIndex = m_frames.back().index;
        return sel;
    }

    const int64_t lead = m_captureOffset + m_latencyBudget;
    const double period = clockPeriod();
    const double step = m_outputMultiplier > 0 ? period / m_outputMultiplier : m_maxOutputPeriod;
    size_t outputFrame = 0;
    if (m_outputMultiplier > 0) {
        if (m_outputFrameIndex == 0) m_outputFrameIndex = m_frames.front().index;
        while (outputFrame < m_frames.size() && m_frames[outputFrame].index < m_outputFrameIndex) ++outputFrame;
        if (outputFrame == m_frames.size()) {
            m_nextOutputTime = 0.0; // re-arm on fresh input, not on an expired deadline
            sel.retireBeforeIndex = m_frames.back().index;
            return sel;
        }
        if (m_nextOutputTime == 0.0) {
            // (Re-)arming: start at the first frame whose own deadline is
            // still ahead. Older queued frames are already expired; starting
            // from them would keep their extra latency for the whole session
            // (the clock never runs faster than the source to give it back).
            while (outputFrame + 1 < m_frames.size() && m_frames[outputFrame].ts + lead < now) ++outputFrame;
            m_outputFrameIndex = m_frames[outputFrame].index;
            m_outputPosition = 0;
            m_clockStarted = true;
        }
    }
    if (m_nextOutputTime == 0.0) {
        m_nextOutputTime = static_cast<double>(std::max(now, m_frames[outputFrame].ts + lead));
    }
    // Absolute deadlines retain the fractional 100 ns remainder (120 fps
    // is 83333.333... ticks). Wake-up jitter must not accumulate as drift.
    const uint64_t missed = now + 0.5 >= m_nextOutputTime + step
        ? static_cast<uint64_t>(std::floor((now - m_nextOutputTime + 0.5) / step)) : 0;
    if (missed) {
        m_nextOutputTime += static_cast<double>(missed) * step;
        m_stats.missedOutputs += missed;
    }

    if (m_outputMultiplier == 0) {
        // Max alone requests display-rate FPS, on the same simple timer;
        // it does not wait for a compositor tick or predict a scanout time.
        sel.fixedCadence = false;
        sel.presentTime = static_cast<int64_t>(m_nextOutputTime + 0.5);
        sel.mediaTime = sel.presentTime - lead;
        sel.f0 = sel.f1 = m_frames.back();
        sel.hold = true;
        if (sel.mediaTime < m_frames.front().ts) {
            sel.f0 = sel.f1 = m_frames.front();
            sel.hold = false;
            sel.startup = true;
        } else {
            for (size_t i = 0; i + 1 < m_frames.size(); ++i) {
                const auto& a = m_frames[i];
                const auto& b = m_frames[i + 1];
                if (sel.mediaTime < a.ts || sel.mediaTime >= b.ts) continue;
                sel.f0 = a;
                sel.f1 = b;
                sel.alpha = static_cast<float>(static_cast<double>(sel.mediaTime - a.ts) /
                                               std::max<int64_t>(1, b.ts - a.ts));
                sel.hold = false;
                break;
            }
        }
    } else {
        // Each pair owns N x periods slots of the clock: N for consecutive
        // source frames, 2N when one frame in between never arrived. The old
        // code gave every pair N slots, so a missing frame played its two
        // periods of motion at double speed and left the queue one period
        // short (--timeline-self-test model, 60 fps 2x with 1% game
        // hitches: 13-49 content jumps in 27 s on 60-240 Hz, now none).
        const uint32_t n = m_outputMultiplier;
        size_t i = outputFrame;
        uint64_t position = static_cast<uint64_t>(m_outputPosition) + missed;
        for (;;) {
            const bool haveNext = i + 1 < m_frames.size();
            if (haveNext && m_frames[i + 1].periods == 0) {
                // Never stretch one frame's motion across a source pause.
                if (position == 0) {
                    // Show this last real frame on time (the old code skipped
                    // it whenever the next frame had already arrived), then
                    // restart the clock at the next frame's own deadline.
                    sel.f0 = sel.f1 = m_frames[i];
                    sel.mediaTime = sel.f0.ts;
                    sel.presentTime = static_cast<int64_t>(m_nextOutputTime + 0.5);
                    m_outputFrameIndex = m_frames[i + 1].index;
                    m_outputPosition = 0;
                    m_nextOutputTime = std::max(m_nextOutputTime + step,
                                                static_cast<double>(m_frames[i + 1].ts + lead));
                    sel.valid = true;
                    sel.retireBeforeIndex = sel.f0.index;
                    m_currentF1Index = sel.f1.index;
                    return sel;
                }
                ++i;
                position = 0;
                m_nextOutputTime = static_cast<double>(std::max(now, m_frames[i].ts + lead));
                continue;
            }
            if (!haveNext) {
                if (position == 0) break; // the real frame needs nothing newer
                m_outputFrameIndex = m_frames[i].index;
                if (position >= n) {
                    // A paused source or blocked Present exhausted the input
                    // (a whole period past the last frame: the fixed budget
                    // covers a missing frame without getting here). Wait for
                    // fresh frames rather than replaying this queue in a
                    // burst; they restart the clock at their own deadline.
                    m_outputFrameIndex = m_frames.back().index + 1;
                    m_outputPosition = 0;
                    m_nextOutputTime = 0.0;
                    sel.retireBeforeIndex = m_frames.back().index;
                    return sel;
                }
                // The next frame is not here yet: keep this deadline and
                // position.
                m_outputPosition = static_cast<uint32_t>(position);
                sel.retireBeforeIndex = m_frames[i].index;
                return sel;
            }
            const uint64_t slots = static_cast<uint64_t>(n) * m_frames[i + 1].periods;
            if (position < slots) break;
            position -= slots;
            ++i;
        }
        const bool haveNext = i + 1 < m_frames.size();
        const uint32_t slots = haveNext ? n * m_frames[i + 1].periods : n;
        m_outputPosition = static_cast<uint32_t>(position);
        m_outputFrameIndex = m_frames[i].index;
        sel.f0 = m_frames[i];
        sel.f1 = haveNext ? m_frames[i + 1] : sel.f0;
        sel.alpha = static_cast<float>(m_outputPosition) / static_cast<float>(slots);
        sel.mediaTime = sel.f0.ts + static_cast<int64_t>(
            static_cast<double>(sel.f1.ts - sel.f0.ts) * m_outputPosition / slots);
        sel.presentTime = static_cast<int64_t>(m_nextOutputTime + 0.5);
        if (++m_outputPosition == slots && haveNext) {
            m_outputPosition = 0;
            m_outputFrameIndex = sel.f1.index;
        }
    }
    sel.valid = true;
    sel.retireBeforeIndex = sel.f0.index;
    m_currentF1Index = sel.f1.index; // keep a reserved pair safe from producer replacement
    m_nextOutputTime += step;
    return sel;
}

// The fixed-N clock consumes source periods, so its period must be measured
// in the same units: the accepted pairs' raw WGC gaps over the source
// periods they span, newest kRateWindow pairs. m_period (the capture-side
// cadence) averages every non-duplicate WGC arrival, including repaints
// that are then deferred, dropped or replaced: a UI repaint splits one
// interval in two and pulls that mean low. That cost nothing while output
// sampled timestamps, but a free-running clock integrates the bias: offline,
// 60 fps 2x with two UI repaints per second drifted 55-75 ms within a
// minute (starving, or filling all 8 capture slots and dropping frames).
// A plain sum over a sliding window keeps the clock locked to the frames it
// consumes: the summed gaps telescope, so timestamp quantization never
// accumulates into drift, and a miscounted pair is paid back within the
// window. Raw timestamps, not regularized ones: regularize() runs on
// m_period and inherits its errors (59.94 fps on 60 Hz: m_period read 21 ms
// while frames came every 16.7, and the clock followed).
// The window ends at the pair the clock is consuming, not at the newest
// frame: a frame that reached WGC a refresh late is followed half a period
// later by an on-time one, so as the newest entry its late timestamp
// counted for half a period, as the oldest for a whole one. That biased
// the clock fast (offline, 120 fps on 240 Hz: -0.056%, 17 ms of lead lost
// in 40 s). Ending at the consumed pair weighs every pair by the slots it
// gets, and a miscounted pair is repaid over exactly the pairs after its
// own slots. --source-fps keeps its exact period.
double FrameTimeline::clockPeriod() const {
    if (m_sourceFpsOverride > 0) return static_cast<double>(kSecond) / m_sourceFpsOverride;
    const uint64_t limit = m_clockStarted ? m_outputFrameIndex + 1 : UINT64_MAX;
    int64_t gaps = 0;
    uint64_t periods = 0;
    size_t used = 0;
    for (auto it = m_clockGaps.rbegin(); it != m_clockGaps.rend() && used < kRateWindow; ++it) {
        if (it->index > limit) continue;
        gaps += it->gap;
        periods += it->periods;
        ++used;
    }
    // Not from fewer pairs than the first output waits for: a span over n
    // pairs carries its first timestamp's quantization (up to a refresh)
    // divided by n, and the clock integrates that while the window grows.
    if (periods >= kClockMinPeriods) return static_cast<double>(gaps) / static_cast<double>(periods);
    return static_cast<double>(m_period);
}

uint32_t FrameTimeline::pairPeriods(int64_t rawGap, bool lostFrame) const {
    const double period = clockPeriod();
    const double x = static_cast<double>(rawGap) / period;
    const double periods = std::floor(x + 0.5);
    if (periods > kMaxPairPeriods) return 0; // a pause
    if (periods <= 1.0) return 1;
    // A frame the capture side itself lost is certain. Otherwise a frame
    // that merely reached the screen a refresh late must stay one period, or
    // its pair plays at half speed: 55-58 fps on 60 Hz arrives after 1 or 2
    // refreshes (an empty refresh every 10-30 frames; 57 fps even passes as
    // a locked cadence), offline 9 -> 165 slow pairs a minute when counted.
    // A second period counts only from 1.75 periods (as regularize() counts
    // a skip) and beyond one period plus a whole refresh of quantization,
    // which a late frame cannot reach: 60 fps on 120-240 Hz (locked or not)
    // and 30 fps on 60 Hz still show their lost frames.
    if (lostFrame) return static_cast<uint32_t>(periods);
    // A rate within 2% of a refresh multiple (59-61 fps on 60 Hz) has no
    // such margin: its present phase creeps across the vblank (for seconds
    // at 59.94 fps), and with +-1 ms jitter WGC loses about every fifth
    // frame meanwhile (two presents in one composition), next to empty
    // refreshes. Counted as one period, a few losses in a row drained the
    // lead into starvation and the restarted clock then overfilled all 8
    // slots (offline: 130-230 ms latency for the rest of the run). Every
    // refresh counts there; the cost is one half-speed pair per genuinely
    // empty refresh (59 fps: about one a second; 58 fps and below are
    // counted as above).
    // Still from 1.75 periods only: 60 fps on 120 Hz reaches WGC one
    // refresh late as a 1.5-period pair holding one period of content
    // (rounded up, each such frame added a period of latency for good under
    // --source-fps 60 until all 8 slots were held).
    if (x < 1.75) return 1;
    if (m_refreshGridSource) return static_cast<uint32_t>(periods);
    const double lateLimit = period + static_cast<double>(m_captureTimestampQuantum + kMs / 2);
    return static_cast<double>(rawGap) > lateLimit ? static_cast<uint32_t>(periods) : 1u;
}

// Whether a source runs at a refresh multiple (within 2%), judged on
// m_period. Entered when m_period says so, locked or not, from the first
// measurement on (59.5 fps on 60 Hz is often unlocked, and counted the
// other way its first losses ran the clock 4% slow: 110-150 ms latency for
// good). Checked once strictly when the rate window first fills: 8
// intervals of 57 fps are often all one refresh (no empty refresh among
// them yet) and read as 16.67 ms. After that it is left only beyond 5% and
// after a whole rate window of such locked updates: a burst of losses
// unlocks the cadence and throws m_period 25% off exactly when this
// matters, m_period drifts for a few frames before the burst unlocks it,
// and m_period of 58.5 fps on 60 Hz wanders across 2-4% (each switch
// restarts the measurement, which flipping turned into 150-190 ms latency).
// Not judged on the clock's own period: counting every refresh makes that
// period the refresh, so 57 fps on 60 Hz could never leave it again once a
// clean start had measured 16.67 ms.
void FrameTimeline::updateRefreshGrid(double period, bool locked) {
    const double quantum = static_cast<double>(m_captureTimestampQuantum);
    const double multiple = std::max(1.0, std::floor(period / quantum + 0.5));
    const double offset = std::fabs(period - multiple * quantum) / period;
    const bool near = offset < 0.02;
    m_offGridRun = locked && offset > 0.05 ? m_offGridRun + 1 : 0;
    const bool was = m_refreshGridSource;
    if (!m_refreshGridChecked && (m_sourceFpsOverride > 0 || m_intervals.size() >= kRateWindow)) {
        m_refreshGridChecked = true;
        m_refreshGridSource = near;
    } else if (near) {
        m_refreshGridSource = true;
    } else if (m_offGridRun >= kRateWindow) {
        m_refreshGridSource = false;
    }
    // The window's pairs were counted the other way: keeping them, the
    // clock repaid a difference that was never consumed (57 fps on 60 Hz
    // starting clean at 16.67 ms: lead drained into starvation, the
    // restarted clock then overfilled to 108 ms). Start over on m_period,
    // the capture cadence measured so far.
    if (was != m_refreshGridSource) resetClockPeriod();
}

// The source changed rate (or its period was first measured): pairs counted
// against the old period would be counted wrong. Recount the queued ones.
void FrameTimeline::resetClockPeriod() {
    m_clockGaps.clear();
    m_clockGapSum = 0;
    m_clockPeriodSum = 0;
    for (size_t i = 1; i < m_frames.size(); ++i) {
        if (m_frames[i].periods == 0) continue; // a pause stays a pause
        m_frames[i].periods = pairPeriods(m_frames[i].tsRaw - m_frames[i - 1].tsRaw);
    }
}

// A real rate change, judged on the accepted pairs themselves rather than
// on m_period: during a burst of WGC losses (a ~60 fps game whose present
// phase sits on a 60 Hz vblank) m_period jumped 25% and reset the clock to
// it, although the source never changed. Halved: 12 of the newest 16 pairs
// count two periods. Otherwise the newest 16 pairs' period is 20% off.
bool FrameTimeline::clockRateChanged() const {
    constexpr size_t kRecent = 16;
    if (m_sourceFpsOverride > 0 || m_clockGaps.size() < kRecent || m_clockPeriodSum < kCadenceMinSamples)
        return false;
    int64_t gaps = 0;
    uint64_t periods = 0;
    size_t doubled = 0;
    for (size_t i = m_clockGaps.size() - kRecent; i < m_clockGaps.size(); ++i) {
        gaps += m_clockGaps[i].gap;
        periods += m_clockGaps[i].periods;
        doubled += m_clockGaps[i].periods == 2;
    }
    const double recent = static_cast<double>(gaps) / static_cast<double>(periods);
    const double period = clockPeriod();
    return doubled * 4 >= kRecent * 3 || std::fabs(recent - period) > 0.2 * period;
}

// Fixed source buffering; the monitor has no say in this budget or clock.
void FrameTimeline::applyBudgetLimits() {
    m_minLatencyBudget = m_period + m_minLatencyExtra;
    m_maxLatencyBudget = m_maxLatencyOverride > 0 ? std::max(m_minLatencyBudget, m_maxLatencyOverride)
                                                   : 3 * m_period + 60 * kMs;
    m_latencyBudget = std::clamp(2 * m_period + 10 * kMs, m_minLatencyBudget, m_maxLatencyBudget);
}

// ---------------------------------------------------------------- cadence

void FrameTimeline::observeInterval(int64_t interval) {
    if (interval <= 0 || interval > kSecond) return;
    m_intervals.push_back(interval);
    while (m_intervals.size() > kRateWindow) m_intervals.pop_front();
    updateCadence();
}

void FrameTimeline::updateCadence() {
    if (m_sourceFpsOverride > 0) return;
    if (m_intervals.size() < kCadenceMinSamples) {
        m_locked = false;
        return;
    }
    // Steadiness from the newest intervals: the mean of their middle half,
    // robust against UI ticks and against the alternating short/long
    // intervals of a 24 fps video repainted at 60 Hz, and whether 3/4 of
    // them lie near it.
    const size_t recentCount = std::min(kCadenceWindow, m_intervals.size());
    const std::vector<int64_t> recent(m_intervals.end() - static_cast<std::ptrdiff_t>(recentCount), m_intervals.end());
    std::vector<int64_t> sorted = recent;
    std::sort(sorted.begin(), sorted.end());
    const size_t lo = sorted.size() / 4;
    const size_t hi = sorted.size() - lo;
    double sum = 0.0;
    for (size_t i = lo; i < hi; ++i) sum += static_cast<double>(sorted[i]);
    const int64_t typical = static_cast<int64_t>(sum / static_cast<double>(hi - lo) + 0.5);
    size_t near = 0;
    for (int64_t v : sorted) {
        if (std::llabs(v - typical) <= typical / 4) ++near;
    }
    // The rate from the whole window. A source whose rate the refresh does
    // not divide reaches the screen (and WGC) after 1, 1, 1, 1, 2 refreshes
    // (120 fps on 144 Hz): the typical interval is then one refresh, not the
    // source's period, and pacing by it ran the content 20% fast between
    // corrections. The mean over many intervals is the true rate. When the
    // newest intervals' mean leaves it, the source changed rate: forget the
    // older ones rather than average across the change.
    int64_t rate = robustMean(std::vector<int64_t>(m_intervals.begin(), m_intervals.end()));
    const int64_t recentRate = robustMean(recent);
    if (m_intervals.size() > recentCount &&
        std::llabs(recentRate - rate) > static_cast<int64_t>(kRateChange * static_cast<double>(rate))) {
        m_intervals.erase(m_intervals.begin(), m_intervals.end() - static_cast<std::ptrdiff_t>(recentCount));
        rate = recentRate;
    }
    const bool wasLocked = m_locked;
    const bool wasMeasured = m_periodMeasured;
    const int64_t oldPeriod = m_period;
    // Locked: steady intervals that are the rate itself (not a refresh-grid
    // quantization of it). Unlocked sources are paced by a smoothed clock at
    // the mean rate (regularize).
    m_locked = near * 4 >= sorted.size() * 3 &&
               std::llabs(typical - rate) <= static_cast<int64_t>(kQuantizedSpread * static_cast<double>(rate));
    m_period = std::max<int64_t>(rate, kMs);
    m_periodMeasured = true;
    updateRefreshGrid(static_cast<double>(m_period), m_locked);
    {
        const bool changed = std::llabs(m_period - oldPeriod) > oldPeriod / 5;
        if ((m_locked && !wasLocked) || changed) {
            applyBudgetLimits();
        }
        // Pairs counted against the 30 fps placeholder must not seed the
        // clock. Later rate changes are judged by clockRateChanged().
        if (!wasMeasured) resetClockPeriod();
    }
}

int64_t FrameTimeline::phaseTolerance() const {
    // A frame that reached the screen one refresh late (60 fps on 180 Hz:
    // 5.6 ms, a third of its period) must still count as on the grid, or it
    // is held back and then dropped as a repaint when the next frame comes
    // early. Never under the old 0.3 periods, never so wide that two real
    // frames could be taken for one.
    const int64_t lo = static_cast<int64_t>(kGridTolerance * static_cast<double>(m_period));
    const int64_t hi = static_cast<int64_t>(0.45 * static_cast<double>(m_period));
    return std::clamp<int64_t>(m_captureTimestampQuantum + kMs / 2, lo, hi);
}

bool FrameTimeline::onGrid(int64_t delta) const {
    if (delta <= 0) return false;
    const double k = std::round(static_cast<double>(delta) / static_cast<double>(m_period));
    if (k < 1.0) return false;
    const int64_t predicted = static_cast<int64_t>(k * static_cast<double>(m_period));
    return std::llabs(delta - predicted) <= phaseTolerance();
}

int64_t FrameTimeline::regularize(int64_t tsRaw) {
    if (m_frames.empty() || !m_periodMeasured) {
        m_outliers = 0;
        return tsRaw;
    }
    const double period = static_cast<double>(m_period);
    // Gap in the raw timestamps: pauses and repaints are judged on those.
    const double gap = static_cast<double>(tsRaw - m_lastAcceptedTsRaw) / period;
    // The position along the regularized clock: the frame count k is taken
    // from it, not from the raw gap, so one late timestamp cannot make the
    // next frame look one period early.
    const double periods = static_cast<double>(tsRaw - m_lastAcceptedTs) / period;
    // A source slower than the refresh rarely skips a frame, but its
    // timestamps carry up to a refresh of quantization plus its own present
    // jitter - more than half a period at 120 fps on 144 Hz. Rounding at 0.5
    // then counted two frames where one had passed now and then, putting the
    // frame a whole period late (offline: 3-8 ms steps until the clock had
    // pulled back). Count a skip only from 1.75 periods on; a real one is
    // caught by the outlier rule below (two frames off the clock: resync).
    const double skipBias = (!m_locked && m_period * 10 > m_captureTimestampQuantum * 11) ? 0.25 : 0.0;
    const double k = std::max(1.0, std::floor(periods + 0.5 - skipBias));
    const int64_t predicted = m_lastAcceptedTs + static_cast<int64_t>(k * period);
    const int64_t err = tsRaw - predicted;
    int64_t ts;
    if (gap < 0.5 || gap > kFreeGap) {
        ts = tsRaw;                     // repaint between frames, or a pause: resync
        m_outliers = 0;
    } else {
        // Within the tolerance the clock follows the raw phase slowly.
        // Locked: a steady source. Unlocked: a game rendering at a rate the
        // refresh does not divide reaches the screen (and WGC's timestamps)
        // after 2, 4, 3... refreshes while its content advances evenly; its
        // raw timestamps would pace the in-between frames unevenly, so they
        // follow a smoothed clock at the mean rate. The tolerance covers at
        // least one refresh of quantization.
        // A locked source near the refresh rate (165 fps on 165 Hz) still has
        // up to half a refresh of quantization in every timestamp.
        const int64_t tolerance = m_locked
            ? std::max<int64_t>(phaseTolerance(), m_captureTimestampQuantum / 2 + kMs)
            : std::max<int64_t>(static_cast<int64_t>(kFreeTolerance * period), m_captureTimestampQuantum + kMs / 2);
        if (std::llabs(err) <= tolerance) {
            ts = predicted + err / (m_locked ? 10 : 16);
            m_outliers = 0;
        } else {
            // One timestamp off the clock is a frame that reached the screen
            // late (or a capture-thread stall), not a new phase: resyncing on
            // it moved that frame, and the next one back, by a refresh each.
            // Two in a row on the same side are a real phase change.
            const int sign = err > 0 ? 1 : -1;
            m_outliers = (m_outliers > 0 && sign == m_outlierSign) ? m_outliers + 1 : 1;
            m_outlierSign = sign;
            if (m_outliers >= 2) {
                ts = tsRaw;
                m_outliers = 0;
            } else {
                ts = predicted + std::clamp(err, -tolerance, tolerance) / 4;
            }
        }
    }
    return std::max(ts, m_frames.back().ts + m_period / 10);
}

// --------------------------------------------------------------- producer

FrameTimeline::IngestPlan FrameTimeline::onFrameArrived(int64_t tsRaw, int64_t arrival) {
    std::lock_guard<std::mutex> lock(m_mutex);
    IngestPlan plan;
    if (m_lastArrivalTsRaw > 0) observeInterval(tsRaw - m_lastArrivalTsRaw);
    m_lastArrivalTsRaw = tsRaw;

    int64_t lastRef = m_lastAcceptedTsRaw;
    bool haveLast = !m_frames.empty();
    if (m_deferred) {
        // A later frame superseded the held one unless enough time passed
        // for the held frame to have been a real source frame (phase shift).
        if (tsRaw - m_deferredTsRaw >= static_cast<int64_t>(kDeferWindow * m_period)) {
            plan.acceptDeferredFirst = true;
            lastRef = m_deferredTsRaw;
            haveLast = true;
        } else {
            plan.dropDeferred = true;
            ++m_stats.dropped;
        }
        m_deferred = false;
    }

    if (!haveLast || !m_locked) {
        plan.acceptThis = true;
        return plan;
    }

    const int64_t delta = tsRaw - lastRef;
    const int64_t tolerance = static_cast<int64_t>(kGridTolerance * m_period);
    if (delta < tolerance) {
        // Two frames far closer than the cadence: the later one is a newer
        // version of the same source frame (a UI tick landing just after
        // or just before the video update). Replace unless that frame is
        // already on screen.
        if (!plan.acceptDeferredFirst && m_frames.back().index > m_currentF1Index) {
            plan.replaceLast = true;
            ++m_stats.replaced;
        }
        plan.acceptThis = true;
        return plan;
    }
    if (onGrid(delta)) {
        plan.acceptThis = true;
        return plan;
    }
    // Off-grid: hold it. If the next frame follows within the window this
    // one was an intermediate repaint; otherwise it is accepted late.
    plan.deferThis = true;
    plan.deferDeadline = arrival + static_cast<int64_t>(kDeferWindow * m_period);
    m_deferred = true;
    m_deferredTsRaw = tsRaw;
    m_deferredArrival = arrival;
    ++m_stats.deferred;
    return plan;
}

FrameTimeline::IngestPlan FrameTimeline::onDeferTimeout() {
    std::lock_guard<std::mutex> lock(m_mutex);
    IngestPlan plan;
    if (m_deferred) {
        plan.acceptDeferredFirst = true;
        m_deferred = false;
    }
    return plan;
}

bool FrameTimeline::hasDeferred() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_deferred;
}

bool FrameTimeline::lastFrame(TimelineFrame& out) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frames.empty()) return false;
    out = m_frames.back();
    return true;
}

int FrameTimeline::popLast() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_frames.empty()) return -1;
    const int slot = m_frames.back().slot;
    if (!m_clockGaps.empty() && m_clockGaps.back().index == m_frames.back().index) {
        m_clockGapSum -= m_clockGaps.back().gap;
        m_clockPeriodSum -= m_clockGaps.back().periods;
        m_clockGaps.pop_back();
    }
    m_frames.pop_back();
    if (!m_frames.empty()) {
        m_lastAcceptedTsRaw = m_frames.back().tsRaw;
        m_lastAcceptedTs = m_frames.back().ts;
    } else {
        m_lastAcceptedTsRaw = 0;
        m_lastAcceptedTs = 0;
    }
    return slot;
}

void FrameTimeline::push(TimelineFrame& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    frame.index = m_nextIndex++;
    frame.ts = regularize(frame.tsRaw);

    // Capture offset: arrival - source timestamp. Track the minimum so
    // scheduling jitter on the capture thread never delays the media clock,
    // with a slow rise so a genuine latency increase is followed.
    const int64_t sample = frame.arrival - frame.tsRaw;
    if (!m_captureOffsetValid) {
        m_captureOffset = sample;
        m_captureOffsetValid = true;
    } else {
        m_captureOffset = std::min(m_captureOffset + kMs / 10, sample);
    }

    frame.periods = 0;
    if (!m_frames.empty()) {
        TimelineFrame& last = m_frames.back();
        const int64_t gap = frame.tsRaw - last.tsRaw;
        frame.periods = pairPeriods(gap, m_frameLost);
        // A frame that reached the screen late (a game frame-time spike)
        // looks like a missing frame until the next one follows less than
        // half a period after it. Together they are two periods of one each,
        // not two and one: move the period back while the clock has shown at
        // most the real frame of that pair (offline, 30 fps 4x with 2%
        // hitches: half-speed pairs and 1.5 periods of drained lead).
        const bool fragment = static_cast<double>(gap) < 0.5 * clockPeriod();
        const bool untouched = m_outputFrameIndex + 1 < last.index ||
                               (m_outputFrameIndex + 1 == last.index && m_outputPosition <= 1);
        if (fragment && last.periods == 2 && m_frames.size() >= 2 && untouched) {
            last.periods = 1;
            if (!m_clockGaps.empty() && m_clockGaps.back().index == last.index) {
                m_clockGaps.back().periods = 1;
                --m_clockPeriodSum;
            }
        }
        if (frame.periods > 0) {
            m_clockGaps.push_back({frame.index, gap, frame.periods});
            m_clockGapSum += gap;
            m_clockPeriodSum += frame.periods;
            // Headroom for the queued pairs the clock has not reached yet.
            while (m_clockGaps.size() > kRateWindow + 32) {
                m_clockGapSum -= m_clockGaps.front().gap;
                m_clockPeriodSum -= m_clockGaps.front().periods;
                m_clockGaps.pop_front();
            }
        }
    }

    m_frameLost = false;

    m_lastAcceptedTsRaw = frame.tsRaw;
    m_lastAcceptedTs = frame.ts;
    m_frames.push_back(frame);
    if (clockRateChanged()) {
        resetClockPeriod();
        frame.periods = m_frames.back().periods;
    }
}

void FrameTimeline::noteLostFrame() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_frameLost = true;
}

std::vector<int> FrameTimeline::retireBefore(uint64_t frameIndex) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<int> slots;
    while (m_frames.size() > 1 && m_frames.front().index < frameIndex) {
        slots.push_back(m_frames.front().slot);
        m_frames.pop_front();
    }
    return slots;
}

void FrameTimeline::reportStarvation() {
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_stats.starvations;
}

FrameTimeline::Stats FrameTimeline::stats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    Stats s = m_stats;
    s.sourcePeriodMs = static_cast<double>(m_period) / kMs;
    s.outputPeriodMs = clockPeriod() / kMs;
    s.cadenceLocked = m_locked;
    s.latencyBudgetMs = static_cast<double>(m_latencyBudget) / kMs;
    s.captureOffsetMs = static_cast<double>(m_captureOffset) / kMs;
    s.frames = m_frames.size();
    return s;
}

int64_t FrameTimeline::sourcePeriod() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_period;
}

bool FrameTimeline::cadenceLocked() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_locked;
}
