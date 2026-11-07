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

    const int64_t lead = m_captureOffset + m_latencyBudget;
    const double period = m_sourceFpsOverride > 0
        ? static_cast<double>(kSecond) / m_sourceFpsOverride : static_cast<double>(m_period);
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
        size_t i = outputFrame;
        const uint64_t position = static_cast<uint64_t>(m_outputPosition) + missed;
        const uint64_t advance = position / m_outputMultiplier;
        if (advance >= m_frames.size() - i) {
            // A paused source or blocked Present exhausted the input. Wait
            // for fresh frames rather than replaying this queue in a burst.
            m_outputFrameIndex = m_frames.back().index + 1;
            m_outputPosition = 0;
            m_nextOutputTime = 0.0;
            sel.retireBeforeIndex = m_frames.back().index;
            return sel;
        }
        i += static_cast<size_t>(advance);
        m_outputPosition = static_cast<uint32_t>(position % m_outputMultiplier);
        m_outputFrameIndex = m_frames[i].index;
        if (i + 1 < m_frames.size() && m_frames[i + 1].ts - m_frames[i].ts > 2 * m_period) {
            // Never stretch one frame's motion across a source pause.
            ++i;
            m_outputFrameIndex = m_frames[i].index;
            m_outputPosition = 0;
            m_nextOutputTime = static_cast<double>(std::max(now, m_frames[i].ts + lead));
        }
        if (m_outputPosition > 0 && i + 1 == m_frames.size()) {
            sel.retireBeforeIndex = m_frames[i].index;
            return sel;
        }
        sel.f0 = m_frames[i];
        sel.f1 = i + 1 < m_frames.size() ? m_frames[i + 1] : sel.f0;
        sel.alpha = static_cast<float>(m_outputPosition) / m_outputMultiplier;
        sel.mediaTime = sel.f0.ts + static_cast<int64_t>(
            static_cast<double>(sel.f1.ts - sel.f0.ts) * m_outputPosition / m_outputMultiplier);
        sel.presentTime = static_cast<int64_t>(m_nextOutputTime + 0.5);
        if (++m_outputPosition == m_outputMultiplier) {
            m_outputPosition = 0;
            m_outputFrameIndex = sel.f1.index + (sel.f0.index == sel.f1.index ? 1 : 0);
        }
    }
    sel.valid = true;
    sel.retireBeforeIndex = sel.f0.index;
    m_currentF1Index = sel.f1.index; // keep a reserved pair safe from producer replacement
    m_nextOutputTime += step;
    return sel;
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
    const int64_t oldPeriod = m_period;
    // Locked: steady intervals that are the rate itself (not a refresh-grid
    // quantization of it). Unlocked sources are paced by a smoothed clock at
    // the mean rate (regularize).
    m_locked = near * 4 >= sorted.size() * 3 &&
               std::llabs(typical - rate) <= static_cast<int64_t>(kQuantizedSpread * static_cast<double>(rate));
    m_period = std::max<int64_t>(rate, kMs);
    m_periodMeasured = true;
    {
        const bool changed = std::llabs(m_period - oldPeriod) > oldPeriod / 5;
        if ((m_locked && !wasLocked) || changed) {
            applyBudgetLimits();
        }
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

    m_lastAcceptedTsRaw = frame.tsRaw;
    m_lastAcceptedTs = frame.ts;
    m_frames.push_back(frame);
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
