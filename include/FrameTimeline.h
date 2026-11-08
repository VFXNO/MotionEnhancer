#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

// All times are in 100 ns units. Source timestamps come from WGC
// (SystemRelativeTime); wall-clock times are QueryPerformanceCounter
// converted to 100 ns. The two domains are related through the measured
// capture offset (see mediaTime).
struct TimelineFrame {
    int      slot = -1;        // capture slot holding the pixels
    uint64_t index = 0;        // accepted-frame sequence number (1-based)
    int64_t  tsRaw = 0;        // WGC timestamp
    int64_t  ts = 0;           // cadence-regularized timestamp
    int64_t  arrival = 0;      // wall clock when the frame left WGC
    int      flowEntry = -1;   // flow ring entry for the pair (previous, this)
    uint64_t flowFence = 0;    // flow-queue fence value that completes it
    bool     sceneCut = false; // pair (previous, this) is a hard cut
    // Source periods the pair (previous, this) spans, from the raw WGC gap,
    // filled by push(): 1 normally, 2 when one source frame never reached the
    // timeline (WGC delivers only the newest frame per composition, a game
    // hitch, a full slot ring), 0 for the first frame and after a pause.
    uint32_t periods = 0;
};

// Ordered source frames and a simple source-rate output clock.
//
// Producer side (capture thread): classifies each new frame against the
// source cadence, regularizes timestamps, appends frames.
// Consumer side: emits positions k/N at source FPS x N, without compositor
// ticks, refresh holds, phase locking, resampling, or adaptive media clocks.
class FrameTimeline {
public:
    // What the producer must do with a newly arrived (non-duplicate) frame.
    struct IngestPlan {
        bool acceptDeferredFirst = false; // the held frame was real: accept it before this one
        bool dropDeferred = false;        // release the held frame's slot
        bool replaceLast = false;         // pop the newest accepted frame (its slot is released)
        bool acceptThis = false;          // ingest and push this frame
        bool deferThis = false;           // hold this frame until deferDeadline or the next frame
        int64_t deferDeadline = 0;        // wall clock
    };

    struct Selection {
        bool valid = false;
        TimelineFrame f0;
        TimelineFrame f1;        // == f0 when a single frame is shown
        float alpha = 0.0f;
        bool hold = false;       // media time is beyond the newest frame
        bool startup = false;    // media time is before the first frame
        bool fixedCadence = true;
        int64_t presentTime = 0;  // output deadline; retry time when !valid
        int64_t mediaTime = 0;
        uint64_t retireBeforeIndex = 0; // release stale input even when no output is valid

        // FSR makes only t=0.5. Never replace a real frame or a different
        // fixed k/N position with that cached image, even with a wide window.
        bool canUseMidpoint(float window) const {
            const float distance = alpha >= 0.5f ? alpha - 0.5f : 0.5f - alpha;
            return alpha > 0.0f && alpha < 1.0f && distance <= window &&
                   (!fixedCadence || alpha == 0.5f);
        }
    };

    struct Stats {
        double sourcePeriodMs = 0.0;
        double outputPeriodMs = 0.0;  // source period the fixed-N output clock runs on
        bool cadenceLocked = false;
        double latencyBudgetMs = 0.0;
        double captureOffsetMs = 0.0;
        uint64_t starvations = 0;
        uint64_t deferred = 0;
        uint64_t dropped = 0;
        uint64_t replaced = 0;
        uint64_t missedOutputs = 0; // late CPU/GPU work, never catch-up bursts
        size_t frames = 0;
    };

    void configure(uint32_t sourceFpsOverride, double refreshHz, uint32_t outputMultiplier);
    // Fixed buffering limits (call before configure). The default budget is
    // two source periods + 10 ms, independent of display Hz. minExtraMs is
    // the minimum margin above one period; maxMs caps it (<= 0: automatic).
    void setLatencyLimits(double minExtraMs, double maxMs);
    // Reserves the next output, including its exact frames and alpha. Keep
    // this selection until its presentTime; do not re-select after waiting.
    // Fixed N always emits k/N at source FPS x N, on every monitor (a pair
    // that spans two source periods because one frame never arrived gets
    // its 2N positions k/2N on the same clock). Only Max (N=0) chooses its
    // output FPS from refreshHz. Missing input returns an invalid selection
    // and a retry time, not another copy of the last frame.
    Selection nextOutput(int64_t now);

    // ---- producer -------------------------------------------------------
    IngestPlan onFrameArrived(int64_t tsRaw, int64_t arrival);
    IngestPlan onDeferTimeout();
    bool hasDeferred() const;
    // The frame that a new frame would pair with (copy; valid() false when
    // the timeline is empty).
    bool lastFrame(TimelineFrame& out) const;
    // Pops the newest frame (Replace). Returns its slot or -1.
    int popLast();
    // Appends an accepted frame; fills index and the regularized ts.
    void push(TimelineFrame& frame);
    // A captured source frame was lost before classification (no free slot,
    // callback queue overflow): the next accepted pair spans it for certain.
    void noteLostFrame();

    // ---- consumer -------------------------------------------------------
    // Frames older than the given index are no longer needed for display.
    // Returns their slots so the capture ring can recycle them.
    std::vector<int> retireBefore(uint64_t frameIndex);
    // Diagnostic only: no automatic latency or output-rate adjustment.
    void reportStarvation();

    Stats stats() const;
    int64_t sourcePeriod() const;
    bool cadenceLocked() const;

private:
    // Cadence estimate from every non-duplicate arrival, not only accepted
    // ones, so a rate change is detected even while deferral is rejecting
    // most frames.
    void observeInterval(int64_t interval);
    void updateCadence();
    int64_t regularize(int64_t tsRaw);
    void applyBudgetLimits();
    bool onGrid(int64_t delta) const;
    // How far from the cadence grid a real frame may land: a frame that
    // reached the screen one refresh late is still a frame, not a repaint.
    int64_t phaseTolerance() const;
    // Fixed-N output clock: the source period measured over the accepted
    // frames' pairs, in the same units the clock consumes (source periods).
    double clockPeriod() const;
    uint32_t pairPeriods(int64_t rawGap, bool lostFrame = false) const;
    void resetClockPeriod();
    bool clockRateChanged() const;
    void updateRefreshGrid(double period, bool locked);

    mutable std::mutex m_mutex;
    std::deque<TimelineFrame> m_frames;
    uint64_t m_nextIndex = 1;

    // cadence
    uint32_t m_sourceFpsOverride = 0;
    std::deque<int64_t> m_intervals;
    int64_t m_period = 333333;       // 30 fps until measured
    bool m_periodMeasured = false;   // m_period is a measurement (or the override)
    bool m_locked = false;
    int m_outliers = 0;              // consecutive timestamps off the regularized clock (same side)
    int m_outlierSign = 0;
    int64_t m_lastArrivalTsRaw = 0;  // newest non-duplicate arrival (any decision)
    int64_t m_lastAcceptedTsRaw = 0;
    int64_t m_lastAcceptedTs = 0;

    // deferral
    bool m_deferred = false;
    int64_t m_deferredTsRaw = 0;
    int64_t m_deferredArrival = 0;

    // clocks
    int64_t m_captureTimestampQuantum = 166667; // WGC timestamp noise, not the output clock
    double m_maxOutputPeriod = 166666.6666667;  // only Max uses display Hz
    uint32_t m_outputMultiplier = 0;
    int64_t m_captureOffset = 0;     // arrival - tsRaw (running minimum, slow rise)
    bool m_captureOffsetValid = false;
    int64_t m_latencyBudget = 0;
    int64_t m_minLatencyBudget = 0;
    int64_t m_maxLatencyBudget = 0;
    int64_t m_minLatencyExtra = 40000;   // setLatencyLimits: 4 ms
    int64_t m_maxLatencyOverride = 0;    // setLatencyLimits: 0 = three periods + 60 ms
    double m_nextOutputTime = 0.0;  // steady deadline, not re-anchored to each wake-up
    uint64_t m_outputFrameIndex = 0;
    uint32_t m_outputPosition = 0;   // next k/(N x periods) within that frame's pair
    uint64_t m_currentF1Index = 0;

    // Output clock period (see clockPeriod): accepted pairs' raw WGC gaps
    // and their source-period counts, newest kRateWindow (+32 queued) pairs.
    struct ClockGap { uint64_t index; int64_t gap; uint32_t periods; };
    std::deque<ClockGap> m_clockGaps;
    int64_t m_clockGapSum = 0;
    uint64_t m_clockPeriodSum = 0;
    bool m_clockStarted = false;     // the fixed-N clock has made its first reservation
    bool m_frameLost = false;        // noteLostFrame() since the last push
    bool m_refreshGridSource = false; // steady rate at a refresh multiple (updateRefreshGrid)
    size_t m_offGridRun = 0;         // consecutive locked cadence updates off the refresh grid
    bool m_refreshGridChecked = false; // judged once on a full rate window

    Stats m_stats;
};
