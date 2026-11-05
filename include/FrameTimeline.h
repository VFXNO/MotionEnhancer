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
};

// Ordered source frames plus the clocks that map them onto display refreshes.
//
// Producer side (capture thread): classifies each new frame against the
// source cadence, regularizes timestamps, appends frames.
// Consumer side (render thread): for a predicted display time returns the
// bracketing pair and blend factor, and adapts the latency budget from the
// observed starvation/slack.
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
        int64_t mediaTime = 0;
    };

    struct Stats {
        double sourcePeriodMs = 0.0;
        bool cadenceLocked = false;
        double latencyBudgetMs = 0.0;
        double captureOffsetMs = 0.0;
        uint64_t starvations = 0;
        uint64_t deferred = 0;
        uint64_t dropped = 0;
        uint64_t replaced = 0;
        size_t frames = 0;
    };

    void configure(uint32_t sourceFpsOverride, double refreshHz, uint32_t outputMultiplier);
    // The latency budget's limits (call before configure): minExtraMs is the
    // margin above the least possible budget (one source period + one
    // refresh; default 4 ms), maxMs caps how far starvation may grow it
    // (<= 0: three periods + 60 ms, the default).
    void setLatencyLimits(double minExtraMs, double maxMs);
    // Vsync off: the timeline decides when frames are shown. With an output
    // multiplier N, nextPresentTime() returns the wall-clock time of the next
    // position k/N of a pair (each position once, in order) and select() snaps
    // to it exactly; without one, one frame per display refresh on its own
    // clock. Off (the default): select() is driven by the display's vsync.
    // position (optional) is set false when the returned time is only when
    // to ask again (the frames for the next position have not arrived): no
    // frame is to be shown then.
    void setFreePacing(bool on);
    int64_t nextPresentTime(int64_t now, bool* position = nullptr);

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

    // ---- consumer -------------------------------------------------------
    Selection select(int64_t displayWallTime);
    // Frames older than the given index are no longer needed for display.
    // Returns their slots so the capture ring can recycle them.
    std::vector<int> retireBefore(uint64_t frameIndex);
    // A pair was needed but its flow was not finished: the budget grows.
    void reportStarvation();
    // How early a pair became usable relative to its first use.
    void reportSlack(int64_t slack);

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

    mutable std::mutex m_mutex;
    std::deque<TimelineFrame> m_frames;
    uint64_t m_nextIndex = 1;

    // cadence
    uint32_t m_sourceFpsOverride = 0;
    std::deque<int64_t> m_intervals;
    int64_t m_period = 333333;       // 30 fps until measured
    bool m_periodMeasured = false;   // m_period is a measurement (or the override)
    bool m_locked = false;
    int64_t m_lastArrivalTsRaw = 0;  // newest non-duplicate arrival (any decision)
    int64_t m_lastAcceptedTsRaw = 0;
    int64_t m_lastAcceptedTs = 0;

    // deferral
    bool m_deferred = false;
    int64_t m_deferredTsRaw = 0;
    int64_t m_deferredArrival = 0;

    // clocks
    int64_t m_refreshInterval = 166667;
    uint32_t m_outputMultiplier = 0;
    int64_t m_captureOffset = 0;     // arrival - tsRaw (running minimum, slow rise)
    bool m_captureOffsetValid = false;
    int64_t m_latencyBudget = 0;
    int64_t m_minLatencyBudget = 0;
    int64_t m_maxLatencyBudget = 0;
    int64_t m_minLatencyExtra = 40000;   // setLatencyLimits: 4 ms
    int64_t m_maxLatencyOverride = 0;    // setLatencyLimits: 0 = three periods + 60 ms
    bool m_freePacing = false;           // setFreePacing
    int64_t m_lastScheduledMedia = INT64_MIN;   // nextPresentTime: the last position given out
    int64_t m_lastScheduled = 0;                // nextPresentTime without a multiplier
    int64_t m_minSlack = INT64_MAX;
    int64_t m_lastBudgetReview = 0;
    int64_t m_holdSince = 0;         // media time when the current hold began, 0 if not holding
    bool m_holdPending = false;      // a hold happened and no frame has arrived since
    uint64_t m_currentF1Index = 0;

    Stats m_stats;
};
