// FlowMedianCS over 5x5 instead of 3x3: the AI engine's field after DIS
// refinement (FlowEngine::recordFlowMedian). Two DIS steps per patch from the
// network's start leave a mottled field - each 8 px patch nudged its own way -
// that pure warping shows as speckle on moving surfaces (a cloak, water).
// Measured on 20 pairs of five 1440p captures (true middle frame, flow
// roughness as mean |gradient|): none 26.66 dB / 0.756, 3x3 26.65 / 0.539,
// 5x5 26.64 / 0.483; refining one level coarser instead 26.39 / 0.533.
#define MEDIAN_RADIUS 2
#include "FlowMedianCS.hlsl"
