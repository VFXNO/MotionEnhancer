#pragma once

#include "Image.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Decode a bounded selection of video frames directly into training images.
// Uses OS-installed Media Foundation codecs; creates no PNG intermediates.
bool decodeTrainingVideo(const std::string& path, size_t frameLimit,
                          std::vector<Image>& frames, std::string& error);

// Sequentially decode every video sample at the training resolution. The
// callback consumes one frame at a time, so clip duration does not determine
// training RAM usage. Returning false stops with an error from the callback.
using TrainingFrameConsumer = std::function<bool(Image&&, uint64_t, uint64_t)>;
bool streamTrainingVideo(const std::string& path, const TrainingFrameConsumer& consume,
                         uint64_t& decodedFrames, std::string& error);
