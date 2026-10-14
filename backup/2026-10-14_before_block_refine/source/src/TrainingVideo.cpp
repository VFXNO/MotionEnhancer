#include "TrainingVideo.h"

#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <propvarutil.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>

using Microsoft::WRL::ComPtr;

namespace {
struct MediaFoundationSession {
    HRESULT comResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool started = false;
    MediaFoundationSession() {
        if (SUCCEEDED(comResult) || comResult == RPC_E_CHANGED_MODE)
            started = SUCCEEDED(MFStartup(MF_VERSION));
    }
    ~MediaFoundationSession() {
        if (started) MFShutdown();
        if (SUCCEEDED(comResult)) CoUninitialize();
    }
};

std::wstring toWide(const std::string& text) {
    const int length = MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, nullptr, 0);
    if (!length) return {};
    std::wstring result(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, result.data(), length);
    result.pop_back(); // remove the terminator written by MultiByteToWideChar
    return result;
}

bool copyFrame(IMFSample* sample, UINT width, UINT height, IMFMediaType* type, Image& result) {
    ComPtr<IMFMediaBuffer> buffer;
    if (FAILED(sample->ConvertToContiguousBuffer(&buffer))) return false;
    BYTE* scanline = nullptr;
    LONG pitch = 0;
    ComPtr<IMF2DBuffer> buffer2D;
    const bool twoD = SUCCEEDED(buffer.As(&buffer2D)) &&
                      SUCCEEDED(buffer2D->Lock2D(&scanline, &pitch));
    DWORD length = 0;
    if (!twoD) {
        if (FAILED(buffer->Lock(&scanline, nullptr, &length))) return false;
        UINT32 defaultStride = 0;
        pitch = SUCCEEDED(type->GetUINT32(MF_MT_DEFAULT_STRIDE, &defaultStride))
            ? static_cast<LONG>(defaultStride) : static_cast<LONG>(width * 4);
        if (static_cast<uint64_t>(std::abs(pitch)) * height > length) {
            buffer->Unlock();
            return false;
        }
        // RGB32 DIBs returned in a plain memory buffer are bottom-up when
        // their stride is positive. IMF2DBuffer::Lock2D already points to
        // the first logical row, regardless of pitch sign.
        if (pitch > 0) scanline += static_cast<size_t>(height - 1) * pitch;
    }
    const UINT largest = std::max(width, height);
    const UINT extent = std::min(largest, 960u);
    const UINT outputW = std::max(1u, static_cast<UINT>(static_cast<uint64_t>(width) * extent / largest));
    const UINT outputH = std::max(1u, static_cast<UINT>(static_cast<uint64_t>(height) * extent / largest));
    result = Image(static_cast<int>(outputW), static_cast<int>(outputH), 3);
    for (UINT y = 0; y < outputH; ++y) {
        const UINT sourceY = static_cast<UINT>(static_cast<uint64_t>(y) * height / outputH);
        const BYTE* row = scanline + (twoD ? static_cast<ptrdiff_t>(sourceY) * pitch
                                             : pitch > 0 ? -static_cast<ptrdiff_t>(sourceY) * pitch
                                                         : static_cast<ptrdiff_t>(sourceY) * (-pitch));
        for (UINT x = 0; x < outputW; ++x) {
            const UINT sourceX = static_cast<UINT>(static_cast<uint64_t>(x) * width / outputW);
            const BYTE* bgra = row + static_cast<size_t>(sourceX) * 4;
            result.set(static_cast<int>(x), static_cast<int>(y), 0, bgra[2]);
            result.set(static_cast<int>(x), static_cast<int>(y), 1, bgra[1]);
            result.set(static_cast<int>(x), static_cast<int>(y), 2, bgra[0]);
        }
    }
    if (twoD) buffer2D->Unlock2D();
    else buffer->Unlock();
    return true;
}
} // namespace

bool decodeTrainingVideo(const std::string& path, size_t frameLimit,
                         std::vector<Image>& frames, std::string& error) {
    if (frameLimit == 0 || frameLimit > 16) { error = "invalid frame limit"; return false; }
    MediaFoundationSession media;
    if (!media.started) { error = "Media Foundation could not start"; return false; }
    const std::wstring widePath = toWide(path);
    if (widePath.empty()) { error = "invalid video filename"; return false; }

    ComPtr<IMFAttributes> attributes;
    ComPtr<IMFSourceReader> reader;
    ComPtr<IMFMediaType> rgbType, currentType;
    if (FAILED(MFCreateAttributes(&attributes, 1)) ||
        FAILED(attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE)) ||
        FAILED(MFCreateSourceReaderFromURL(widePath.c_str(), attributes.Get(), &reader))) {
        error = "cannot open video (or Windows has no decoder for it)";
        return false;
    }
    if (FAILED(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE)) ||
        FAILED(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE)) ||
        FAILED(MFCreateMediaType(&rgbType)) ||
        FAILED(rgbType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) ||
        FAILED(rgbType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32)) ||
        FAILED(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, rgbType.Get())) ||
        FAILED(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &currentType))) {
        error = "Windows could not convert the video to RGB32";
        return false;
    }
    UINT width = 0, height = 0;
    if (FAILED(MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE, &width, &height)) ||
        width == 0 || height == 0 || width > 16384 || height > 16384) {
        error = "invalid video dimensions";
        return false;
    }

    PROPVARIANT duration;
    PropVariantInit(&duration);
    const bool canSeek = SUCCEEDED(reader->GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE,
                                                                    MF_PD_DURATION, &duration)) &&
                         duration.vt == VT_UI8 && duration.uhVal.QuadPart > 0;
    const uint64_t duration100ns = canSeek ? duration.uhVal.QuadPart : 0;
    PropVariantClear(&duration);
    LONGLONG previousTimestamp = -1;
    const size_t groups = (frameLimit + 2) / 3;
    for (size_t i = 0; i < groups; ++i) {
        const LONGLONG requested = canSeek
            ? static_cast<LONGLONG>(duration100ns * (i + 1) / (groups + 1)) : 0;
        if (canSeek) {
            PROPVARIANT position;
            PropVariantInit(&position);
            position.vt = VT_I8;
            position.hVal.QuadPart = requested;
            const HRESULT hr = reader->SetCurrentPosition(GUID_NULL, position);
            PropVariantClear(&position);
            if (FAILED(hr)) { error = "could not seek video"; return false; }
        }
        // Media Foundation seeks to a nearby keyframe, which can be well
        // before the requested time. Decode forward to the desired frame.
        // Without seeking, take roughly one frame every 30 decoded frames.
        const int reads = canSeek ? 120 : 30;
        ComPtr<IMFSample> selected;
        LONGLONG selectedTimestamp = 0;
        bool atEnd = false;
        for (int n = 0; n < reads; ++n) {
            DWORD flags = 0;
            LONGLONG timestamp = 0;
            ComPtr<IMFSample> sample;
            if (FAILED(reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr,
                                          &flags, &timestamp, &sample))) {
                error = "video decoding failed";
                return false;
            }
            if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { atEnd = true; break; }
            if ((flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) != 0) {
                // Common on real files: the decoder changes resolution or
                // stride mid-stream. Re-read the type and keep decoding.
                currentType = nullptr;
                if (FAILED(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                       &currentType)) ||
                    FAILED(MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE, &width, &height)) ||
                    width == 0 || height == 0 || width > 16384 || height > 16384) {
                    error = "video reported an invalid frame size";
                    return false;
                }
            }
            if (!sample) continue;
            selected = std::move(sample);
            selectedTimestamp = timestamp;
            if (canSeek && timestamp >= requested) break;
        }
        for (int triplet = 0; selected && triplet < 3 && frames.size() < frameLimit; ++triplet) {
            if (triplet > 0) {
                selected = nullptr;
                for (int retry = 0; retry < 16; ++retry) {
                    DWORD flags = 0;
                    LONGLONG timestamp = 0;
                    ComPtr<IMFSample> sample;
                    if (FAILED(reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr,
                                                  &flags, &timestamp, &sample))) {
                        error = "could not decode consecutive video frames";
                        return false;
                    }
                    if (flags & MF_SOURCE_READERF_ENDOFSTREAM) break;
                    if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
                        currentType = nullptr;
                        if (FAILED(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                               &currentType)) ||
                            FAILED(MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE,
                                                      &width, &height))) {
                            error = "video changed to an invalid frame size";
                            return false;
                        }
                    }
                    if (sample && timestamp > selectedTimestamp) {
                        selected = std::move(sample);
                        selectedTimestamp = timestamp;
                        break;
                    }
                }
            }
            if (selected && selectedTimestamp != previousTimestamp) {
                Image frame;
                if (!copyFrame(selected.Get(), width, height, currentType.Get(), frame)) {
                    error = "could not read a decoded video frame";
                    return false;
                }
                frames.push_back(std::move(frame));
                previousTimestamp = selectedTimestamp;
            }
        }
        if (atEnd) break;
    }
    if (frames.empty()) { error = "no video frames were decoded"; return false; }
    return true;
}

bool streamTrainingVideo(const std::string& path, const TrainingFrameConsumer& consume,
                         uint64_t& decodedFrames, std::string& error) {
    decodedFrames = 0;
    MediaFoundationSession media;
    if (!media.started) { error = "Media Foundation could not start"; return false; }
    const std::wstring widePath = toWide(path);
    if (widePath.empty()) { error = "invalid video filename"; return false; }
    ComPtr<IMFAttributes> attributes;
    ComPtr<IMFSourceReader> reader;
    ComPtr<IMFMediaType> rgbType, currentType;
    if (FAILED(MFCreateAttributes(&attributes, 1)) ||
        FAILED(attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE)) ||
        FAILED(MFCreateSourceReaderFromURL(widePath.c_str(), attributes.Get(), &reader))) {
        error = "cannot open video";
        return false;
    }
    if (FAILED(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE)) ||
        FAILED(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE)) ||
        FAILED(MFCreateMediaType(&rgbType)) ||
        FAILED(rgbType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) ||
        FAILED(rgbType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32)) ||
        FAILED(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, rgbType.Get())) ||
        FAILED(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &currentType))) {
        error = "Windows could not convert video to RGB32";
        return false;
    }
    UINT width = 0, height = 0;
    if (FAILED(MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE, &width, &height)) ||
        width == 0 || height == 0 || width > 16384 || height > 16384) {
        error = "invalid video dimensions";
        return false;
    }
    PROPVARIANT duration;
    PropVariantInit(&duration);
    const bool knownDuration = SUCCEEDED(reader->GetPresentationAttribute(
        MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &duration)) &&
        duration.vt == VT_UI8 && duration.uhVal.QuadPart > 0;
    const uint64_t duration100ns = knownDuration ? duration.uhVal.QuadPart : 0;
    PropVariantClear(&duration);
    while (true) {
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        if (FAILED(reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr,
                                      &flags, &timestamp, &sample))) {
            error = "video decoding failed";
            return false;
        }
        if (flags & MF_SOURCE_READERF_ERROR) {
            error = "video decoder reported a stream error";
            return false;
        }
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) break;
        if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
            currentType = nullptr;
            if (FAILED(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                   &currentType)) ||
                FAILED(MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE, &width, &height)) ||
                width == 0 || height == 0 || width > 16384 || height > 16384) {
                error = "video changed to an invalid frame size";
                return false;
            }
        }
        if (!sample) continue;
        Image frame;
        if (!copyFrame(sample.Get(), width, height, currentType.Get(), frame)) {
            error = "could not read a decoded video frame";
            return false;
        }
        if (!consume(std::move(frame), static_cast<uint64_t>(std::max<LONGLONG>(0, timestamp)),
                     duration100ns)) {
            error = "training failed while processing a video frame";
            return false;
        }
        ++decodedFrames;
    }
    if (!decodedFrames) { error = "no video frames were decoded"; return false; }
    return true;
}
