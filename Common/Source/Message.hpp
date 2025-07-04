/*
 * Copyright (c) 2020 Andreas Pohl
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * Author: Andreas Pohl
 */

#ifndef Message_hpp
#define Message_hpp

#include "json.hpp"

using json = nlohmann::json;

#if defined(AG_PLUGIN) || defined(AG_SERVER)

#include "KeyAndMouseCommon.hpp"
#include "Utils.hpp"
#include "Metrics.hpp"

namespace e47 {

/*
 * Core I/O functions
 */
struct MessageHelper {
    enum ErrorCode { E_NONE, E_DATA, E_TIMEOUT, E_STATE, E_SYSCALL, E_SIZE };

    static String errorCodeToString(ErrorCode ec) {
        switch (ec) {
            case E_NONE:
                return "E_NONE";
                break;
            case E_DATA:
                return "E_DATA";
                break;
            case E_TIMEOUT:
                return "E_TIMEOUT";
                break;
            case E_STATE:
                return "E_STATE";
                break;
            case E_SYSCALL:
                return "E_SYSCALL";
                break;
            case E_SIZE:
                return "E_SIZE";
                break;
        }
        return "";
    }

    struct Error {
        ErrorCode code = E_NONE;
        String str = "";
        String toString() const {
            String ret;
            if (str.isNotEmpty()) {
                ret << str << " (" << errorCodeToString(code) << ")";
            } else {
                ret << errorCodeToString(code);
            }
            return ret;
        }
    };

    static void seterr(Error* e, ErrorCode c, String s = "") {
        if (nullptr != e) {
            e->code = c;
            e->str = s;
        }
    }

    static void seterrstr(Error* e, String s) {
        if (nullptr != e) {
            e->str = s;
        }
    }
};

bool send(StreamingSocket* socket, const char* data, int size, MessageHelper::Error* e = nullptr,
          Meter* metric = nullptr);
bool read(StreamingSocket* socket, void* data, int size, int timeoutMilliseconds = 0, MessageHelper::Error* e = nullptr,
          Meter* metric = nullptr);

bool setNonBlocking(int handle) noexcept;
StreamingSocket* accept(StreamingSocket*, int timeoutMs = 1000, std::function<bool()> abortFn = nullptr);

/*
 * Client/Server handshake
 */
static constexpr int AG_PROTOCOL_VERSION = 13;

struct HandshakeRequest {
    int version;
    int channelsIn;
    int channelsOut;
    int channelsSC;
    double sampleRate;
    int samplesPerBlock;
    bool doublePrecision;
    uint64 clientId;
    uint8 flags;
    uint8 unused1;
    uint64 activeChannels;
    uint16 unused2;

    enum FLAGS : uint8 { NO_PLUGINLIST_FILTER = 1 };
    void setFlag(uint8 f) { flags |= f; }
    bool isFlag(uint8 f) { return (flags & f) == f; }

    json toJson() const {
        json j;
        j["version"] = version;
        j["channelsIn"] = channelsIn;
        j["channelsOut"] = channelsOut;
        j["channelsSC"] = channelsSC;
        j["rate"] = sampleRate;
        j["samplesPerBlock"] = samplesPerBlock;
        j["doublePrecision"] = doublePrecision;
        j["clientId"] = clientId;
        j["flags"] = flags;
        j["activeChannels"] = activeChannels;
        return j;
    }

    void fromJson(const json& j) {
        version = j["version"].get<int>();
        channelsIn = j["channelsIn"].get<int>();
        channelsOut = j["channelsOut"].get<int>();
        channelsSC = j["channelsSC"].get<int>();
        sampleRate = j["rate"].get<double>();
        samplesPerBlock = j["samplesPerBlock"].get<int>();
        doublePrecision = j["doublePrecision"].get<bool>();
        clientId = j["clientId"].get<uint64>();
        flags = j["flags"].get<uint8>();
        activeChannels = j["activeChannels"].get<uint64>();
    }
};

struct HandshakeResponse {
    int version;
    uint32 flags;
    int port;
    uint32 unused1;
    uint32 unused2;
    uint32 unused3;
    uint32 unused4;
    uint32 unused5;
    uint32 unused6;

    enum FLAGS : uint32 { SANDBOX_ENABLED = 1, LOCAL_MODE = 2 };
    void setFlag(uint32 f) { flags |= f; }
    bool isFlag(uint32 f) { return (flags & f) == f; }
};

/*
 * Audio streaming
 */
class AudioMessage : public LogTagDelegate {
  public:
    AudioMessage(const LogTag* tag) : LogTagDelegate(tag) {}

    struct RequestHeader {
        int channels;
        int samples;
        int channelsRequested;  // If only midi data is sent, let the server know about the expected audio buffer size
        int samplesRequested;   // If only midi data is sent, let the server know about the expected audio buffer size
        int numMidiEvents;
        bool isDouble;
        Uuid traceId;
    };

    struct ResponseHeader {
        int channels;
        int samples;
        int numMidiEvents;
        int latencySamples;
    };

    struct MidiHeader {
        int sampleNumber;
        int size;
    };

    int getChannels() const { return m_reqHeader.channels; }
    int getChannelsRequested() const { return m_reqHeader.channelsRequested; }
    int getSamples() const { return m_reqHeader.samples; }
    int getSamplesRequested() const { return m_reqHeader.samplesRequested; }
    bool isDouble() const { return m_reqHeader.isDouble; }

    int getLatencySamples() const { return m_resHeader.latencySamples; }

    template <typename T>
    bool sendToServer(StreamingSocket* socket, AudioBuffer<T>& buffer, MidiBuffer& midi,
                      AudioPlayHead::PositionInfo& posInfo, int channelsRequested, int samplesRequested,
                      MessageHelper::Error* e, Meter& metric) {
        traceScope();
        m_reqHeader.channels = buffer.getNumChannels();
        m_reqHeader.samples = buffer.getNumSamples();
        m_reqHeader.channelsRequested = channelsRequested > -1 ? channelsRequested : buffer.getNumChannels();
        m_reqHeader.samplesRequested = samplesRequested > -1 ? samplesRequested : buffer.getNumSamples();
        m_reqHeader.isDouble = std::is_same<T, double>::value;
        m_reqHeader.numMidiEvents = midi.getNumEvents();
        m_reqHeader.traceId = TimeTrace::getTraceId();
        if (nullptr != socket && socket->isConnected()) {
            if (!send(socket, reinterpret_cast<const char*>(&m_reqHeader), sizeof(m_reqHeader), e, &metric)) {
                return false;
            }
            for (int chan = 0; chan < m_reqHeader.channels; ++chan) {
                if (!send(socket, reinterpret_cast<const char*>(buffer.getReadPointer(chan)),
                          m_reqHeader.samples * (int)sizeof(T), e, &metric)) {
                    return false;
                }
            }
            MidiHeader midiHdr;
            for (auto midiIt = midi.begin(); midiIt != midi.end(); midiIt++) {
                midiHdr.size = (*midiIt).numBytes;
                midiHdr.sampleNumber = (*midiIt).samplePosition;
                if (!send(socket, reinterpret_cast<const char*>(&midiHdr), sizeof(midiHdr), e, &metric)) {
                    return false;
                }
                if (!send(socket, reinterpret_cast<const char*>((*midiIt).data), midiHdr.size, e, &metric)) {
                    return false;
                }
            }
            if (!send(socket, reinterpret_cast<const char*>(&posInfo), sizeof(posInfo), e, &metric)) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool sendToClient(StreamingSocket* socket, AudioBuffer<T>& buffer, MidiBuffer& midi, int latencySamples,
                      int channelsToSend, MessageHelper::Error* e, Meter& metric) {
        traceScope();
        m_resHeader.channels = channelsToSend;
        m_resHeader.samples = buffer.getNumSamples();
        m_resHeader.latencySamples = latencySamples;
        m_resHeader.numMidiEvents = midi.getNumEvents();
        if (nullptr != socket && socket->isConnected()) {
            if (!send(socket, reinterpret_cast<const char*>(&m_resHeader), sizeof(m_resHeader), e, &metric)) {
                return false;
            }
            for (int chan = 0; chan < m_resHeader.channels; ++chan) {
                if (!send(socket, reinterpret_cast<const char*>(buffer.getReadPointer(chan)),
                          m_resHeader.samples * (int)sizeof(T), e, &metric)) {
                    return false;
                }
            }
            MidiHeader midiHdr;
            for (auto midiIt = midi.begin(); midiIt != midi.end(); midiIt++) {
                midiHdr.size = (*midiIt).numBytes;
                midiHdr.sampleNumber = (*midiIt).samplePosition;
                if (!send(socket, reinterpret_cast<const char*>(&midiHdr), sizeof(midiHdr), e, &metric)) {
                    return false;
                }
                if (!send(socket, reinterpret_cast<const char*>((*midiIt).data), midiHdr.size, e, &metric)) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    bool readFromServer(StreamingSocket* socket, AudioBuffer<T>& buffer, MidiBuffer& midi, MessageHelper::Error* e,
                        Meter& metric) {
        traceScope();
        if (nullptr != socket && socket->isConnected()) {
            if (!read(socket, &m_resHeader, sizeof(m_resHeader), 1000, e, &metric)) {
                MessageHelper::seterrstr(e, "response header");
                return false;
            }

            traceln("  buffer: channels=" << buffer.getNumChannels() << ", samples=" << buffer.getNumSamples());
            traceln("  header: channels=" << m_resHeader.channels << ", samples=" << m_resHeader.samples);

            bool needTmpBuffer = false;
            int channels = jmin(buffer.getNumChannels(), m_resHeader.channels);
            int samples = jmin(buffer.getNumSamples(), m_resHeader.samples);

            if (channels < m_resHeader.channels) {
                logln("warning: target buffer has "
                      << (m_resHeader.channels - channels)
                      << " channels less then what was received from the server, discarding audio "
                         "data");
                needTmpBuffer = true;
            }

            if (m_resHeader.channels < buffer.getNumChannels()) {
                logln("warning: target buffer has " << (buffer.getNumChannels() - m_resHeader.channels)
                                                    << " more channels then what was received from the server");
            }

            if (samples < m_resHeader.samples) {
                logln(
                    "warning: target buffer has less samples then what was received from the server, discarding audio "
                    "data");
                needTmpBuffer = true;
            }

            if (m_resHeader.samples < buffer.getNumSamples()) {
                logln(
                    "warning: target buffer has more samples then what was received from the server, audio artifacts "
                    "expected");
            }

            auto readAudio = [&](AudioBuffer<T>* targetBuffer) {
                for (int chan = 0; chan < m_resHeader.channels; ++chan) {
                    if (!read(socket, targetBuffer->getWritePointer(chan), m_resHeader.samples * (int)sizeof(T), 1000,
                              e, &metric)) {
                        MessageHelper::seterrstr(e, "audio data");
                        return false;
                    }
                }
                return true;
            };

            if (needTmpBuffer) {
                AudioBuffer<T> tmpBuf(m_resHeader.channels, m_resHeader.samples);
                if (!readAudio(&tmpBuf)) {
                    return false;
                }
                for (int chan = 0; chan < channels; chan++) {
                    buffer.copyFrom(chan, 0, tmpBuf, chan, 0, samples);
                }
            } else {
                if (!readAudio(&buffer)) {
                    return false;
                }
            }

            midi.clear();
            std::vector<char> midiData;
            MidiHeader midiHdr;
            for (int i = 0; i < m_resHeader.numMidiEvents; i++) {
                if (!read(socket, &midiHdr, sizeof(midiHdr), 1000, e, &metric)) {
                    MessageHelper::seterrstr(e, "midi header");
                    return false;
                }
                auto size = (size_t)midiHdr.size;
                if (midiData.size() < size) {
                    midiData.resize(size);
                }
                if (!read(socket, midiData.data(), midiHdr.size, 1000, e, &metric)) {
                    MessageHelper::seterrstr(e, "midi data");
                    return false;
                }
                midi.addEvent(midiData.data(), midiHdr.size, midiHdr.sampleNumber);
            }
        } else {
            MessageHelper::seterr(e, MessageHelper::E_STATE, "not connected");
            traceln("failed: E_STATE");
            return false;
        }
        MessageHelper::seterr(e, MessageHelper::E_NONE);
        return true;
    }

    bool readFromClient(StreamingSocket* socket, AudioBuffer<float>& bufferF, AudioBuffer<double>& bufferD,
                        MidiBuffer& midi, AudioPlayHead::PositionInfo& posInfo, MessageHelper::Error* e, Meter& metric,
                        Uuid& traceId) {
        traceScope();
        if (nullptr != socket && socket->isConnected()) {
            if (!read(socket, &m_reqHeader, sizeof(m_reqHeader), 0, e, &metric)) {
                MessageHelper::seterrstr(e, "request header");
                return false;
            }

            traceln("  buffer: channels=" << bufferF.getNumChannels() << ", samples=" << bufferF.getNumSamples());
            traceln("  header: channels=" << m_reqHeader.channels << ", samples=" << m_reqHeader.samples);

            traceId = m_reqHeader.traceId;

            int size = m_reqHeader.samples * (int)(m_reqHeader.isDouble ? sizeof(double) : sizeof(float));
            if (m_reqHeader.isDouble) {
                bufferD.setSize(jmax(m_reqHeader.channels, m_reqHeader.channelsRequested),
                                jmax(m_reqHeader.samples, m_reqHeader.samplesRequested), false, true);
            } else {
                bufferF.setSize(jmax(m_reqHeader.channels, m_reqHeader.channelsRequested),
                                jmax(m_reqHeader.samples, m_reqHeader.samplesRequested), false, true);
            }

            // Read the channel data from the client, if any
            for (int chan = 0; chan < m_reqHeader.channels; ++chan) {
                char* data = m_reqHeader.isDouble ? reinterpret_cast<char*>(bufferD.getWritePointer(chan))
                                                  : reinterpret_cast<char*>(bufferF.getWritePointer(chan));
                if (!read(socket, data, size, 0, e, &metric)) {
                    MessageHelper::seterrstr(e, "audio data");
                    return false;
                }
            }

            midi.clear();

            for (int i = 0; i < m_reqHeader.numMidiEvents; i++) {
                MidiHeader midiHdr;
                if (!read(socket, &midiHdr, sizeof(midiHdr), 0, e, &metric)) {
                    MessageHelper::seterrstr(e, "midi header");
                    return false;
                }
                if (midiHdr.size > 0) {
                    std::vector<char> midiData;
                    midiData.resize(static_cast<size_t>(midiHdr.size));
                    if (!read(socket, midiData.data(), midiHdr.size, 0, e, &metric)) {
                        MessageHelper::seterrstr(e, "midi data");
                        return false;
                    }
                    midi.addEvent(midiData.data(), midiHdr.size, midiHdr.sampleNumber);
                }
            }

            if (!read(socket, &posInfo, sizeof(posInfo), 0, e, &metric)) {
                MessageHelper::seterrstr(e, "pos info");
                return false;
            }
        } else {
            MessageHelper::seterr(e, MessageHelper::E_STATE, "not connected");
            traceln("failed: E_STATE");
            return false;
        }
        MessageHelper::seterr(e, MessageHelper::E_NONE);
        return true;
    }

  private:
    RequestHeader m_reqHeader;
    ResponseHeader m_resHeader;
};

/*
 * Command I/O
 */
class Payload : public LogTagDelegate {
  public:
    using Buffer = std::vector<char>;

    Payload() : payloadType(-1) {}
    Payload(int t, size_t s = 0) : payloadType(t), payloadBuffer(s) { memset(getData(), 0, s); }
    virtual ~Payload() {}
    Payload& operator=(const Payload& other) = delete;
    Payload& operator=(Payload&& other) {
        if (this != &other) {
            payloadType = other.payloadType;
            other.payloadType = -1;
            payloadBuffer = std::move(other.payloadBuffer);
        }
        return *this;
    }

    int getType() const { return payloadType; }
    void setType(int t) { payloadType = t; }
    int getSize() const { return (int)payloadBuffer.size(); }
    void setSize(int size) {
        payloadBuffer.resize((size_t)size);
        realign();
    }
    char* getData() { return reinterpret_cast<char*>(payloadBuffer.data()); }
    const char* getData() const { return payloadBuffer.data(); }

    virtual void realign() {}

    int payloadType;
    Buffer payloadBuffer;
};

template <typename T>
class DataPayload : public Payload {
  public:
    T* data;
    DataPayload(int type) : Payload(type, sizeof(T)) { realignInternal(); }
    virtual void realign() override { realignInternal(); }

  private:
    void realignInternal() { data = reinterpret_cast<T*>(payloadBuffer.data()); }
};

class NumberPayload : public DataPayload<int> {
  public:
    NumberPayload(int type) : DataPayload<int>(type) {}
    void setNumber(int n) { *data = n; }
    int getNumber() const { return *data; }
};

class FloatPayload : public DataPayload<float> {
  public:
    FloatPayload(int type) : DataPayload<float>(type) {}
    void setFloat(float n) { *data = n; }
    float getFloat() const { return *data; }
};

class StringPayload : public Payload {
  public:
    int* size;
    char* str;

    StringPayload(int type) : Payload(type, sizeof(int)) { realignInternal(); }

    void setString(const String& s) {
        setSize((int)sizeof(int) + s.length());
        *size = s.length();
        memcpy(str, s.getCharPointer(), (size_t)s.length());
    }

    String getString() const {
        if (nullptr != str && nullptr != size && *size > 0) {
            return String(str, static_cast<size_t>(*size));
        }
        return {};
    }

    virtual void realign() override { realignInternal(); }

  private:
    void realignInternal() {
        size = reinterpret_cast<int*>(payloadBuffer.data());
        str = (size_t)getSize() > sizeof(int) ? reinterpret_cast<char*>(payloadBuffer.data()) + sizeof(int) : nullptr;
    }
};

class BinaryPayload : public Payload {
  public:
    int* size;
    char* data;

    BinaryPayload(int type) : Payload(type, sizeof(int)) { realignInternal(); }

    void setData(const char* src, int len) {
        setSize((int)sizeof(int) + len);
        *size = len;
        memcpy(data, src, static_cast<size_t>(len));
    }

    virtual void realign() override { realignInternal(); }

  private:
    void realignInternal() {
        size = reinterpret_cast<int*>(payloadBuffer.data());
        data = (size_t)getSize() > sizeof(int) ? reinterpret_cast<char*>(payloadBuffer.data()) + sizeof(int) : nullptr;
    }
};

class JsonPayload : public BinaryPayload {
  public:
    JsonPayload(int type) : BinaryPayload(type) {}

    void setJson(const json& j) {
        auto str = j.dump();
        setData(str.data(), (int)str.size());
    }

    json getJson() {
        if (nullptr == data) {
            return {};
        }
        try {
            return json::parse(data, data + *size);
        } catch (json::parse_error& e) {
            logln("failed to parse json payload: " << e.what());
            return {};
        }
    }
};

class MsgPackPayload : public BinaryPayload {
  public:
    MsgPackPayload(int type) : BinaryPayload(type) {}

    void setJson(const json& j) {
        std::vector<uint8> v;
        json::to_msgpack(j, v);
        setData((const char*)v.data(), (int)v.size());
    }

    json getJson() {
        if (nullptr == data) {
            return {};
        }
        try {
            return json::from_msgpack(data, data + *size);
        } catch (json::parse_error& e) {
            logln("failed to parse msgPack payload: " << e.what());
            return {};
        }
    }
};

class Any : public Payload {
  public:
    static constexpr int Type = 0;
    Any() : Payload(Type) {}
};

class Quit : public Payload {
  public:
    static constexpr int Type = 1;
    Quit() : Payload(Type) {}
};

class Result : public Payload {
  public:
    static constexpr int Type = 2;

    struct hdr_t {
        int rc;
        int size;
    };
    hdr_t* hdr;
    char* str;

    Result() : Payload(Type) { realignInternal(); }

    void setResult(int rc, const String& s) {
        setSize(static_cast<int>(sizeof(hdr_t)) + s.length());
        hdr->rc = rc;
        hdr->size = s.length();
        memcpy(str, s.getCharPointer(), (size_t)s.length());
    }

    int getReturnCode() const { return hdr->rc; }
    String getString() const { return String(str, (size_t)hdr->size); }

    virtual void realign() override { realignInternal(); }

  private:
    void realignInternal() {
        hdr = reinterpret_cast<hdr_t*>(payloadBuffer.data());
        str =
            (size_t)getSize() > sizeof(hdr_t) ? reinterpret_cast<char*>(payloadBuffer.data()) + sizeof(hdr_t) : nullptr;
    }
};

class PluginList : public MsgPackPayload {
  public:
    static constexpr int Type = 10;
    PluginList() : MsgPackPayload(Type) {}
};

class AddPlugin : public JsonPayload {
  public:
    static constexpr int Type = 20;
    AddPlugin() : JsonPayload(Type) {}
};

class AddPluginResult : public JsonPayload {
  public:
    static constexpr int Type = 21;
    AddPluginResult() : JsonPayload(Type) {}
};

class DelPlugin : public NumberPayload {
  public:
    static constexpr int Type = 30;
    DelPlugin() : NumberPayload(Type) {}
};

class PluginStatus : public JsonPayload {
  public:
    static constexpr int Type = 40;
    PluginStatus() : JsonPayload(Type) {}
};

struct setmonochannels_t {
    int idx;
    uint64 channels;
};

class SetMonoChannels : public DataPayload<setmonochannels_t> {
  public:
    static constexpr int Type = 50;
    SetMonoChannels() : DataPayload<setmonochannels_t>(Type) {}
};

struct editplugin_t {
    int index;
    int channel;
    int x;
    int y;
};

class EditPlugin : public DataPayload<editplugin_t> {
  public:
    static constexpr int Type = 60;
    EditPlugin() : DataPayload<editplugin_t>(Type) {}
};

class HidePlugin : public NumberPayload {
  public:
    static constexpr int Type = 61;
    HidePlugin() : NumberPayload(Type) {}
};

class GetScreenBounds : public NumberPayload {
  public:
    static constexpr int Type = 62;
    GetScreenBounds() : NumberPayload(Type) {}
};

struct screenbounds_t {
    int x;
    int y;
    int w;
    int h;
};

class ScreenBounds : public DataPayload<screenbounds_t> {
  public:
    static constexpr int Type = 63;
    ScreenBounds() : DataPayload<screenbounds_t>(Type) {}
};

class ScreenCapture : public Payload {
  public:
    static constexpr int Type = 64;

    struct hdr_t {
        int width;
        int height;
        int widthPadded;
        int heightPadded;
        double scale;
        size_t size;
    };
    hdr_t* hdr;
    char* data;

    ScreenCapture() : Payload(Type) { realign(); }

    void setImage(int width, int height, int widthPadded, int heightPadded, double scale, const void* p, size_t size) {
        setSize((int)(sizeof(hdr_t) + size));
        hdr->width = width;
        hdr->height = height;
        hdr->widthPadded = widthPadded;
        hdr->heightPadded = heightPadded;
        hdr->scale = scale;
        hdr->size = size;
        if (nullptr != p) {
            memcpy(data, p, size);
        }
    }

    virtual void realign() override {
        hdr = reinterpret_cast<hdr_t*>(payloadBuffer.data());
        data =
            (size_t)getSize() > sizeof(hdr_t) ? reinterpret_cast<char*>(payloadBuffer.data()) + sizeof(hdr_t) : nullptr;
    }
};

class UpdateScreenCaptureArea : public NumberPayload {
  public:
    static constexpr int Type = 65;
    UpdateScreenCaptureArea() : NumberPayload(Type) {}
};

struct mouseevent_t {
    MouseEvType type;
    float x;
    float y;
    bool isShiftDown;
    bool isCtrlDown;
    bool isAltDown;
    // wheel parameters
    float deltaX;
    float deltaY;
    bool isSmooth;
};

class Mouse : public DataPayload<mouseevent_t> {
  public:
    static constexpr int Type = 66;
    Mouse() : DataPayload<mouseevent_t>(Type) {}
};

class Key : public BinaryPayload {
  public:
    static constexpr int Type = 67;
    Key() : BinaryPayload(Type) {}

    const uint16_t* getKeyCodes() const { return reinterpret_cast<const uint16_t*>(data); }
    int getKeyCount() const { return *size / (int)sizeof(uint16_t); }
};

class Clipboard : public StringPayload {
  public:
    static constexpr int Type = 68;
    Clipboard() : StringPayload(Type) {}
};

class GetPluginSettings : public NumberPayload {
  public:
    static constexpr int Type = 70;
    GetPluginSettings() : NumberPayload(Type) {}
};

class SetPluginSettings : public NumberPayload {
  public:
    static constexpr int Type = 71;
    SetPluginSettings() : NumberPayload(Type) {}
};

class PluginSettings : public StringPayload {
  public:
    static constexpr int Type = 72;
    PluginSettings() : StringPayload(Type) {}
};

class BypassPlugin : public NumberPayload {
  public:
    static constexpr int Type = 73;
    BypassPlugin() : NumberPayload(Type) {}
};

class UnbypassPlugin : public NumberPayload {
  public:
    static constexpr int Type = 74;
    UnbypassPlugin() : NumberPayload(Type) {}
};

struct exchange_t {
    int idxA;
    int idxB;
};

class ExchangePlugins : public DataPayload<exchange_t> {
  public:
    static constexpr int Type = 80;
    ExchangePlugins() : DataPayload<exchange_t>(Type) {}
};

class RecentsList : public StringPayload {
  public:
    static constexpr int Type = 90;
    RecentsList() : StringPayload(Type) {}
};

class Parameters : public MsgPackPayload {
  public:
    static constexpr int Type = 100;
    Parameters() : MsgPackPayload(Type) {}
};

struct parametervalue_t {
    int idx;
    int paramIdx;
    float value;
    int channel = 0;
};

class ParameterValue : public DataPayload<parametervalue_t> {
  public:
    static constexpr int Type = 101;
    ParameterValue() : DataPayload<parametervalue_t>(Type) {}
};

struct getparametervalue_t {
    int idx;
    int paramIdx;
    int channel = 0;
};

class GetParameterValue : public DataPayload<getparametervalue_t> {
  public:
    static constexpr int Type = 102;
    GetParameterValue() : DataPayload<getparametervalue_t>(Type) {}
};

class GetAllParameterValues : public NumberPayload {
  public:
    static constexpr int Type = 103;
    GetAllParameterValues() : NumberPayload(Type) {}
};

struct parametergesture_t {
    int idx;
    int paramIdx;
    bool gestureIsStarting;
    int channel = 0;
};

class ParameterGesture : public DataPayload<parametergesture_t> {
  public:
    static constexpr int Type = 104;
    ParameterGesture() : DataPayload<parametergesture_t>(Type) {}
};

class Presets : public StringPayload {
  public:
    static constexpr int Type = 110;
    Presets() : StringPayload(Type) {}
};

struct preset_t {
    int idx;
    int preset;
    int channel = 0;
};

class Preset : public DataPayload<preset_t> {
  public:
    static constexpr int Type = 111;
    Preset() : DataPayload<preset_t>(Type) {}
};

class Rescan : public NumberPayload {
  public:
    static constexpr int Type = 120;
    Rescan() : NumberPayload(Type) {}
};

class Restart : public Payload {
  public:
    static constexpr int Type = 121;
    Restart() : Payload(Type) {}
};

class CPULoad : public FloatPayload {
  public:
    static constexpr int Type = 130;
    CPULoad() : FloatPayload(Type) {}
};

class ServerError : public StringPayload {
  public:
    static constexpr int Type = 200;
    ServerError() : StringPayload(Type) {}
};

template <typename T>
class Message : public LogTagDelegate {
  public:
    static constexpr int MAX_SIZE = 1024 * 1024 * 60;  // 60 MB

    Message(const LogTag* tag = nullptr) : LogTagDelegate(tag) {
        traceScope();
        payload.setLogTagSource(tag);
        m_bytesIn = Metrics::getStatistic<Meter>("NetBytesIn");
        m_bytesOut = Metrics::getStatistic<Meter>("NetBytesOut");
    }

    struct Header {
        int type;
        int size;
    };

    virtual ~Message() {}

    bool read(StreamingSocket* socket, MessageHelper::Error* e = nullptr, int timeoutMilliseconds = 1000) {
        traceScope();
        traceln("type=" << T::Type);
        bool success = false;
        MessageHelper::seterr(e, MessageHelper::E_NONE);
        if (nullptr != socket && socket->isConnected()) {
            Header hdr;
            success = true;
            int ret = socket->waitUntilReady(true, timeoutMilliseconds);
            if (ret > 0) {
                if (e47::read(socket, &hdr, sizeof(hdr), 2000, e, m_bytesIn.get())) {
                    auto t = T::Type;
                    if (t > 0 && hdr.type != t) {
                        success = false;
                        String estr;
                        estr << "invalid message type " << hdr.type << " (" << t << " expected)";
                        MessageHelper::seterr(e, MessageHelper::E_DATA, estr);
                        traceln(estr);
                    } else {
                        payload.setType(hdr.type);
                        traceln("size=" << hdr.size);
                        if (hdr.size > 0) {
                            if (hdr.size > MAX_SIZE) {
                                success = false;
                                String estr;
                                estr << "max size of " << MAX_SIZE << " bytes exceeded (" << hdr.size << " bytes)";
                                MessageHelper::seterr(e, MessageHelper::E_DATA, estr);
                                traceln(estr);
                            } else {
                                if (payload.getSize() != hdr.size) {
                                    payload.setSize(hdr.size);
                                }
                                if (!e47::read(socket, payload.getData(), hdr.size, 2000, e, m_bytesIn.get())) {
                                    success = false;
                                    MessageHelper::seterr(e, MessageHelper::E_DATA, "failed to read message body");
                                    traceln("read of message body failed");
                                }
                            }
                        }
                    }
                } else {
                    success = false;
                    MessageHelper::seterr(e, MessageHelper::E_DATA, "failed to read message header");
                    traceln("read of message header failed");
                }
            } else if (ret < 0) {
                success = false;
                MessageHelper::seterr(e, MessageHelper::E_SYSCALL, "failed to wait for message header");
                traceln("failed: E_SYSCALL");
            } else {
                success = false;
                MessageHelper::seterr(e, MessageHelper::E_TIMEOUT);
                traceln("failed: E_TIMEOUT");
            }
        } else {
            MessageHelper::seterr(e, MessageHelper::E_STATE, "no socket or not connected");
            traceln("failed: E_STATE");
        }
        return success;
    }

    bool send(StreamingSocket* socket) {
        traceScope();
        traceln("type=" << T::Type);
        Header hdr = {payload.getType(), payload.getSize()};
        if (static_cast<size_t>(hdr.size) > MAX_SIZE) {
            std::cerr << "max size of " << MAX_SIZE << " bytes exceeded (" << hdr.size << " bytes)" << std::endl;
            return false;
        }
        if (!e47::send(socket, reinterpret_cast<const char*>(&hdr), sizeof(hdr), nullptr, m_bytesOut.get())) {
            return false;
        }
        if (payload.getSize() > 0 &&
            !e47::send(socket, payload.getData(), payload.getSize(), nullptr, m_bytesOut.get())) {
            return false;
        }
        return true;
    }

    int getType() const { return payload.getType(); }
    int getSize() const { return payload.getSize(); }
    const char* getData() const { return payload.getData(); }

    template <typename T2>
    static std::shared_ptr<Message<T2>> convert(std::shared_ptr<Message<T>> in) {
        auto out = std::make_shared<Message<T2>>(in->getLogTagSource());
        out->payload.payloadBuffer = std::move(in->payload.payloadBuffer);
        out->payload.realign();
        return out;
    }

    T payload;

  private:
    std::shared_ptr<Meter> m_bytesIn, m_bytesOut;
};

#define PLD(m) m.payload
#define pPLD(m) m->payload
#define DATA(m) PLD(m).data
#define pDATA(m) pPLD(m).data

class MessageFactory : public LogTagDelegate {
  public:
    MessageFactory(const LogTag* tag) : LogTagDelegate(tag) {}

    std::shared_ptr<Message<Any>> getNextMessage(StreamingSocket* socket, MessageHelper::Error* e, int timeout = 1000) {
        traceScope();
        if (nullptr != socket) {
            auto msg = std::make_shared<Message<Any>>(getLogTagSource());
            if (msg->read(socket, e, timeout)) {
                return msg;
            } else {
                traceln("read failed");
            }
        }
        traceln("no socket");
        return nullptr;
    }

    std::shared_ptr<Result> getResult(StreamingSocket* socket, int attempts = 5, MessageHelper::Error* e = nullptr) {
        traceScope();
        if (nullptr != socket) {
            auto msg = std::make_shared<Message<Result>>(getLogTagSource());
            MessageHelper::Error err;
            int count = 0;
            do {
                if (msg->read(socket, &err)) {
                    auto res = std::make_shared<Result>();
                    *res = std::move(msg->payload);
                    return res;
                } else {
                    traceln("read failed");
                }
            } while (++count < attempts && err.code == MessageHelper::E_TIMEOUT);
            if (nullptr != e) {
                *e = err;
                String m = "unable to retrieve result message after ";
                m << attempts;
                m << " attempts";
                MessageHelper::seterrstr(e, m);
                traceln(m);
                return nullptr;
            }
        } else {
            traceln("no socket");
            MessageHelper::seterr(e, MessageHelper::E_STATE, "no socket");
        }
        return nullptr;
    }

    bool sendResult(StreamingSocket* socket, int rc) { return sendResult(socket, rc, ""); }

    bool sendResult(StreamingSocket* socket, int rc, const String& str) {
        traceScope();
        Message<Result> msg(getLogTagSource());
        msg.payload.setResult(rc, str);
        return msg.send(socket);
    }
};

}  // namespace e47

#endif

namespace e47 {

struct JsonMessage {
    using Type = uint16;

    Type type;
    Uuid id;
    json data;

    JsonMessage() : type(0) {}
    JsonMessage(Type t, const json& d) : type(t), data(d) {}
    JsonMessage(Type t, const json& d, const String& i) : JsonMessage(t, d) { id = i; }

    void serialize(MemoryBlock& block) const {
        json dataJson;
        dataJson["type"] = type;
        dataJson["uuid"] = id.toString().toStdString();
        dataJson["data"] = data;
        auto dump = dataJson.dump();
        block.append(dump.data(), dump.length());
    }

    void deserialize(const MemoryBlock& block) {
        try {
            auto j = json::parse(block.begin(), block.end());
            type = j["type"].get<Type>();
            data = std::move(j["data"]);
            id = j["uuid"].get<std::string>();
        } catch (const json::parse_error& e) {
            setLogTagStatic("json");
            logln("failed to deserialize json message: " << e.what());
        }
    }
};

struct PluginTrayMessage : JsonMessage {
    enum Type : JsonMessage::Type { STATUS, CHANGE_SERVER, RELOAD, GET_RECENTS, UPDATE_RECENTS, SHOW_MONITOR, STOP };
    PluginTrayMessage() {}
    PluginTrayMessage(Type t, const json& d) : JsonMessage(t, d) {}
    PluginTrayMessage(Type t, const json& d, const String& i) : JsonMessage(t, d, i) {}
};

struct SandboxMessage : JsonMessage {
    enum Type : JsonMessage::Type { CONFIG, SANDBOX_PORT, SHOW_EDITOR, HIDE_EDITOR, METRICS };
    SandboxMessage() {}
    SandboxMessage(Type t, const json& d) : JsonMessage(t, d) {}
    SandboxMessage(Type t, const json& d, const String& i) : JsonMessage(t, d, i) {}
};

}  // namespace e47

#endif /* Message_hpp */
