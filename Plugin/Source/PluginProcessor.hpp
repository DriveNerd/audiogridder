/*
 * Copyright (c) 2020 Andreas Pohl
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * Author: Andreas Pohl
 */

#pragma once

#include <JuceHeader.h>
#include <set>

#include "Client.hpp"
#include "Utils.hpp"
#include "json.hpp"
#include "ChannelSet.hpp"
#include "ChannelMapper.hpp"
#include "AudioRingBuffer.hpp"

using json = nlohmann::json;

#ifndef JucePlugin_Name
#define JucePlugin_Name "AGridder"
#endif

namespace e47 {

class WrapperTypeReaderAudioProcessor : public AudioProcessor {
  public:
    const String getName() const override { return {}; }
    void prepareToPlay(double, int) override {}
    void releaseResources() override {}
    void processBlock(AudioBuffer<float>&, MidiBuffer&) override {}
    void processBlock(AudioBuffer<double>&, MidiBuffer&) override {}
    double getTailLengthSeconds() const override { return 0.0; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    AudioProcessorEditor* createEditor() override { return nullptr; }
    bool hasEditor() const override { return false; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const String getProgramName(int) override { return {}; }
    void changeProgramName(int, const String&) override {}
    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}
};

class PluginProcessor : public AudioProcessor, public AudioProcessorParameter::Listener, public LogTagDelegate {
  public:
    PluginProcessor(WrapperType wt);
    ~PluginProcessor() override;

    void prepareToPlay(double sampleRate, int blockSize) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    Array<std::pair<short, short>> getAUChannelInfo() const override;

    bool canAddBus(bool /*isInput*/) const override { return true; }
    bool canRemoveBus(bool /*isInput*/) const override { return true; }
    void numChannelsChanged() override;

    int getCustomBlockSize() const;
    void setCustomBlockSize(int b);

    void processBlock(AudioBuffer<float>& buf, MidiBuffer& midi) override { processBlockInternal(buf, midi); }
    void processBlock(AudioBuffer<double>& buf, MidiBuffer& midi) override { processBlockInternal(buf, midi); }

    void processBlockBypassed(AudioBuffer<float>& buf, MidiBuffer& /*midi*/) override {
        processBlockBypassedInternal(buf, m_bypassBufferF);
    }

    void processBlockBypassed(AudioBuffer<double>& buf, MidiBuffer& /*midi*/) override {
        processBlockBypassedInternal(buf, m_bypassBufferD);
    }

    void updateLatency();

    AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const String getName() const override;
    StringArray getAlternateDisplayNames() const override { return {"AGrid", "AG"}; }

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    bool supportsDoublePrecisionProcessing() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const String getProgramName(int index) override;
    void changeProgramName(int index, const String& newName) override;

    void getStateInformation(MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    json getState(bool withActiveServer);
    bool setState(const json& j);

    const String& getMode() const { return m_mode; }

    void updateTrackProperties(const TrackProperties& properties) override {
        traceScope();
        std::lock_guard<std::mutex> lock(m_trackPropertiesMtx);
        m_trackProperties = properties;
    }

    TrackProperties getTrackProperties() {
        traceScope();
        std::lock_guard<std::mutex> lock(m_trackPropertiesMtx);
        return m_trackProperties;
    }

    void loadConfig();
    void loadConfig(const json& j, bool isUpdate = false);
    void saveConfig(int numOfBuffers = -1, bool saveBufferDefaults = false);

    Client& getClient() { return *m_client; }
    std::vector<ServerPlugin> getPlugins(const String& type) const;
    const std::vector<ServerPlugin>& getPlugins() const { return m_client->getPlugins(); }
    std::set<String> getPluginTypes() const;

    struct LoadedPlugin {
        enum Indexes : uint8 {
            ID_DEPRECATED,
            NAME,
            SETTINGS,
            PRESETS,
            PARAMS,
            BYPASSED,
            ID,
            LAYOUT,
            MONO_CHANNELS,
            ACTIVE_CHANNEL,
            PARAMSLIST
        };
        enum Indexes_v1 : uint8 { BYPASSED_V1 = 3 };

        String idDeprecated;
        String name;
        String layout;
        ChannelSet monoChannels = 0;
        int activeChannel = 0;
        String settings;
        StringArray presets;
        Client::ParameterByChannelList params;
        bool bypassed = false;
        String id;

        bool hasEditor = true;
        bool ok = false;
        String error;

        json toJson() {
            auto jpresets = json::array();
            for (auto& p : presets) {
                jpresets.push_back(p.toStdString());
            }
            // for backwards compatibility
            auto jparamsLegacy = json::array();
            if (params.size() > 0) {
                for (auto& p : params[0]) {
                    jparamsLegacy.push_back(p.toJson());
                }
            }
            auto jparams = json::array();
            for (size_t ch = 0; ch < params.size(); ch++) {
                jparams.push_back(json::array());
                for (auto& p : params[ch]) {
                    jparams[ch].push_back(p.toJson());
                }
            }
            return {idDeprecated.toStdString(),
                    name.toStdString(),
                    settings.toStdString(),
                    jpresets,
                    jparamsLegacy,
                    bypassed,
                    id.toStdString(),
                    layout.toStdString(),
                    monoChannels.toInt(),
                    activeChannel,
                    jparams};
        }

        LoadedPlugin() {}

        LoadedPlugin(const json& j, int version) : monoChannels(0, 0, Defaults::PLUGIN_CHANNELS_MAX) {
            try {
                idDeprecated = j[ID_DEPRECATED].get<std::string>();
                name = j[NAME].get<std::string>();
                settings = j[SETTINGS].get<std::string>();
                if (version == 1) {
                    bypassed = j[BYPASSED_V1].get<bool>();
                } else if (version > 1) {
                    bypassed = j[BYPASSED].get<bool>();
                }
                if (version >= 2) {
                    for (auto& p : j[PRESETS]) {
                        presets.add(p.get<std::string>());
                    }
                    if (version < 5) {
                        params.resize(1);
                        for (auto& p : j[PARAMS]) {
                            params[0].push_back(Client::Parameter::fromJson(p));
                        }
                    }
                }
                if (version >= 3) {
                    id = j[ID].get<std::string>();
                } else {
                    id = idDeprecated;
                }
                if (version >= 4) {
                    layout = j[LAYOUT].get<std::string>();
                    monoChannels = j[MONO_CHANNELS].get<uint64>();
                }
                if (version >= 5) {
                    activeChannel = j[ACTIVE_CHANNEL].get<int>();
                    // version 5 was broken, as the structure changed
                    auto paramsListIdx = version == 5 ? PARAMS : PARAMSLIST;
                    params.resize(j[paramsListIdx].size());
                    for (size_t ch = 0; ch < j[paramsListIdx].size(); ch++) {
                        for (auto& p : j[paramsListIdx][ch]) {
                            params[ch].push_back(Client::Parameter::fromJson(p));
                        }
                    }
                }
            } catch (const json::exception& e) {
                setLogTagStatic("loadedplugin");
                logln("failed to deserialize loaded plugin: " << e.what());
            }
        }

        LoadedPlugin(const String& id_, const String& idDeprecated_, const String& name_, const String& layout_,
                     const ChannelSet& monoChannels_, int activeChannel_, const String& settings_,
                     const StringArray& presets_, const Client::ParameterByChannelList& params_, bool bypassed_,
                     bool hasEditor_, bool ok_, const String& error_)
            : idDeprecated(idDeprecated_),
              name(name_),
              layout(layout_),
              monoChannels(monoChannels_),
              activeChannel(activeChannel_),
              settings(settings_),
              presets(presets_),
              params(params_),
              bypassed(bypassed_),
              id(id_),
              hasEditor(hasEditor_),
              ok(ok_),
              error(error_) {}

        const Client::ParameterList& getActiveParams() const { return params[(size_t)activeChannel]; }
        Client::ParameterList& getActiveParams() { return params[(size_t)activeChannel]; }
    };

    // Called by the client to trigger resyncing the remote plugin settings
    void sync();

    // Called by the client to check for failed plugins that we want to auto retry
    void autoRetry();

    enum SyncRemoteMode { SYNC_ALWAYS, SYNC_WITH_EDITOR, SYNC_DISABLED };
    SyncRemoteMode getSyncRemoteMode() const { return m_syncRemote; }
    void setSyncRemoteMode(SyncRemoteMode m) { m_syncRemote = m; }

    ChannelSet& getActiveChannels() { return m_activeChannels; }

    void updateChannelMapping() {
        m_channelMapper.createPluginMapping(m_activeChannels);
        m_channelMapper.print();
    }

    int getNumOfLoadedPlugins() { return (int)m_loadedPluginsCount; }

    LoadedPlugin& getLoadedPlugin(int idx) {
        std::lock_guard<std::mutex> lock(m_loadedPluginsSyncMtx);
        return getLoadedPluginNoLock(idx);
    }

    bool loadPlugin(const ServerPlugin& plugin, const String& layout, uint64 monoChannels, String& err);
    void unloadPlugin(int idx);
    String getLoadedPluginsString() const;
    void editPlugin(int idx, int channel, int x, int y);
    void hidePlugin(bool updateServer = true);
    void hidePluginFromServer(int idx);
    void enableMonoChannel(int idx, int channel);
    void disableMonoChannel(int idx, int channel);
    int getActivePlugin() const { return m_activePlugin; }
    int getActivePluginChannel() { return getLoadedPlugin(m_activePlugin).activeChannel; }
    StringArray getOutputChannelNames() const;
    String getPluginChannelName(int ch);
    String getActivePluginChannelName() { return getPluginChannelName(getActivePluginChannel()); }
    int getLastActivePlugin() const { return m_lastActivePlugin; }
    bool isEditAlways() const { return m_editAlways; }
    void setEditAlways(bool b) { m_editAlways = b; }
    bool isBypassed(int idx);
    void bypassPlugin(int idx);
    void unbypassPlugin(int idx);
    void exchangePlugins(int idxA, int idxB);
    bool enableParamAutomation(int idx, int channel, int paramIdx, int slot = -1);
    void disableParamAutomation(int idx, int channel, int paramIdx);
    void getAllParameterValues(int idx);
    void updateParameterValue(int idx, int channel, int paramIdx, float val, bool updateServer = true);
    void updateParameterGestureTracking(int idx, int channel, int paramIdx, bool starting);
    void updatePluginStatus(int idx, bool ok, const String& err);
    void increaseSCArea();
    void decreaseSCArea();
    void toggleFullscreenSCArea();

    void storeSettingsA();
    void storeSettingsB();
    void restoreSettingsA();
    void restoreSettingsB();
    void resetSettingsAB();

    bool getMenuShowType() const { return m_menuShowType; }
    void setMenuShowType(bool b) { m_menuShowType = b; }
    bool getMenuShowCategory() const { return m_menuShowCategory; }
    void setMenuShowCategory(bool b) { m_menuShowCategory = b; }
    bool getMenuShowCompany() const { return m_menuShowCompany; }
    void setMenuShowCompany(bool b) { m_menuShowCompany = b; }
    bool getGenericEditor() const { return m_genericEditor; }
    void setGenericEditor(bool b) { m_genericEditor = b; }
    bool getConfirmDelete() const { return m_confirmDelete; }
    void setConfirmDelete(bool b) { m_confirmDelete = b; }
    bool getShowSidechainDisabledInfo() const { return m_showSidechainDisabledInfo; }
    void setShowSidechainDisabledInfo(bool b) { m_showSidechainDisabledInfo = b; }
    bool getNoSrvPluginListFilter() const { return m_noSrvPluginListFilter; }
    void setNoSrvPluginListFilter(bool b) { m_noSrvPluginListFilter = b; }
    float getScaleFactor() const { return m_scale; }
    void setScaleFactor(float f) { m_scale = f; }
    bool getCrashReporting() const { return m_crashReporting; }
    void setCrashReporting(bool b) { m_crashReporting = b; }
    bool supportsCrashReporting() const { return wrapperType != wrapperType_AAX; }
    Array<ServerPlugin> getRecents();
    void updateRecents(const ServerPlugin& plugin);

    auto& getServers() const { return m_servers; }
    void addServer(const String& s) { m_servers.add(s); }
    void delServer(const String& s);
    String getActiveServerHost() const { return m_client->getServer().getHostAndID(); }
    String getActiveServerName() const;
    void setActiveServer(const ServerInfo& s);
    Array<ServerInfo> getServersMDNS();
    void setCPULoad(float load);

    int getLatencyMillis() const {
        return (int)lround(m_client->NUM_OF_BUFFERS * getCustomBlockSize() * 1000 / getSampleRate());
    }

    void showMonitor() {
        if (m_tray != nullptr) {
            m_tray->showMonitor();
        }
    }

    String getPresetDir() const { return m_presetsDir; }
    void setPresetDir(const String& d) { m_presetsDir = d; }
    bool hasDefaultPreset() const { return m_defaultPreset.isNotEmpty() && File(m_defaultPreset).existsAsFile(); }
    void storePreset(const File& file);
    bool loadPreset(const File& file);
    void storePresetDefault();
    void resetPresetDefault();

    enum TransferMode : int { TM_ALWAYS, TM_WHEN_PLAYING, TM_WITH_MIDI };

    TransferMode getTransferMode() const {
        return (TransferMode)(m_mode == "FX" ? m_transferModeFx.load() : m_transferModeMidi.load());
    }

    void setTransferMode(TransferMode m) {
        if (m_mode == "FX") {
            m_transferModeFx = m;
        } else {
            m_transferModeMidi = m;
        }
    }

    bool getDisableTray() const { return m_disableTray; }
    void setDisableTray(bool b);
    bool getDisableRecents() const { return m_disableRecents; }
    void setDisableRecents(bool b) { m_disableRecents = b; }
    bool getKeepEditorOpen() const { return m_keepEditorOpen; }
    void setKeepEditorOpen(bool b) { m_keepEditorOpen = b; }
    bool getBypassWhenNotConnected() const { return m_bypassWhenNotConnected; }
    void setBypassWhenNotConnected(bool b) { m_bypassWhenNotConnected = b; }
    bool getBufferSizeByPlugin() const { return m_bufferSizeByPlugin; }
    void setBufferSizeByPlugin(bool b) { m_bufferSizeByPlugin = b; }
    bool getFixedOutboundBuffer() const { return m_client->FIXED_OUTBOUND_BUFFER; }
    void setFixedOutboundBuffer(bool b);

    int getNumBuffers() const { return m_client->NUM_OF_BUFFERS; }
    void setNumBuffers(int n);

    // AudioProcessorParameter::Listener
    void parameterValueChanged(int parameterIndex, float newValue) override;
    void parameterGestureChanged(int, bool) override {}

    // It looks like most hosts do not support dynamic parameter creation or changes to existing parameters. Logic
    // at least allows for the name to be updated. So we create slots at the start.
    class Parameter : public AudioProcessorParameter, public LogTagDelegate {
      public:
        Parameter(PluginProcessor& proc, int slot) : m_proc(proc), m_slotId(slot) {
            setLogTagSource(m_proc.getLogTagSource());
            initAsyncFunctors();
        }
        ~Parameter() override {
            traceScope();
            stopAsyncFunctors();
        }
        float getValue() const override;
        void setValue(float newValue) override;
        float getValueForText(const String& /* text */) const override { return 0; }
        float getDefaultValue() const override { return getParam().defaultValue; }
        String getName(int maximumStringLength) const override;
        String getLabel() const override { return getParam().label; }
        int getNumSteps() const override { return getParam().numSteps; }
        bool isDiscrete() const override { return getParam().isDiscrete; }
        bool isBoolean() const override { return getParam().isBoolean; }
        bool isOrientationInverted() const override { return getParam().isOrientInv; }
        bool isMetaParameter() const override { return getParam().isMeta; }

      private:
        friend PluginProcessor;
        PluginProcessor& m_proc;
        int m_idx = -1;
        int m_channel = 0;
        int m_paramIdx = 0;
        int m_slotId = 0;

        const LoadedPlugin& getPlugin() const { return m_proc.getLoadedPluginNoLock(m_idx); }
        LoadedPlugin& getPlugin() { return m_proc.getLoadedPluginNoLock(m_idx); }
        const Client::Parameter& getParam() const { return getPlugin().params[(size_t)m_channel][(size_t)m_paramIdx]; }
        Client::Parameter& getParam() { return getPlugin().params[(size_t)m_channel][(size_t)m_paramIdx]; }

        void reset() {
            m_idx = -1;
            m_channel = 0;
            m_paramIdx = 0;
        }

        ENABLE_ASYNC_FUNCTORS();
    };

    class TrayConnection : public InterprocessConnection, public Thread, public LogTagDelegate {
      public:
        std::atomic_bool connected{false};

        TrayConnection(PluginProcessor* p)
            : InterprocessConnection(false), Thread("TrayConnection"), LogTagDelegate(p), m_processor(p) {}

        ~TrayConnection() override { stopThread(-1); }

        void run() override;

        void connectionMade() override { connected = true; }
        void connectionLost() override { connected = false; }
        void messageReceived(const MemoryBlock& message) override;
        void sendStatus();
        void sendStop();
        void showMonitor();
        void sendMessage(const PluginTrayMessage& msg);

        Array<ServerPlugin> getRecents() {
            std::lock_guard<std::mutex> lock(m_recentsMtx);
            return m_recents;
        }

      private:
        PluginProcessor* m_processor;
        Array<ServerPlugin> m_recents;
        std::mutex m_recentsMtx;
        std::mutex m_sendMtx;
    };

  private:
    Uuid m_instId;
    String m_mode;
    std::unique_ptr<Client> m_client;
    std::unique_ptr<TrayConnection> m_tray;
    std::atomic_bool m_prepared{false};
    std::vector<LoadedPlugin> m_loadedPlugins;
    mutable std::mutex m_loadedPluginsSyncMtx;
    std::atomic_bool m_loadedPluginsOk{false};
    std::atomic_uint64_t m_loadedPluginsCount{0};
    int m_autoReconnects = 0;
    int m_activePlugin = -1;
    int m_lastActivePlugin = -1;
    bool m_editAlways = true;
    StringArray m_servers;
    String m_activeServerFromCfg;
    int m_activeServerLegacyFromCfg;
    String m_presetsDir;
    String m_defaultPreset;
    int m_customBlockSize = 0;

    int m_numberOfBuffersDefault = Defaults::DEFAULT_NUM_OF_BUFFERS;
    int m_customBlockSizeDefault = 0;
    bool m_fixedOutboundBufferDefault = false;

    int m_numberOfAutomationSlots = 16;
    LoadedPlugin m_unusedDummyPlugin;
    Client::Parameter m_unusedParam;

    AudioRingBuffer<float> m_bypassBufferF;
    AudioRingBuffer<double> m_bypassBufferD;
    std::mutex m_bypassBufferMtx;

    String m_settingsA, m_settingsB;

    bool m_menuShowType = true;
    bool m_menuShowCategory = true;
    bool m_menuShowCompany = true;
    bool m_genericEditor = false;
    bool m_confirmDelete = true;
    bool m_showSidechainDisabledInfo = true;
    bool m_noSrvPluginListFilter = false;
    float m_scale = 1.0;
    bool m_crashReporting = true;

    std::atomic_int m_transferModeFx{TM_ALWAYS};
    std::atomic_int m_transferModeMidi{TM_ALWAYS};

    bool m_disableTray = false;
    bool m_disableRecents = false;
    bool m_keepEditorOpen = false;
    std::atomic_bool m_bypassWhenNotConnected{false};
    bool m_bufferSizeByPlugin = false;

    TrackProperties m_trackProperties;
    std::mutex m_trackPropertiesMtx;

    SyncRemoteMode m_syncRemote = SYNC_WITH_EDITOR;

    ChannelSet m_activeChannels;
    ChannelMapper m_channelMapper;

    bool m_activeMidiNotes[128];
    bool m_midiIsPlaying = false;
    int m_blocksWithoutMidi = 0;

    double m_processingTraceTresholdMs = 0.0;
    TimeStatistic::Duration m_processingDurationGlobal;
    TimeStatistic::Duration m_processingDurationLocal;

    static BusesProperties createBusesProperties(WrapperType wt) {
        int chIn = Defaults::PLUGIN_CHANNELS_IN;
        int chOut = Defaults::PLUGIN_CHANNELS_OUT;
        int chSC = Defaults::PLUGIN_CHANNELS_SC;
        bool useMonoOutputBuses = true;
        bool useMultipleOutputBuses = false;

#if JucePlugin_IsSynth
        chIn = 0;
        useMultipleOutputBuses = true;
        if (wt == WrapperType::wrapperType_AudioUnit) {
            chOut = 2;
            useMultipleOutputBuses = false;
        } else if (wt == WrapperType::wrapperType_AAX) {
            useMonoOutputBuses = false;
        }
#else
        ignoreUnused(wt);
#endif

        auto bp = BusesProperties();

        if (chIn == 1) {
            bp = bp.withInput("Input", AudioChannelSet::mono(), true);
        } else if (chIn == 2) {
            bp = bp.withInput("Input", AudioChannelSet::stereo(), true);
        } else if (chIn > 0) {
            bp = bp.withInput("Input", AudioChannelSet::discreteChannels(chIn), true);
        }

        if (chOut == 1) {
            bp = bp.withOutput("Output", AudioChannelSet::mono(), true);
        } else if (chOut == 2) {
            bp = bp.withOutput("Output", AudioChannelSet::stereo(), true);
        } else if (chOut > 0) {
            if (useMultipleOutputBuses) {
                bp = bp.withOutput("Main", AudioChannelSet::stereo(), true);
                if (useMonoOutputBuses) {
                    for (int i = 2; i < chOut; i++) {
                        bp = bp.withOutput("Ch " + String(i + 1), AudioChannelSet::mono(), true);
                    }
                } else {
                    for (int i = 2; i < chOut; i += 2) {
                        bp = bp.withOutput("Ch " + String(i / 2 + 1), AudioChannelSet::stereo(), true);
                    }
                }
            } else {
                bp = bp.withOutput("Output", AudioChannelSet::discreteChannels(chOut), true);
            }
        }

        if (chSC == 1) {
            bp = bp.withInput("Sidechain", AudioChannelSet::mono(), true);
        } else if (chSC == 2) {
            bp = bp.withInput("Sidechain", AudioChannelSet::stereo(), true);
        } else if (chSC > 0) {
            bp = bp.withInput("Sidechain", AudioChannelSet::discreteChannels(chSC), true);
        }

        return bp;
    }

    template <typename T>
    void processBlockInternal(AudioBuffer<T>& buf, MidiBuffer& midi);

    template <typename T>
    void processBlockBypassedInternal(AudioBuffer<T>& buf, AudioRingBuffer<T>& bypassBuffer);

    LoadedPlugin& getLoadedPluginNoLock(int idx) {
        return idx > -1 && idx < (int)m_loadedPlugins.size() ? m_loadedPlugins[(size_t)idx] : m_unusedDummyPlugin;
    }

    ENABLE_ASYNC_FUNCTORS();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

}  // namespace e47
