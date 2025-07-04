/*
 * Copyright (c) 2020 Andreas Pohl
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * Author: Andreas Pohl
 */

#ifndef Worker_hpp
#define Worker_hpp

#include <JuceHeader.h>
#include <thread>

#include "AudioWorker.hpp"
#include "Message.hpp"
#include "ScreenWorker.hpp"
#include "Utils.hpp"

namespace e47 {

class Server;

class Worker : public Thread, public LogTag {
  public:
    static std::atomic_uint32_t count;
    static std::atomic_uint32_t runCount;

    Worker(std::shared_ptr<StreamingSocket> masterSocket, const HandshakeRequest& cfg, int sandboxModeRuntime = 0);

    ~Worker() override;
    void run() override;

    void shutdown();

    void handleMessage(std::shared_ptr<Message<Quit>> msg);
    void handleMessage(std::shared_ptr<Message<AddPlugin>> msg);
    void handleMessage(std::shared_ptr<Message<DelPlugin>> msg);
    void handleMessage(std::shared_ptr<Message<EditPlugin>> msg);
    void handleMessage(std::shared_ptr<Message<HidePlugin>> msg, bool fromMaster = false);
    void handleMessage(std::shared_ptr<Message<Mouse>> msg);
    void handleMessage(std::shared_ptr<Message<Key>> msg);
    void handleMessage(std::shared_ptr<Message<GetPluginSettings>> msg);
    void handleMessage(std::shared_ptr<Message<SetPluginSettings>> msg);
    void handleMessage(std::shared_ptr<Message<BypassPlugin>> msg);
    void handleMessage(std::shared_ptr<Message<UnbypassPlugin>> msg);
    void handleMessage(std::shared_ptr<Message<ExchangePlugins>> msg);
    void handleMessage(std::shared_ptr<Message<RecentsList>> msg);
    void handleMessage(std::shared_ptr<Message<Preset>> msg);
    void handleMessage(std::shared_ptr<Message<ParameterValue>> msg);
    void handleMessage(std::shared_ptr<Message<GetParameterValue>> msg);
    void handleMessage(std::shared_ptr<Message<GetAllParameterValues>> msg);
    void handleMessage(std::shared_ptr<Message<UpdateScreenCaptureArea>> msg);
    void handleMessage(std::shared_ptr<Message<Rescan>> msg);
    void handleMessage(std::shared_ptr<Message<Restart>> msg);
    void handleMessage(std::shared_ptr<Message<CPULoad>> msg);
    void handleMessage(std::shared_ptr<Message<PluginList>> msg);
    void handleMessage(std::shared_ptr<Message<GetScreenBounds>> msg);
    void handleMessage(std::shared_ptr<Message<Clipboard>> msg);
    void handleMessage(std::shared_ptr<Message<SetMonoChannels>> msg);

  private:
    std::shared_ptr<StreamingSocket> m_masterSocket;
    std::unique_ptr<StreamingSocket> m_cmdIn;
    std::unique_ptr<StreamingSocket> m_cmdOut;
    std::mutex m_cmdOutMtx;
    HandshakeRequest m_cfg;
    std::shared_ptr<AudioWorker> m_audio;
    std::shared_ptr<ScreenWorker> m_screen;
    std::atomic_int m_activeEditorIdx{-1};
    MessageFactory m_msgFactory;
    bool m_noPluginListFilter = false;
    int m_sandboxModeRuntime = 0;

    struct KeyWatcher : KeyListener {
        Worker* worker;
        KeyWatcher(Worker* w) : worker(w) {}
        bool keyPressed(const KeyPress& kp, Component*);
    };

    struct ClipboardTracker : Timer {
        String current;
        Worker* worker;

        ClipboardTracker(Worker* w) : worker(w) {}
        ~ClipboardTracker() override { stopTimer(); }

        void start() { startTimer(200); }
        void stop() { stopTimer(); }

        void timerCallback() override {
            auto val = SystemClipboard::getTextFromClipboard();
            if (val != current) {
                current = val;
                worker->sendClipboard(val);
            }
        }
    };

    std::unique_ptr<KeyWatcher> m_keyWatcher;
    std::unique_ptr<ClipboardTracker> m_clipboardTracker;

    void sendKeys(const std::vector<uint16_t>& keysToPress);
    void sendClipboard(const String& val);
    void sendParamValueChange(int idx, int channel, int paramIdx, float val);
    void sendParamGestureChange(int idx, int channel, int paramIdx, bool guestureIsStarting);
    void sendStatusChange(int idx, bool ok, const String& err);
    void sendHideEditor(int idx);
    void sendError(const String& error);

    ENABLE_ASYNC_FUNCTORS();
};

}  // namespace e47

#endif /* Worker_hpp */
