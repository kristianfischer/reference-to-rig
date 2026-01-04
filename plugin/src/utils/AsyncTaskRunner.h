#pragma once

#include <JuceHeader.h>
#include <functional>
#include <future>

/**
 * Utility class for running async tasks with proper JUCE message thread handling.
 * Ensures callbacks are always called on the message thread.
 */
class AsyncTaskRunner
{
public:
    AsyncTaskRunner() = default;
    ~AsyncTaskRunner() = default;

    /**
     * Run a task on a background thread and call the callback on the message thread
     * when complete.
     */
    template<typename ResultType>
    static void run(std::function<ResultType()> task, 
                    std::function<void(ResultType)> callback)
    {
        std::thread([task = std::move(task), callback = std::move(callback)]() {
            auto result = task();
            juce::MessageManager::callAsync([callback = std::move(callback), result = std::move(result)]() {
                callback(result);
            });
        }).detach();
    }

    /**
     * Run a void task on a background thread and call the callback on the message thread
     * when complete.
     */
    static void runVoid(std::function<void()> task, 
                        std::function<void()> callback = nullptr)
    {
        std::thread([task = std::move(task), callback = std::move(callback)]() {
            task();
            if (callback)
            {
                juce::MessageManager::callAsync([callback = std::move(callback)]() {
                    callback();
                });
            }
        }).detach();
    }

    /**
     * Schedule a callback to run on the message thread after a delay.
     */
    static void delayed(int milliseconds, std::function<void()> callback)
    {
        juce::Timer::callAfterDelay(milliseconds, [callback = std::move(callback)]() {
            callback();
        });
    }

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AsyncTaskRunner)
};

/**
 * A cancellable polling timer for checking task status.
 */
class PollingTimer : public juce::Timer
{
public:
    PollingTimer() = default;
    ~PollingTimer() override { stopTimer(); }

    void startPolling(int intervalMs, std::function<void()> callback)
    {
        pollCallback = std::move(callback);
        startTimer(intervalMs);
    }

    void stopPolling()
    {
        stopTimer();
        pollCallback = nullptr;
    }

    void timerCallback() override
    {
        if (pollCallback)
            pollCallback();
    }

private:
    std::function<void()> pollCallback;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PollingTimer)
};

