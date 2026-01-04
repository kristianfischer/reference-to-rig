#pragma once

#include <JuceHeader.h>

class AudioPlayerComponent : public juce::Component,
                              public juce::ChangeListener,
                              public juce::Timer
{
public:
    AudioPlayerComponent();
    ~AudioPlayerComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;

    void setAudioUrls(const juce::String& originalUrl, const juce::String& processedUrl);
    void loadAudioFiles(const juce::File& originalFile, const juce::File& processedFile);
    void timerCallback() override;

private:
    enum class PlaybackMode { Original, Processed, AB };
    PlaybackMode currentMode = PlaybackMode::Original;
    
    juce::AudioFormatManager formatManager;
    juce::AudioTransportSource transportSource;
    std::unique_ptr<juce::AudioFormatReaderSource> readerSource;
    
    juce::TextButton playButton;
    juce::TextButton stopButton;
    juce::TextButton originalButton;
    juce::TextButton processedButton;
    juce::TextButton abButton;
    
    juce::Slider positionSlider;
    juce::Label timeLabel;
    juce::Label titleLabel;
    
    juce::String originalAudioUrl;
    juce::String processedAudioUrl;
    
    // Colors
    juce::Colour bgColor{ 0xff16213e };
    juce::Colour accentColor{ 0xff00d4ff };
    juce::Colour textColor{ 0xffeaeaea };
    
    void updateButtonStates();
    void loadUrl(const juce::String& url);
    juce::String formatTime(double seconds);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPlayerComponent)
};

