#include "AudioPlayerComponent.h"

AudioPlayerComponent::AudioPlayerComponent()
{
    formatManager.registerBasicFormats();
    transportSource.addChangeListener(this);
    
    titleLabel.setText("A/B Comparison", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(16.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, accentColor);
    addAndMakeVisible(titleLabel);
    
    playButton.setButtonText("Play");
    playButton.setColour(juce::TextButton::buttonColourId, accentColor);
    playButton.setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    playButton.onClick = [this]() {
        if (transportSource.isPlaying())
            transportSource.stop();
        else
            transportSource.start();
    };
    addAndMakeVisible(playButton);
    
    stopButton.setButtonText("Stop");
    stopButton.setColour(juce::TextButton::buttonColourId, bgColor.brighter(0.3f));
    stopButton.onClick = [this]() {
        transportSource.stop();
        transportSource.setPosition(0);
    };
    addAndMakeVisible(stopButton);
    
    originalButton.setButtonText("Original");
    originalButton.setColour(juce::TextButton::buttonColourId, accentColor);
    originalButton.setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    originalButton.onClick = [this]() {
        currentMode = PlaybackMode::Original;
        updateButtonStates();
        loadUrl(originalAudioUrl);
    };
    addAndMakeVisible(originalButton);
    
    processedButton.setButtonText("Processed");
    processedButton.setColour(juce::TextButton::buttonColourId, bgColor.brighter(0.2f));
    processedButton.onClick = [this]() {
        currentMode = PlaybackMode::Processed;
        updateButtonStates();
        loadUrl(processedAudioUrl);
    };
    addAndMakeVisible(processedButton);
    
    abButton.setButtonText("A/B");
    abButton.setColour(juce::TextButton::buttonColourId, bgColor.brighter(0.2f));
    abButton.onClick = [this]() {
        currentMode = PlaybackMode::AB;
        updateButtonStates();
    };
    addAndMakeVisible(abButton);
    
    positionSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    positionSlider.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    positionSlider.setColour(juce::Slider::trackColourId, bgColor.brighter(0.1f));
    positionSlider.setColour(juce::Slider::thumbColourId, accentColor);
    positionSlider.setRange(0.0, 1.0);
    positionSlider.onValueChange = [this]() {
        if (!transportSource.isPlaying())
        {
            auto length = transportSource.getLengthInSeconds();
            transportSource.setPosition(positionSlider.getValue() * length);
        }
    };
    addAndMakeVisible(positionSlider);
    
    timeLabel.setFont(juce::Font(12.0f));
    timeLabel.setColour(juce::Label::textColourId, textColor.withAlpha(0.7f));
    timeLabel.setText("0:00 / 0:00", juce::dontSendNotification);
    addAndMakeVisible(timeLabel);
    
    startTimerHz(10);
}

AudioPlayerComponent::~AudioPlayerComponent()
{
    transportSource.removeChangeListener(this);
    stopTimer();
}

void AudioPlayerComponent::paint(juce::Graphics& g)
{
    g.setColour(bgColor);
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 8.0f);
}

void AudioPlayerComponent::resized()
{
    auto bounds = getLocalBounds().reduced(15);
    
    titleLabel.setBounds(bounds.removeFromTop(25));
    bounds.removeFromTop(10);
    
    // Mode buttons
    auto modeRow = bounds.removeFromTop(35);
    originalButton.setBounds(modeRow.removeFromLeft(90).reduced(2));
    processedButton.setBounds(modeRow.removeFromLeft(90).reduced(2));
    abButton.setBounds(modeRow.removeFromLeft(60).reduced(2));
    
    bounds.removeFromTop(10);
    
    // Transport controls
    auto transportRow = bounds.removeFromTop(35);
    playButton.setBounds(transportRow.removeFromLeft(70).reduced(2));
    stopButton.setBounds(transportRow.removeFromLeft(70).reduced(2));
    
    bounds.removeFromTop(10);
    
    // Position slider
    auto sliderRow = bounds.removeFromTop(25);
    timeLabel.setBounds(sliderRow.removeFromRight(100));
    positionSlider.setBounds(sliderRow);
}

void AudioPlayerComponent::changeListenerCallback(juce::ChangeBroadcaster*)
{
    if (transportSource.isPlaying())
        playButton.setButtonText("Pause");
    else
        playButton.setButtonText("Play");
}

void AudioPlayerComponent::setAudioUrls(const juce::String& original, const juce::String& processed)
{
    originalAudioUrl = original;
    processedAudioUrl = processed;
    
    // Load original by default
    currentMode = PlaybackMode::Original;
    updateButtonStates();
    loadUrl(originalAudioUrl);
}

void AudioPlayerComponent::loadAudioFiles(const juce::File& originalFile, const juce::File& processedFile)
{
    // Store file paths as URLs
    originalAudioUrl = originalFile.getFullPathName();
    processedAudioUrl = processedFile.getFullPathName();
    
    currentMode = PlaybackMode::Original;
    updateButtonStates();
    
    if (originalFile.existsAsFile())
    {
        auto* reader = formatManager.createReaderFor(originalFile);
        if (reader)
        {
            auto newSource = std::make_unique<juce::AudioFormatReaderSource>(reader, true);
            transportSource.setSource(newSource.get(), 0, nullptr, reader->sampleRate);
            readerSource = std::move(newSource);
        }
    }
}

void AudioPlayerComponent::updateButtonStates()
{
    auto activeColor = accentColor;
    auto inactiveColor = bgColor.brighter(0.2f);
    
    originalButton.setColour(juce::TextButton::buttonColourId, 
                             currentMode == PlaybackMode::Original ? activeColor : inactiveColor);
    originalButton.setColour(juce::TextButton::textColourOffId,
                             currentMode == PlaybackMode::Original ? juce::Colours::black : textColor);
    
    processedButton.setColour(juce::TextButton::buttonColourId,
                              currentMode == PlaybackMode::Processed ? activeColor : inactiveColor);
    processedButton.setColour(juce::TextButton::textColourOffId,
                              currentMode == PlaybackMode::Processed ? juce::Colours::black : textColor);
    
    abButton.setColour(juce::TextButton::buttonColourId,
                       currentMode == PlaybackMode::AB ? activeColor : inactiveColor);
    abButton.setColour(juce::TextButton::textColourOffId,
                       currentMode == PlaybackMode::AB ? juce::Colours::black : textColor);
}

void AudioPlayerComponent::loadUrl(const juce::String& url)
{
    // For now, we'll handle local files
    // HTTP streaming would require more complex implementation
    juce::File file(url);
    if (file.existsAsFile())
    {
        auto* reader = formatManager.createReaderFor(file);
        if (reader)
        {
            auto newSource = std::make_unique<juce::AudioFormatReaderSource>(reader, true);
            transportSource.setSource(newSource.get(), 0, nullptr, reader->sampleRate);
            readerSource = std::move(newSource);
        }
    }
}

juce::String AudioPlayerComponent::formatTime(double seconds)
{
    int mins = static_cast<int>(seconds) / 60;
    int secs = static_cast<int>(seconds) % 60;
    return juce::String::formatted("%d:%02d", mins, secs);
}

void AudioPlayerComponent::timerCallback()
{
    if (transportSource.getLengthInSeconds() > 0)
    {
        auto pos = transportSource.getCurrentPosition();
        auto length = transportSource.getLengthInSeconds();
        
        positionSlider.setValue(pos / length, juce::dontSendNotification);
        timeLabel.setText(formatTime(pos) + " / " + formatTime(length), juce::dontSendNotification);
    }
}

