#pragma once

#include <JuceHeader.h>

class ReferenceToRigProcessor : public juce::AudioProcessor
{
public:
    ReferenceToRigProcessor();
    ~ReferenceToRigProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // A/B comparison
    void setBypassEnabled(bool bypass) { bypassEnabled = bypass; }
    bool isBypassEnabled() const { return bypassEnabled; }

    // Audio file playback for preview
    void loadAudioFile(const juce::File& file);
    void setPlaybackEnabled(bool enabled) { playbackEnabled = enabled; }
    bool isPlaybackEnabled() const { return playbackEnabled; }

private:
    bool bypassEnabled = false;
    bool playbackEnabled = false;
    
    // For audio file preview
    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::AudioFormatReaderSource> readerSource;
    juce::AudioTransportSource transportSource;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReferenceToRigProcessor)
};

