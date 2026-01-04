#pragma once

#include <JuceHeader.h>

class ProgressComponent : public juce::Component,
                          public juce::Timer
{
public:
    ProgressComponent();
    ~ProgressComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

    void setStatus(const juce::String& message, float progress);
    void setIndeterminate(bool indeterminate);

private:
    juce::String statusMessage = "Processing...";
    float progressValue = 0.0f;
    bool isIndeterminate = false;
    float animationPhase = 0.0f;
    
    juce::Label statusLabel;
    juce::Label percentLabel;
    
    // Colors
    juce::Colour bgColor{ 0xff16213e };
    juce::Colour accentColor{ 0xff00d4ff };
    juce::Colour textColor{ 0xffeaeaea };
    juce::Colour trackColor{ 0xff0a0a1a };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ProgressComponent)
};

