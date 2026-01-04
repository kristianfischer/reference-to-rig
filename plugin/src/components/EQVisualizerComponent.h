#pragma once

#include <JuceHeader.h>
#include "../api/EngineClient.h"

class EQVisualizerComponent : public juce::Component
{
public:
    EQVisualizerComponent();
    ~EQVisualizerComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    void setEQSettings(const juce::Array<EQBand>& bands);

private:
    juce::Array<EQBand> eqBands;
    juce::Label titleLabel;
    
    // Colors
    juce::Colour bgColor{ 0xff16213e };
    juce::Colour gridColor{ 0xff2a2a4a };
    juce::Colour curveColor{ 0xff00d4ff };
    juce::Colour textColor{ 0xffeaeaea };
    
    // Frequency display range
    static constexpr float minFreq = 20.0f;
    static constexpr float maxFreq = 20000.0f;
    static constexpr float minDb = -18.0f;
    static constexpr float maxDb = 18.0f;
    
    float freqToX(float freq, float width) const;
    float dbToY(float db, float height) const;
    float calculateEQResponse(float freq) const;
    void drawGrid(juce::Graphics& g, juce::Rectangle<float> bounds);
    void drawCurve(juce::Graphics& g, juce::Rectangle<float> bounds);
    void drawBandMarkers(juce::Graphics& g, juce::Rectangle<float> bounds);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EQVisualizerComponent)
};

