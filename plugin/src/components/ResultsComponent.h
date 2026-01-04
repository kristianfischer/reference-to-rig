#pragma once

#include <JuceHeader.h>
#include "../api/EngineClient.h"

class ResultsComponent : public juce::Component
{
public:
    ResultsComponent();
    ~ResultsComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    void setResults(const MatchingResults& results);
    
    // Callbacks
    std::function<void(int matchIndex)> onMatchSelected;
    std::function<void(const juce::String& namId)> onDownloadNam;

private:
    MatchingResults currentResults;
    int selectedMatchIndex = 0;
    
    juce::Label titleLabel;
    juce::Label namLabel;
    juce::Label irLabel;
    juce::Label similarityLabel;
    juce::Label gainLabel;
    
    juce::TextButton downloadButton;
    juce::TextButton prevButton;
    juce::TextButton nextButton;
    juce::Label matchIndexLabel;
    
    // Colors
    juce::Colour bgColor{ 0xff16213e };
    juce::Colour accentColor{ 0xff00d4ff };
    juce::Colour textColor{ 0xffeaeaea };
    juce::Colour successColor{ 0xff00ff88 };
    
    void updateDisplay();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ResultsComponent)
};

