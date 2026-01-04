#include "ResultsComponent.h"

ResultsComponent::ResultsComponent()
{
    titleLabel.setText("Best Match", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(20.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, accentColor);
    addAndMakeVisible(titleLabel);
    
    namLabel.setFont(juce::Font(16.0f, juce::Font::bold));
    namLabel.setColour(juce::Label::textColourId, textColor);
    addAndMakeVisible(namLabel);
    
    irLabel.setFont(juce::Font(14.0f));
    irLabel.setColour(juce::Label::textColourId, textColor.withAlpha(0.8f));
    addAndMakeVisible(irLabel);
    
    similarityLabel.setFont(juce::Font(24.0f, juce::Font::bold));
    similarityLabel.setColour(juce::Label::textColourId, successColor);
    similarityLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(similarityLabel);
    
    gainLabel.setFont(juce::Font(14.0f));
    gainLabel.setColour(juce::Label::textColourId, textColor);
    addAndMakeVisible(gainLabel);
    
    downloadButton.setButtonText("Download NAM");
    downloadButton.setColour(juce::TextButton::buttonColourId, successColor);
    downloadButton.setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    downloadButton.onClick = [this]() {
        if (selectedMatchIndex < currentResults.matches.size() && onDownloadNam)
        {
            onDownloadNam(currentResults.matches[selectedMatchIndex].namModelId);
        }
    };
    addAndMakeVisible(downloadButton);
    
    prevButton.setButtonText("<");
    prevButton.setColour(juce::TextButton::buttonColourId, bgColor.brighter(0.2f));
    prevButton.onClick = [this]() {
        if (selectedMatchIndex > 0)
        {
            selectedMatchIndex--;
            updateDisplay();
            if (onMatchSelected) onMatchSelected(selectedMatchIndex);
        }
    };
    addAndMakeVisible(prevButton);
    
    nextButton.setButtonText(">");
    nextButton.setColour(juce::TextButton::buttonColourId, bgColor.brighter(0.2f));
    nextButton.onClick = [this]() {
        if (selectedMatchIndex < currentResults.matches.size() - 1)
        {
            selectedMatchIndex++;
            updateDisplay();
            if (onMatchSelected) onMatchSelected(selectedMatchIndex);
        }
    };
    addAndMakeVisible(nextButton);
    
    matchIndexLabel.setFont(juce::Font(12.0f));
    matchIndexLabel.setColour(juce::Label::textColourId, textColor.withAlpha(0.6f));
    matchIndexLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(matchIndexLabel);
}

void ResultsComponent::paint(juce::Graphics& g)
{
    g.setColour(bgColor);
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 8.0f);
    
    // Similarity circle background
    auto circleArea = getLocalBounds().removeFromRight(150).reduced(20);
    g.setColour(successColor.withAlpha(0.1f));
    g.fillEllipse(circleArea.toFloat());
    g.setColour(successColor);
    g.drawEllipse(circleArea.toFloat(), 3.0f);
}

void ResultsComponent::resized()
{
    auto bounds = getLocalBounds().reduced(20);
    
    // Right side - similarity circle
    auto rightPanel = bounds.removeFromRight(150);
    auto circleArea = rightPanel.reduced(10);
    similarityLabel.setBounds(circleArea.withSizeKeepingCentre(100, 40));
    
    // Navigation
    auto navArea = bounds.removeFromBottom(40);
    prevButton.setBounds(navArea.removeFromLeft(40));
    nextButton.setBounds(navArea.removeFromRight(40));
    matchIndexLabel.setBounds(navArea);
    
    // Title
    titleLabel.setBounds(bounds.removeFromTop(30));
    bounds.removeFromTop(10);
    
    // NAM info
    namLabel.setBounds(bounds.removeFromTop(25));
    irLabel.setBounds(bounds.removeFromTop(22));
    bounds.removeFromTop(10);
    gainLabel.setBounds(bounds.removeFromTop(22));
    
    bounds.removeFromTop(20);
    downloadButton.setBounds(bounds.removeFromTop(40).withWidth(160));
}

void ResultsComponent::setResults(const MatchingResults& results)
{
    currentResults = results;
    selectedMatchIndex = 0;
    updateDisplay();
}

void ResultsComponent::updateDisplay()
{
    if (currentResults.matches.isEmpty())
    {
        namLabel.setText("No matches found", juce::dontSendNotification);
        irLabel.setText("", juce::dontSendNotification);
        similarityLabel.setText("--", juce::dontSendNotification);
        gainLabel.setText("", juce::dontSendNotification);
        matchIndexLabel.setText("", juce::dontSendNotification);
        return;
    }
    
    auto& match = currentResults.matches.getReference(selectedMatchIndex);
    
    namLabel.setText("NAM: " + match.namModelName, juce::dontSendNotification);
    irLabel.setText("IR: " + (match.irName.isEmpty() ? "None" : match.irName), juce::dontSendNotification);
    
    int similarityPercent = static_cast<int>(match.similarity * 100);
    similarityLabel.setText(juce::String(similarityPercent) + "%", juce::dontSendNotification);
    
    juce::String gainStr = match.gain >= 0 ? "+" : "";
    gainStr += juce::String(match.gain, 1) + " dB";
    gainLabel.setText("Gain: " + gainStr, juce::dontSendNotification);
    
    matchIndexLabel.setText(juce::String(selectedMatchIndex + 1) + " / " + 
                            juce::String(currentResults.matches.size()), 
                            juce::dontSendNotification);
    
    prevButton.setEnabled(selectedMatchIndex > 0);
    nextButton.setEnabled(selectedMatchIndex < currentResults.matches.size() - 1);
}

