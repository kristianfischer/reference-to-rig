#include "ProgressComponent.h"

ProgressComponent::ProgressComponent()
{
    statusLabel.setFont(juce::Font(18.0f));
    statusLabel.setColour(juce::Label::textColourId, textColor);
    statusLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(statusLabel);
    
    percentLabel.setFont(juce::Font(32.0f, juce::Font::bold));
    percentLabel.setColour(juce::Label::textColourId, accentColor);
    percentLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(percentLabel);
    
    startTimerHz(30);
}

ProgressComponent::~ProgressComponent()
{
    stopTimer();
}

void ProgressComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat();
    
    // Background
    g.setColour(bgColor);
    g.fillRoundedRectangle(bounds, 8.0f);
    
    // Progress bar area
    auto barArea = bounds.reduced(40).withHeight(12.0f).withCentre({ bounds.getCentreX(), bounds.getCentreY() + 50 });
    
    // Track
    g.setColour(trackColor);
    g.fillRoundedRectangle(barArea, 6.0f);
    
    if (isIndeterminate)
    {
        // Animated indeterminate bar
        float barWidth = barArea.getWidth() * 0.3f;
        float x = barArea.getX() + (barArea.getWidth() - barWidth) * (0.5f + 0.5f * std::sin(animationPhase));
        
        g.setColour(accentColor);
        g.fillRoundedRectangle(x, barArea.getY(), barWidth, barArea.getHeight(), 6.0f);
    }
    else
    {
        // Determinate progress bar
        float fillWidth = barArea.getWidth() * progressValue;
        if (fillWidth > 0)
        {
            // Gradient fill
            g.setGradientFill(juce::ColourGradient(
                accentColor.brighter(0.2f), barArea.getX(), 0,
                accentColor, barArea.getX() + fillWidth, 0, false
            ));
            g.fillRoundedRectangle(barArea.getX(), barArea.getY(), fillWidth, barArea.getHeight(), 6.0f);
            
            // Glow effect
            g.setColour(accentColor.withAlpha(0.3f));
            g.fillRoundedRectangle(barArea.getX(), barArea.getY() - 2, fillWidth, barArea.getHeight() + 4, 8.0f);
        }
    }
    
    // Animated dots
    juce::String dots;
    int numDots = static_cast<int>(animationPhase * 2) % 4;
    for (int i = 0; i < numDots; ++i) dots += ".";
}

void ProgressComponent::resized()
{
    auto bounds = getLocalBounds();
    auto center = bounds.getCentre();
    
    percentLabel.setBounds(bounds.withHeight(50).withCentre({ center.x, center.y - 30 }));
    statusLabel.setBounds(bounds.withHeight(30).withCentre({ center.x, center.y + 90 }));
}

void ProgressComponent::timerCallback()
{
    animationPhase += 0.05f;
    if (animationPhase > juce::MathConstants<float>::twoPi)
        animationPhase -= juce::MathConstants<float>::twoPi;
    
    repaint();
}

void ProgressComponent::setStatus(const juce::String& message, float progress)
{
    statusMessage = message;
    progressValue = juce::jlimit(0.0f, 1.0f, progress);
    isIndeterminate = false;
    
    statusLabel.setText(statusMessage, juce::dontSendNotification);
    percentLabel.setText(juce::String(static_cast<int>(progressValue * 100)) + "%", juce::dontSendNotification);
}

void ProgressComponent::setIndeterminate(bool indeterminate)
{
    isIndeterminate = indeterminate;
    if (indeterminate)
    {
        percentLabel.setText("", juce::dontSendNotification);
    }
}

