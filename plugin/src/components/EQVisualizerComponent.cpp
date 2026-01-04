#include "EQVisualizerComponent.h"

EQVisualizerComponent::EQVisualizerComponent()
{
    titleLabel.setText("Suggested EQ", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(16.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, curveColor);
    addAndMakeVisible(titleLabel);
}

void EQVisualizerComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat();
    
    // Background
    g.setColour(bgColor);
    g.fillRoundedRectangle(bounds, 8.0f);
    
    // Graph area
    auto graphBounds = bounds.reduced(15).withTrimmedTop(35);
    
    drawGrid(g, graphBounds);
    drawCurve(g, graphBounds);
    drawBandMarkers(g, graphBounds);
}

void EQVisualizerComponent::resized()
{
    auto bounds = getLocalBounds().reduced(15);
    titleLabel.setBounds(bounds.removeFromTop(30));
}

void EQVisualizerComponent::setEQSettings(const juce::Array<EQBand>& bands)
{
    eqBands = bands;
    repaint();
}

float EQVisualizerComponent::freqToX(float freq, float width) const
{
    float logMin = std::log10(minFreq);
    float logMax = std::log10(maxFreq);
    float logFreq = std::log10(juce::jlimit(minFreq, maxFreq, freq));
    return width * (logFreq - logMin) / (logMax - logMin);
}

float EQVisualizerComponent::dbToY(float db, float height) const
{
    float clampedDb = juce::jlimit(minDb, maxDb, db);
    return height * (1.0f - (clampedDb - minDb) / (maxDb - minDb));
}

float EQVisualizerComponent::calculateEQResponse(float freq) const
{
    float totalGain = 0.0f;
    
    for (const auto& band : eqBands)
    {
        float freqRatio = freq / band.frequency;
        float octaves = std::log2(freqRatio);
        float bandwidth = 1.0f / band.q;
        
        // Simple bell curve approximation
        float response = band.gain * std::exp(-0.5f * std::pow(octaves / (bandwidth * 0.5f), 2.0f));
        
        // Shelving filters
        if (band.type == "lowshelf")
        {
            if (freq <= band.frequency)
                response = band.gain;
            else
                response = band.gain * std::exp(-2.0f * octaves);
        }
        else if (band.type == "highshelf")
        {
            if (freq >= band.frequency)
                response = band.gain;
            else
                response = band.gain * std::exp(2.0f * octaves);
        }
        
        totalGain += response;
    }
    
    return juce::jlimit(minDb, maxDb, totalGain);
}

void EQVisualizerComponent::drawGrid(juce::Graphics& g, juce::Rectangle<float> bounds)
{
    g.setColour(gridColor);
    
    // Frequency grid lines
    float frequencies[] = { 50, 100, 200, 500, 1000, 2000, 5000, 10000 };
    for (float freq : frequencies)
    {
        float x = bounds.getX() + freqToX(freq, bounds.getWidth());
        g.drawVerticalLine(static_cast<int>(x), bounds.getY(), bounds.getBottom());
        
        // Labels
        g.setColour(textColor.withAlpha(0.4f));
        g.setFont(10.0f);
        juce::String label = freq >= 1000 ? juce::String(static_cast<int>(freq / 1000)) + "k" : juce::String(static_cast<int>(freq));
        g.drawText(label, static_cast<int>(x) - 15, static_cast<int>(bounds.getBottom()) + 2, 30, 15, juce::Justification::centred);
        g.setColour(gridColor);
    }
    
    // dB grid lines
    float dbLevels[] = { -12, -6, 0, 6, 12 };
    for (float db : dbLevels)
    {
        float y = bounds.getY() + dbToY(db, bounds.getHeight());
        g.drawHorizontalLine(static_cast<int>(y), bounds.getX(), bounds.getRight());
        
        // Labels
        g.setColour(textColor.withAlpha(0.4f));
        g.setFont(10.0f);
        juce::String label = db > 0 ? "+" + juce::String(static_cast<int>(db)) : juce::String(static_cast<int>(db));
        g.drawText(label, static_cast<int>(bounds.getX()) - 30, static_cast<int>(y) - 7, 25, 15, juce::Justification::centredRight);
        g.setColour(gridColor);
    }
    
    // 0 dB line (brighter)
    float zeroY = bounds.getY() + dbToY(0, bounds.getHeight());
    g.setColour(gridColor.brighter(0.3f));
    g.drawHorizontalLine(static_cast<int>(zeroY), bounds.getX(), bounds.getRight());
}

void EQVisualizerComponent::drawCurve(juce::Graphics& g, juce::Rectangle<float> bounds)
{
    if (eqBands.isEmpty())
    {
        // Draw flat line at 0 dB
        g.setColour(curveColor.withAlpha(0.5f));
        float y = bounds.getY() + dbToY(0, bounds.getHeight());
        g.drawHorizontalLine(static_cast<int>(y), bounds.getX(), bounds.getRight());
        return;
    }
    
    // Build the curve path
    juce::Path curvePath;
    bool started = false;
    
    for (float x = 0; x < bounds.getWidth(); x += 2.0f)
    {
        float logMin = std::log10(minFreq);
        float logMax = std::log10(maxFreq);
        float freq = std::pow(10.0f, logMin + (x / bounds.getWidth()) * (logMax - logMin));
        
        float db = calculateEQResponse(freq);
        float y = dbToY(db, bounds.getHeight());
        
        if (!started)
        {
            curvePath.startNewSubPath(bounds.getX() + x, bounds.getY() + y);
            started = true;
        }
        else
        {
            curvePath.lineTo(bounds.getX() + x, bounds.getY() + y);
        }
    }
    
    // Draw fill
    juce::Path fillPath = curvePath;
    fillPath.lineTo(bounds.getRight(), bounds.getY() + dbToY(0, bounds.getHeight()));
    fillPath.lineTo(bounds.getX(), bounds.getY() + dbToY(0, bounds.getHeight()));
    fillPath.closeSubPath();
    
    g.setColour(curveColor.withAlpha(0.15f));
    g.fillPath(fillPath);
    
    // Draw curve
    g.setColour(curveColor);
    g.strokePath(curvePath, juce::PathStrokeType(2.5f));
}

void EQVisualizerComponent::drawBandMarkers(juce::Graphics& g, juce::Rectangle<float> bounds)
{
    for (const auto& band : eqBands)
    {
        float x = bounds.getX() + freqToX(band.frequency, bounds.getWidth());
        float y = bounds.getY() + dbToY(band.gain, bounds.getHeight());
        
        // Draw marker
        g.setColour(curveColor);
        g.fillEllipse(x - 5, y - 5, 10, 10);
        g.setColour(bgColor);
        g.fillEllipse(x - 3, y - 3, 6, 6);
    }
}

