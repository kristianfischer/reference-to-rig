#include "FileDropComponent.h"

FileDropComponent::FileDropComponent()
{
    instructionLabel.setText("Drop your reference audio here", juce::dontSendNotification);
    instructionLabel.setFont(juce::Font(24.0f, juce::Font::bold));
    instructionLabel.setColour(juce::Label::textColourId, textColor);
    instructionLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(instructionLabel);
    
    formatLabel.setText("Supports WAV, MP3, FLAC, M4A", juce::dontSendNotification);
    formatLabel.setFont(juce::Font(14.0f));
    formatLabel.setColour(juce::Label::textColourId, textColor.withAlpha(0.6f));
    formatLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(formatLabel);
    
    browseButton.setButtonText("Browse Files");
    browseButton.setColour(juce::TextButton::buttonColourId, borderColor);
    browseButton.setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    browseButton.onClick = [this]() { browseForFile(); };
    addAndMakeVisible(browseButton);
}

void FileDropComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(2.0f);
    
    // Background
    g.setColour(isDragging ? bgColor.brighter(0.1f) : bgColor);
    g.fillRoundedRectangle(bounds, 12.0f);
    
    // Dashed border
    juce::Path borderPath;
    borderPath.addRoundedRectangle(bounds, 12.0f);
    
    float dashLengths[] = { 8.0f, 4.0f };
    juce::PathStrokeType stroke(2.0f);
    stroke.createDashedStroke(borderPath, borderPath, dashLengths, 2);
    
    g.setColour(isDragging ? borderColor : borderColor.withAlpha(0.5f));
    g.strokePath(borderPath, stroke);
    
    // Drop icon
    auto iconBounds = getLocalBounds().reduced(getWidth() / 4, getHeight() / 4).toFloat();
    iconBounds = iconBounds.removeFromTop(80).withSizeKeepingCentre(80, 80);
    
    g.setColour(borderColor.withAlpha(0.3f));
    g.fillEllipse(iconBounds);
    
    // Arrow
    g.setColour(borderColor);
    auto arrowBounds = iconBounds.reduced(20);
    juce::Path arrow;
    arrow.addArrow(juce::Line<float>(
        arrowBounds.getCentreX(), arrowBounds.getBottom() - 10,
        arrowBounds.getCentreX(), arrowBounds.getY() + 10
    ), 3.0f, 15.0f, 10.0f);
    g.fillPath(arrow);
}

void FileDropComponent::resized()
{
    auto bounds = getLocalBounds();
    auto center = bounds.getCentre();
    
    instructionLabel.setBounds(bounds.withHeight(40).withCentre({ center.x, center.y + 20 }));
    formatLabel.setBounds(bounds.withHeight(30).withCentre({ center.x, center.y + 55 }));
    browseButton.setBounds(bounds.withSize(150, 40).withCentre({ center.x, center.y + 100 }));
}

bool FileDropComponent::isInterestedInFileDrag(const juce::StringArray& files)
{
    for (const auto& file : files)
    {
        juce::File f(file);
        auto ext = f.getFileExtension().toLowerCase();
        if (ext == ".wav" || ext == ".mp3" || ext == ".flac" || 
            ext == ".m4a" || ext == ".aiff" || ext == ".ogg")
            return true;
    }
    return false;
}

void FileDropComponent::filesDropped(const juce::StringArray& files, int, int)
{
    isDragging = false;
    repaint();
    
    if (files.size() > 0 && onFileDropped)
    {
        onFileDropped(juce::File(files[0]));
    }
}

void FileDropComponent::fileDragEnter(const juce::StringArray&, int, int)
{
    isDragging = true;
    repaint();
}

void FileDropComponent::fileDragExit(const juce::StringArray&)
{
    isDragging = false;
    repaint();
}

void FileDropComponent::browseForFile()
{
    auto chooser = std::make_shared<juce::FileChooser>(
        "Select Reference Audio",
        juce::File::getSpecialLocation(juce::File::userMusicDirectory),
        "*.wav;*.mp3;*.flac;*.m4a;*.aiff;*.ogg"
    );
    
    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc) {
            auto result = fc.getResult();
            if (result.existsAsFile() && onFileDropped)
            {
                onFileDropped(result);
            }
        });
}

