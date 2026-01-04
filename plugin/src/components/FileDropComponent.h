#pragma once

#include <JuceHeader.h>

class FileDropComponent : public juce::Component,
                          public juce::FileDragAndDropTarget
{
public:
    FileDropComponent();
    ~FileDropComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // FileDragAndDropTarget
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;

    // Callback
    std::function<void(const juce::File&)> onFileDropped;

private:
    bool isDragging = false;
    juce::TextButton browseButton;
    juce::Label instructionLabel;
    juce::Label formatLabel;
    
    // Colors
    juce::Colour bgColor{ 0xff16213e };
    juce::Colour borderColor{ 0xff00d4ff };
    juce::Colour textColor{ 0xffeaeaea };
    
    void browseForFile();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(FileDropComponent)
};

