#pragma once

#include <JuceHeader.h>
#include "../api/EngineClient.h"

class ProjectListComponent : public juce::Component,
                              public juce::ListBoxModel
{
public:
    ProjectListComponent();
    ~ProjectListComponent() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // ListBoxModel
    int getNumRows() override;
    void paintListBoxItem(int rowNumber, juce::Graphics& g, int width, int height, bool rowIsSelected) override;
    void listBoxItemClicked(int row, const juce::MouseEvent&) override;
    void listBoxItemDoubleClicked(int row, const juce::MouseEvent&) override;

    // Data
    void setProjects(const juce::Array<RtrProject>& projects);
    
    // Callback
    std::function<void(const juce::String& projectId)> onProjectSelected;

private:
    juce::ListBox listBox;
    juce::Array<RtrProject> projectList;
    juce::Label emptyLabel;
    
    // Colors
    juce::Colour bgColor{ 0xff16213e };
    juce::Colour selectedColor{ 0xff00d4ff };
    juce::Colour textColor{ 0xffeaeaea };
    juce::Colour subtextColor{ 0xff888888 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ProjectListComponent)
};

