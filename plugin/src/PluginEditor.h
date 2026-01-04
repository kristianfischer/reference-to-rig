#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "api/EngineClient.h"
#include "components/FileDropComponent.h"
#include "components/ProjectListComponent.h"
#include "components/ResultsComponent.h"
#include "components/ProgressComponent.h"
#include "components/AudioPlayerComponent.h"
#include "components/EQVisualizerComponent.h"

class ReferenceToRigEditor : public juce::AudioProcessorEditor,
                              public juce::Timer
{
public:
    explicit ReferenceToRigEditor(ReferenceToRigProcessor&);
    ~ReferenceToRigEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    ReferenceToRigProcessor& processorRef;
    
    // Engine client for API communication
    std::unique_ptr<EngineClient> engineClient;
    
    // UI State
    enum class ViewState { Projects, Upload, Processing, Results };
    ViewState currentView = ViewState::Projects;
    juce::String currentProjectId;
    
    // Header
    juce::Label titleLabel;
    juce::TextButton backButton;
    juce::Label connectionStatus;
    
    // Main components
    std::unique_ptr<FileDropComponent> fileDropComponent;
    std::unique_ptr<ProjectListComponent> projectListComponent;
    std::unique_ptr<ResultsComponent> resultsComponent;
    std::unique_ptr<ProgressComponent> progressComponent;
    std::unique_ptr<AudioPlayerComponent> audioPlayerComponent;
    std::unique_ptr<EQVisualizerComponent> eqVisualizerComponent;
    
    // Buttons
    juce::TextButton newProjectButton;
    juce::TextButton refreshButton;
    
    // Methods
    void setupUI();
    void showProjectsView();
    void showUploadView();
    void showProcessingView();
    void showResultsView();
    void checkEngineConnection();
    void refreshProjects();
    void onFileDropped(const juce::File& file);
    void onProjectSelected(const juce::String& projectId);
    void pollTaskStatus();
    
    // Colors
    juce::Colour bgColor{ 0xff1a1a2e };
    juce::Colour accentColor{ 0xff00d4ff };
    juce::Colour surfaceColor{ 0xff16213e };
    juce::Colour textColor{ 0xffeaeaea };
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReferenceToRigEditor)
};

