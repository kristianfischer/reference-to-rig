#include "PluginEditor.h"

ReferenceToRigEditor::ReferenceToRigEditor(ReferenceToRigProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    engineClient = std::make_unique<EngineClient>("http://127.0.0.1:8000");
    
    setupUI();
    setSize(900, 700);
    
    // Start connection check timer
    startTimer(2000);
    checkEngineConnection();
    refreshProjects();
}

ReferenceToRigEditor::~ReferenceToRigEditor()
{
    stopTimer();
}

void ReferenceToRigEditor::setupUI()
{
    // Title
    titleLabel.setText("Reference to Rig", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(28.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, accentColor);
    addAndMakeVisible(titleLabel);
    
    // Back button
    backButton.setButtonText("<< Back");
    backButton.setColour(juce::TextButton::buttonColourId, surfaceColor);
    backButton.setColour(juce::TextButton::textColourOffId, textColor);
    backButton.onClick = [this]() { showProjectsView(); };
    addAndMakeVisible(backButton);
    backButton.setVisible(false);
    
    // Connection status
    connectionStatus.setText("Checking engine...", juce::dontSendNotification);
    connectionStatus.setFont(juce::Font(12.0f));
    connectionStatus.setColour(juce::Label::textColourId, juce::Colours::orange);
    connectionStatus.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(connectionStatus);
    
    // New Project button
    newProjectButton.setButtonText("+ New Project");
    newProjectButton.setColour(juce::TextButton::buttonColourId, accentColor);
    newProjectButton.setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    newProjectButton.onClick = [this]() { showUploadView(); };
    addAndMakeVisible(newProjectButton);
    
    // Refresh button
    refreshButton.setButtonText("Refresh");
    refreshButton.setColour(juce::TextButton::buttonColourId, surfaceColor);
    refreshButton.setColour(juce::TextButton::textColourOffId, textColor);
    refreshButton.onClick = [this]() { refreshProjects(); };
    addAndMakeVisible(refreshButton);
    
    // Create components
    fileDropComponent = std::make_unique<FileDropComponent>();
    fileDropComponent->onFileDropped = [this](const juce::File& f) { onFileDropped(f); };
    addAndMakeVisible(*fileDropComponent);
    fileDropComponent->setVisible(false);
    
    projectListComponent = std::make_unique<ProjectListComponent>();
    projectListComponent->onProjectSelected = [this](const juce::String& id) { onProjectSelected(id); };
    addAndMakeVisible(*projectListComponent);
    
    resultsComponent = std::make_unique<ResultsComponent>();
    addAndMakeVisible(*resultsComponent);
    resultsComponent->setVisible(false);
    
    progressComponent = std::make_unique<ProgressComponent>();
    addAndMakeVisible(*progressComponent);
    progressComponent->setVisible(false);
    
    audioPlayerComponent = std::make_unique<AudioPlayerComponent>();
    addAndMakeVisible(*audioPlayerComponent);
    audioPlayerComponent->setVisible(false);
    
    eqVisualizerComponent = std::make_unique<EQVisualizerComponent>();
    addAndMakeVisible(*eqVisualizerComponent);
    eqVisualizerComponent->setVisible(false);
}

void ReferenceToRigEditor::paint(juce::Graphics& g)
{
    // Background gradient
    g.setGradientFill(juce::ColourGradient(
        bgColor, 0, 0,
        bgColor.darker(0.3f), 0, static_cast<float>(getHeight()),
        false));
    g.fillAll();
    
    // Header bar
    g.setColour(surfaceColor);
    g.fillRect(0, 0, getWidth(), 70);
    
    // Accent line
    g.setColour(accentColor);
    g.fillRect(0, 70, getWidth(), 2);
}

void ReferenceToRigEditor::resized()
{
    auto area = getLocalBounds();
    
    // Header
    auto header = area.removeFromTop(70);
    backButton.setBounds(header.removeFromLeft(100).reduced(15, 20));
    titleLabel.setBounds(header.removeFromLeft(250).reduced(10, 15));
    connectionStatus.setBounds(header.removeFromRight(200).reduced(10, 25));
    
    // Accent line area
    area.removeFromTop(2);
    
    // Button bar
    auto buttonBar = area.removeFromTop(50);
    newProjectButton.setBounds(buttonBar.removeFromLeft(150).reduced(15, 10));
    refreshButton.setBounds(buttonBar.removeFromLeft(100).reduced(10, 10));
    
    // Main content area
    auto content = area.reduced(20);
    
    projectListComponent->setBounds(content);
    fileDropComponent->setBounds(content);
    progressComponent->setBounds(content);
    
    // Results view - split layout
    if (currentView == ViewState::Results)
    {
        auto leftPanel = content.removeFromLeft(content.getWidth() / 2);
        resultsComponent->setBounds(leftPanel.withTrimmedRight(10));
        
        auto rightPanel = content;
        auto playerArea = rightPanel.removeFromTop(200);
        audioPlayerComponent->setBounds(playerArea.withTrimmedLeft(10).withTrimmedBottom(10));
        eqVisualizerComponent->setBounds(rightPanel.withTrimmedLeft(10).withTrimmedTop(10));
    }
    else
    {
        resultsComponent->setBounds(content);
    }
}

void ReferenceToRigEditor::timerCallback()
{
    checkEngineConnection();
    
    if (currentView == ViewState::Processing)
    {
        pollTaskStatus();
    }
}

void ReferenceToRigEditor::checkEngineConnection()
{
    engineClient->checkHealth([this](bool connected) {
        juce::MessageManager::callAsync([this, connected]() {
            if (connected)
            {
                connectionStatus.setText("Engine Connected", juce::dontSendNotification);
                connectionStatus.setColour(juce::Label::textColourId, juce::Colours::limegreen);
            }
            else
            {
                connectionStatus.setText("Engine Offline", juce::dontSendNotification);
                connectionStatus.setColour(juce::Label::textColourId, juce::Colours::red);
            }
        });
    });
}

void ReferenceToRigEditor::refreshProjects()
{
    engineClient->getProjects([this](const juce::Array<RtrProject>& projects) {
        juce::MessageManager::callAsync([this, projects]() {
            projectListComponent->setProjects(projects);
        });
    });
}

void ReferenceToRigEditor::showProjectsView()
{
    currentView = ViewState::Projects;
    backButton.setVisible(false);
    newProjectButton.setVisible(true);
    refreshButton.setVisible(true);
    
    projectListComponent->setVisible(true);
    fileDropComponent->setVisible(false);
    progressComponent->setVisible(false);
    resultsComponent->setVisible(false);
    audioPlayerComponent->setVisible(false);
    eqVisualizerComponent->setVisible(false);
    
    refreshProjects();
    resized();
}

void ReferenceToRigEditor::showUploadView()
{
    currentView = ViewState::Upload;
    backButton.setVisible(true);
    newProjectButton.setVisible(false);
    refreshButton.setVisible(false);
    
    projectListComponent->setVisible(false);
    fileDropComponent->setVisible(true);
    progressComponent->setVisible(false);
    resultsComponent->setVisible(false);
    audioPlayerComponent->setVisible(false);
    eqVisualizerComponent->setVisible(false);
    
    resized();
}

void ReferenceToRigEditor::showProcessingView()
{
    currentView = ViewState::Processing;
    backButton.setVisible(true);
    newProjectButton.setVisible(false);
    refreshButton.setVisible(false);
    
    projectListComponent->setVisible(false);
    fileDropComponent->setVisible(false);
    progressComponent->setVisible(true);
    resultsComponent->setVisible(false);
    audioPlayerComponent->setVisible(false);
    eqVisualizerComponent->setVisible(false);
    
    resized();
}

void ReferenceToRigEditor::showResultsView()
{
    currentView = ViewState::Results;
    backButton.setVisible(true);
    newProjectButton.setVisible(false);
    refreshButton.setVisible(false);
    
    projectListComponent->setVisible(false);
    fileDropComponent->setVisible(false);
    progressComponent->setVisible(false);
    resultsComponent->setVisible(true);
    audioPlayerComponent->setVisible(true);
    eqVisualizerComponent->setVisible(true);
    
    resized();
}

void ReferenceToRigEditor::onFileDropped(const juce::File& file)
{
    progressComponent->setStatus("Uploading...", 0.1f);
    showProcessingView();
    
    engineClient->uploadAudio(file, [this](const juce::String& projectId, bool success) {
        juce::MessageManager::callAsync([this, projectId, success]() {
            if (success)
            {
                currentProjectId = projectId;
                progressComponent->setStatus("Starting isolation...", 0.2f);
                
                // Start isolation task
                engineClient->startIsolation(projectId, "guitar", [this](bool ok) {
                    juce::MessageManager::callAsync([this, ok]() {
                        if (ok)
                        {
                            progressComponent->setStatus("Isolating guitar...", 0.3f);
                        }
                        else
                        {
                            progressComponent->setStatus("Isolation failed!", 0.0f);
                        }
                    });
                });
            }
            else
            {
                progressComponent->setStatus("Upload failed!", 0.0f);
            }
        });
    });
}

void ReferenceToRigEditor::onProjectSelected(const juce::String& projectId)
{
    currentProjectId = projectId;
    
    // Fetch project details and show results
    engineClient->getProjectResults(projectId, [this](const MatchingResults& results, bool success) {
        juce::MessageManager::callAsync([this, results, success]() {
            if (success && results.matches.size() > 0)
            {
                resultsComponent->setResults(results);
                
                // Load audio URLs for player
                audioPlayerComponent->setAudioUrls(
                    results.originalAudioUrl,
                    results.processedAudioUrl
                );
                
                // Set EQ curve if available
                if (results.matches.size() > 0)
                {
                    eqVisualizerComponent->setEQSettings(results.matches[0].eqSettings);
                }
                
                showResultsView();
            }
            else
            {
                // Project exists but no results yet - show processing
                showProcessingView();
                progressComponent->setStatus("Processing...", 0.5f);
            }
        });
    });
}

void ReferenceToRigEditor::pollTaskStatus()
{
    if (currentProjectId.isEmpty())
        return;
    
    engineClient->getTaskStatus(currentProjectId, [this](const TaskStatus& status) {
        juce::MessageManager::callAsync([this, status]() {
            progressComponent->setStatus(status.message, status.progress);
            
            if (status.state == "completed")
            {
                // Fetch results and show
                onProjectSelected(currentProjectId);
            }
            else if (status.state == "failed")
            {
                progressComponent->setStatus("Processing failed: " + status.message, 0.0f);
            }
        });
    });
}

