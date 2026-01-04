#pragma once

#include <JuceHeader.h>

// Data structures for API responses
struct RtrProject
{
    juce::String id;
    juce::String name;
    juce::String status;
    juce::String createdAt;
};

struct EQBand
{
    float frequency;
    float gain;
    float q;
    juce::String type; // "lowshelf", "highshelf", "peak"
};

struct MatchResult
{
    juce::String namModelId;
    juce::String namModelName;
    juce::String irId;
    juce::String irName;
    float similarity;
    float gain;
    juce::Array<EQBand> eqSettings;
};

struct MatchingResults
{
    juce::Array<MatchResult> matches;
    juce::String originalAudioUrl;
    juce::String processedAudioUrl;
    juce::String isolatedAudioUrl;
};

struct TaskStatus
{
    juce::String state; // "pending", "running", "completed", "failed"
    juce::String message;
    float progress; // 0.0 to 1.0
};

class EngineClient
{
public:
    explicit EngineClient(const juce::String& baseUrl);
    ~EngineClient();
    
    // Health check
    void checkHealth(std::function<void(bool connected)> callback);
    
    // Projects
    void getProjects(std::function<void(const juce::Array<RtrProject>&)> callback);
    void createProject(const juce::String& name, std::function<void(const juce::String& projectId, bool success)> callback);
    void deleteProject(const juce::String& projectId, std::function<void(bool success)> callback);
    
    // Audio upload
    void uploadAudio(const juce::File& audioFile, std::function<void(const juce::String& projectId, bool success)> callback);
    
    // Tasks
    void startIsolation(const juce::String& projectId, const juce::String& prompt, 
                        std::function<void(bool success)> callback);
    void startMatching(const juce::String& projectId, std::function<void(bool success)> callback);
    void getTaskStatus(const juce::String& projectId, std::function<void(const TaskStatus&)> callback);
    
    // Results
    void getProjectResults(const juce::String& projectId, 
                           std::function<void(const MatchingResults&, bool success)> callback);
    
    // Audio streaming
    juce::URL getAudioUrl(const juce::String& projectId, const juce::String& audioType);
    
private:
    juce::String baseUrl;
    
    // Helper methods
    void makeGetRequest(const juce::String& endpoint, 
                        std::function<void(const juce::var&, bool success)> callback);
    void makePostRequest(const juce::String& endpoint, const juce::var& body,
                         std::function<void(const juce::var&, bool success)> callback);
    void makeMultipartRequest(const juce::String& endpoint, const juce::File& file,
                              std::function<void(const juce::var&, bool success)> callback);
    
    static RtrProject parseProject(const juce::var& json);
    static MatchResult parseMatchResult(const juce::var& json);
    static TaskStatus parseTaskStatus(const juce::var& json);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EngineClient)
};

