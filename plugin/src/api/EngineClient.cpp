#include "EngineClient.h"

EngineClient::EngineClient(const juce::String& url)
    : baseUrl(url)
{
}

EngineClient::~EngineClient() = default;

void EngineClient::checkHealth(std::function<void(bool)> callback)
{
    makeGetRequest("/api/health", [callback](const juce::var& response, bool success) {
        callback(success && response.hasProperty("status") && 
                 response["status"].toString() == "ok");
    });
}

void EngineClient::getProjects(std::function<void(const juce::Array<RtrProject>&)> callback)
{
    makeGetRequest("/api/projects", [callback](const juce::var& response, bool success) {
        juce::Array<RtrProject> projects;
        
        if (success && response.isArray())
        {
            for (int i = 0; i < response.size(); ++i)
            {
                projects.add(parseProject(response[i]));
            }
        }
        
        callback(projects);
    });
}

void EngineClient::createProject(const juce::String& name, 
                                  std::function<void(const juce::String&, bool)> callback)
{
    juce::DynamicObject::Ptr body = new juce::DynamicObject();
    body->setProperty("name", name);
    
    makePostRequest("/api/projects", juce::var(body.get()), 
                    [callback](const juce::var& response, bool success) {
        juce::String projectId;
        if (success && response.hasProperty("id"))
        {
            projectId = response["id"].toString();
        }
        callback(projectId, success && projectId.isNotEmpty());
    });
}

void EngineClient::deleteProject(const juce::String& projectId, std::function<void(bool)> callback)
{
    auto url = juce::URL(baseUrl + "/api/projects/" + projectId);
    
    std::thread([url, callback]() {
        auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
            .withHttpRequestCmd("DELETE")
            .withConnectionTimeoutMs(5000);
        
        auto stream = url.createInputStream(options);
        bool success = stream != nullptr;
        callback(success);
    }).detach();
}

void EngineClient::uploadAudio(const juce::File& audioFile, 
                                std::function<void(const juce::String&, bool)> callback)
{
    makeMultipartRequest("/api/projects/upload", audioFile,
                         [callback](const juce::var& response, bool success) {
        juce::String projectId;
        if (success && response.hasProperty("project_id"))
        {
            projectId = response["project_id"].toString();
        }
        callback(projectId, success && projectId.isNotEmpty());
    });
}

void EngineClient::startIsolation(const juce::String& projectId, const juce::String& prompt,
                                   std::function<void(bool)> callback)
{
    juce::DynamicObject::Ptr body = new juce::DynamicObject();
    body->setProperty("prompt", prompt);
    
    makePostRequest("/api/projects/" + projectId + "/isolate", juce::var(body.get()),
                    [callback](const juce::var& response, bool success) {
        callback(success);
    });
}

void EngineClient::startMatching(const juce::String& projectId, std::function<void(bool)> callback)
{
    makePostRequest("/api/projects/" + projectId + "/match", juce::var(),
                    [callback](const juce::var& response, bool success) {
        callback(success);
    });
}

void EngineClient::getTaskStatus(const juce::String& projectId, 
                                  std::function<void(const TaskStatus&)> callback)
{
    makeGetRequest("/api/projects/" + projectId + "/status", 
                   [callback](const juce::var& response, bool success) {
        TaskStatus status;
        if (success)
        {
            status = parseTaskStatus(response);
        }
        else
        {
            status.state = "unknown";
            status.message = "Failed to get status";
            status.progress = 0.0f;
        }
        callback(status);
    });
}

void EngineClient::getProjectResults(const juce::String& projectId,
                                      std::function<void(const MatchingResults&, bool)> callback)
{
    makeGetRequest("/api/projects/" + projectId + "/results",
                   [this, projectId, callback](const juce::var& response, bool success) {
        MatchingResults results;
        
        if (success && response.hasProperty("matches"))
        {
            auto matchesArray = response["matches"];
            if (matchesArray.isArray())
            {
                for (int i = 0; i < matchesArray.size(); ++i)
                {
                    results.matches.add(parseMatchResult(matchesArray[i]));
                }
            }
            
            // Build audio URLs
            results.originalAudioUrl = baseUrl + "/api/projects/" + projectId + "/audio/original";
            results.processedAudioUrl = baseUrl + "/api/projects/" + projectId + "/audio/processed";
            results.isolatedAudioUrl = baseUrl + "/api/projects/" + projectId + "/audio/isolated";
        }
        
        callback(results, success && results.matches.size() > 0);
    });
}

juce::URL EngineClient::getAudioUrl(const juce::String& projectId, const juce::String& audioType)
{
    return juce::URL(baseUrl + "/api/projects/" + projectId + "/audio/" + audioType);
}

// HTTP Helper Methods

void EngineClient::makeGetRequest(const juce::String& endpoint,
                                   std::function<void(const juce::var&, bool)> callback)
{
    auto url = juce::URL(baseUrl + endpoint);
    
    std::thread([url, callback]() {
        auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
            .withConnectionTimeoutMs(5000);
        
        auto stream = url.createInputStream(options);
        
        if (stream != nullptr)
        {
            auto responseString = stream->readEntireStreamAsString();
            auto json = juce::JSON::parse(responseString);
            callback(json, true);
        }
        else
        {
            callback(juce::var(), false);
        }
    }).detach();
}

void EngineClient::makePostRequest(const juce::String& endpoint, const juce::var& body,
                                    std::function<void(const juce::var&, bool)> callback)
{
    juce::String jsonBody = body.isVoid() ? "{}" : juce::JSON::toString(body);
    
    auto url = juce::URL(baseUrl + endpoint)
        .withPOSTData(jsonBody);
    
    std::thread([url, callback]() {
        auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inPostData)
            .withExtraHeaders("Content-Type: application/json")
            .withConnectionTimeoutMs(10000);
        
        auto stream = url.createInputStream(options);
        
        if (stream != nullptr)
        {
            auto responseString = stream->readEntireStreamAsString();
            auto json = juce::JSON::parse(responseString);
            callback(json, true);
        }
        else
        {
            callback(juce::var(), false);
        }
    }).detach();
}

void EngineClient::makeMultipartRequest(const juce::String& endpoint, const juce::File& file,
                                         std::function<void(const juce::var&, bool)> callback)
{
    std::thread([this, endpoint, file, callback]() {
        // Read file into memory
        juce::MemoryBlock fileData;
        if (!file.loadFileAsData(fileData))
        {
            callback(juce::var(), false);
            return;
        }
        
        // Build multipart form data
        juce::String boundary = "----JUCEFormBoundary" + juce::String::toHexString(juce::Random::getSystemRandom().nextInt64());
        
        juce::MemoryOutputStream postData;
        postData << "--" << boundary << "\r\n";
        postData << "Content-Disposition: form-data; name=\"file\"; filename=\"" << file.getFileName() << "\"\r\n";
        postData << "Content-Type: audio/" << file.getFileExtension().substring(1) << "\r\n\r\n";
        postData.write(fileData.getData(), fileData.getSize());
        postData << "\r\n--" << boundary << "--\r\n";
        
        auto url = juce::URL(baseUrl + endpoint)
            .withPOSTData(postData.getMemoryBlock());
        
        auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inPostData)
            .withExtraHeaders("Content-Type: multipart/form-data; boundary=" + boundary)
            .withConnectionTimeoutMs(60000);
        
        auto stream = url.createInputStream(options);
        
        if (stream != nullptr)
        {
            auto responseString = stream->readEntireStreamAsString();
            auto json = juce::JSON::parse(responseString);
            callback(json, true);
        }
        else
        {
            callback(juce::var(), false);
        }
    }).detach();
}

// Parse helpers

RtrProject EngineClient::parseProject(const juce::var& json)
{
    RtrProject info;
    info.id = json.getProperty("id", "").toString();
    info.name = json.getProperty("name", "Untitled").toString();
    info.status = json.getProperty("status", "unknown").toString();
    info.createdAt = json.getProperty("created_at", "").toString();
    return info;
}

MatchResult EngineClient::parseMatchResult(const juce::var& json)
{
    MatchResult result;
    result.namModelId = json.getProperty("nam_model_id", "").toString();
    result.namModelName = json.getProperty("nam_model_name", "Unknown").toString();
    result.irId = json.getProperty("ir_id", "").toString();
    result.irName = json.getProperty("ir_name", "").toString();
    result.similarity = static_cast<float>(json.getProperty("similarity", 0.0));
    result.gain = static_cast<float>(json.getProperty("gain", 0.0));
    
    auto eqArray = json.getProperty("eq_settings", juce::var());
    if (eqArray.isArray())
    {
        for (int i = 0; i < eqArray.size(); ++i)
        {
            EQBand band;
            band.frequency = static_cast<float>(eqArray[i].getProperty("frequency", 1000.0));
            band.gain = static_cast<float>(eqArray[i].getProperty("gain", 0.0));
            band.q = static_cast<float>(eqArray[i].getProperty("q", 1.0));
            band.type = eqArray[i].getProperty("type", "peak").toString();
            result.eqSettings.add(band);
        }
    }
    
    return result;
}

TaskStatus EngineClient::parseTaskStatus(const juce::var& json)
{
    TaskStatus status;
    status.state = json.getProperty("state", "unknown").toString();
    status.message = json.getProperty("message", "").toString();
    status.progress = static_cast<float>(json.getProperty("progress", 0.0));
    return status;
}

