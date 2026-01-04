#include "PluginProcessor.h"
#include "PluginEditor.h"

ReferenceToRigProcessor::ReferenceToRigProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    formatManager.registerBasicFormats();
}

ReferenceToRigProcessor::~ReferenceToRigProcessor()
{
    transportSource.setSource(nullptr);
}

const juce::String ReferenceToRigProcessor::getName() const
{
    return JucePlugin_Name;
}

bool ReferenceToRigProcessor::acceptsMidi() const { return false; }
bool ReferenceToRigProcessor::producesMidi() const { return false; }
bool ReferenceToRigProcessor::isMidiEffect() const { return false; }
double ReferenceToRigProcessor::getTailLengthSeconds() const { return 0.0; }

int ReferenceToRigProcessor::getNumPrograms() { return 1; }
int ReferenceToRigProcessor::getCurrentProgram() { return 0; }
void ReferenceToRigProcessor::setCurrentProgram(int) {}
const juce::String ReferenceToRigProcessor::getProgramName(int) { return {}; }
void ReferenceToRigProcessor::changeProgramName(int, const juce::String&) {}

void ReferenceToRigProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    transportSource.prepareToPlay(samplesPerBlock, sampleRate);
}

void ReferenceToRigProcessor::releaseResources()
{
    transportSource.releaseResources();
}

bool ReferenceToRigProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void ReferenceToRigProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                           juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Clear unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // If bypass is enabled, pass through audio unchanged
    if (bypassEnabled)
        return;

    // If playback is enabled, mix in the preview audio
    if (playbackEnabled && readerSource != nullptr)
    {
        juce::AudioSourceChannelInfo info(&buffer, 0, buffer.getNumSamples());
        transportSource.getNextAudioBlock(info);
    }
}

juce::AudioProcessorEditor* ReferenceToRigProcessor::createEditor()
{
    return new ReferenceToRigEditor(*this);
}

bool ReferenceToRigProcessor::hasEditor() const { return true; }

void ReferenceToRigProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    juce::ignoreUnused(destData);
}

void ReferenceToRigProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    juce::ignoreUnused(data, sizeInBytes);
}

void ReferenceToRigProcessor::loadAudioFile(const juce::File& file)
{
    auto* reader = formatManager.createReaderFor(file);
    
    if (reader != nullptr)
    {
        auto newSource = std::make_unique<juce::AudioFormatReaderSource>(reader, true);
        transportSource.setSource(newSource.get(), 0, nullptr, reader->sampleRate);
        readerSource = std::move(newSource);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ReferenceToRigProcessor();
}

