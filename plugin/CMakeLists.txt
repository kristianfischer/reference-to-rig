cmake_minimum_required(VERSION 3.22)
project(ReferenceToRig VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add JUCE
add_subdirectory(JUCE)

# Define source files
set(PLUGIN_SOURCES
    src/PluginProcessor.cpp
    src/PluginProcessor.h
    src/PluginEditor.cpp
    src/PluginEditor.h
    src/api/EngineClient.cpp
    src/api/EngineClient.h
    src/components/FileDropComponent.cpp
    src/components/FileDropComponent.h
    src/components/ProjectListComponent.cpp
    src/components/ProjectListComponent.h
    src/components/ResultsComponent.cpp
    src/components/ResultsComponent.h
    src/components/ProgressComponent.cpp
    src/components/ProgressComponent.h
    src/components/AudioPlayerComponent.cpp
    src/components/AudioPlayerComponent.h
    src/components/EQVisualizerComponent.cpp
    src/components/EQVisualizerComponent.h
    src/utils/AsyncTaskRunner.cpp
    src/utils/AsyncTaskRunner.h
)

# Create the plugin
juce_add_plugin(ReferenceToRig
    COMPANY_NAME "ReferenceToRig"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_MANUFACTURER_CODE Rtr0
    PLUGIN_CODE Rtr1
    FORMATS VST3 Standalone
    PRODUCT_NAME "Reference to Rig"
)

target_sources(ReferenceToRig
    PRIVATE
        ${PLUGIN_SOURCES}
)

# Generate JuceHeader.h
juce_generate_juce_header(ReferenceToRig)

target_compile_definitions(ReferenceToRig
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_include_directories(ReferenceToRig
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/ReferenceToRig_artefacts/JuceLibraryCode
)

target_link_libraries(ReferenceToRig
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

