#include "ProjectListComponent.h"

ProjectListComponent::ProjectListComponent()
{
    listBox.setModel(this);
    listBox.setRowHeight(70);
    listBox.setColour(juce::ListBox::backgroundColourId, juce::Colours::transparentBlack);
    listBox.setColour(juce::ListBox::outlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(listBox);
    
    emptyLabel.setText("No projects yet.\nCreate a new project to get started!", juce::dontSendNotification);
    emptyLabel.setFont(juce::Font(18.0f));
    emptyLabel.setColour(juce::Label::textColourId, textColor.withAlpha(0.5f));
    emptyLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(emptyLabel);
}

void ProjectListComponent::paint(juce::Graphics& g)
{
    g.setColour(bgColor);
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 8.0f);
}

void ProjectListComponent::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    listBox.setBounds(bounds);
    emptyLabel.setBounds(bounds);
    
    emptyLabel.setVisible(projectList.isEmpty());
    listBox.setVisible(!projectList.isEmpty());
}

int ProjectListComponent::getNumRows()
{
    return projectList.size();
}

void ProjectListComponent::paintListBoxItem(int rowNumber, juce::Graphics& g, 
                                             int width, int height, bool rowIsSelected)
{
    if (rowNumber < 0 || rowNumber >= projectList.size())
        return;
    
    auto& project = projectList.getReference(rowNumber);
    
    // Background
    if (rowIsSelected)
    {
        g.setColour(selectedColor.withAlpha(0.2f));
        g.fillRoundedRectangle(2.0f, 2.0f, width - 4.0f, height - 4.0f, 6.0f);
        
        g.setColour(selectedColor);
        g.drawRoundedRectangle(2.0f, 2.0f, width - 4.0f, height - 4.0f, 6.0f, 2.0f);
    }
    else
    {
        g.setColour(bgColor.brighter(0.05f));
        g.fillRoundedRectangle(2.0f, 2.0f, width - 4.0f, height - 4.0f, 6.0f);
    }
    
    // Status indicator
    juce::Colour statusColor = juce::Colours::grey;
    if (project.status == "completed") statusColor = juce::Colours::limegreen;
    else if (project.status == "processing") statusColor = juce::Colours::orange;
    else if (project.status == "failed") statusColor = juce::Colours::red;
    
    g.setColour(statusColor);
    g.fillEllipse(15.0f, (height - 10.0f) / 2.0f, 10.0f, 10.0f);
    
    // Project name
    g.setColour(textColor);
    g.setFont(juce::Font(16.0f, juce::Font::bold));
    g.drawText(project.name.isEmpty() ? "Untitled Project" : project.name,
               35, 12, width - 150, 24, juce::Justification::centredLeft);
    
    // Status text
    g.setColour(subtextColor);
    g.setFont(juce::Font(12.0f));
    g.drawText(project.status, 35, 38, width - 150, 20, juce::Justification::centredLeft);
    
    // Date
    g.drawText(project.createdAt.substring(0, 10), width - 110, 12, 100, height - 24, 
               juce::Justification::centredRight);
}

void ProjectListComponent::listBoxItemClicked(int row, const juce::MouseEvent&)
{
    // Single click just selects
}

void ProjectListComponent::listBoxItemDoubleClicked(int row, const juce::MouseEvent&)
{
    if (row >= 0 && row < projectList.size() && onProjectSelected)
    {
        onProjectSelected(projectList[row].id);
    }
}

void ProjectListComponent::setProjects(const juce::Array<RtrProject>& projects)
{
    projectList = projects;
    listBox.updateContent();
    resized();
}

