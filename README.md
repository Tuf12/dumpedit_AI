# DumpEdit with Agent Byte AI

**Intelligent text editor with AI-powered filename generation and content categorization**

## Overview

DumpEdit enhanced with Agent Byte v1.2 is a text editor that automatically generates contextually relevant filenames and categorizes content using advanced reinforcement learning. The AI agent learns from user feedback to continuously improve its understanding of text patterns and naming conventions.

## Features

### 🤖 AI-Powered Intelligence
- **Dual Learning System**: Neural network + symbolic knowledge for smart decision making
- **Adaptive Learning**: Environment-specific parameter optimization (gamma, learning rate)
- **Real-time Content Analysis**: Automatic categorization and filename generation
- **User Feedback Learning**: Improves over time based on thumbs up/down feedback

### 📝 Text Editor Features
- **Auto-save**: Content automatically saved with AI-generated filenames
- **Single File Workflow**: Maintains one active file, renamed dynamically
- **Manual Override**: Type `title: your_name` to manually set filename
- **Undo/Redo**: Full content history with backup system
- **Right-click Context**: Quick access to AI feedback options

### 🧠 Agent Byte v1.2 Architecture
- **Neural Network**: PyTorch-based Dueling DQN with 15,000+ parameters
- **Symbolic Knowledge**: Context-aware decision making with strategy effectiveness tracking
- **Modular Design**: Environment-specific behavior adaptation
- **Experience Replay**: 5000-item buffer for continuous learning
- **Knowledge System**: Lessons, strategies, and reflections with intelligent application

## Installation

### Prerequisites
- Python 3.7+
- tkinter (usually included with Python)
- Required packages: `numpy`, `torch`

### Setup
1. Clone or download all files to a directory
2. Install dependencies:
   ```bash
   pip install numpy torch
   ```
3. Run the application:
   ```bash
   python dumpedit.py
   ```

## File Structure

```
dumpedit/
├── dumpedit.py                    # Main application with dual learning UI
├── agent_byte.py                  # Agent Byte v1.2 main class
├── dual_brain_system.py          # Dual brain architecture (neural + symbolic)
├── knowledge_system.py           # Symbolic decision making system
├── dueling_network_pytorch.py    # PyTorch neural networks
├── filename_environment.py       # Filename generation environment
├── agent_integration_plan.txt    # Project documentation
├── autosaved_notes/               # Auto-saved files directory
├── agent_brain.json              # Core learning engine state (auto-generated)
├── agent_knowledge.json          # Symbolic knowledge base (auto-generated)
├── agent_main_network.pth        # PyTorch main network weights (auto-generated)
├── agent_target_network.pth      # PyTorch target network weights (auto-generated)
├── agent_matches.json            # Training logs (auto-generated)
├── dual_learning_log.json        # Dual learning feedback log (auto-generated)
└── README.md                      # This file
```

## Usage

### Basic Workflow
1. **Paste/Type Content**: Agent Byte analyzes content in real-time
2. **AI Processing**: 
   - Predicts content category (code, email, documentation, etc.)
   - Generates contextual filename using learned strategies
   - Updates window title with new filename
3. **Provide Feedback**: 
   - 👍 **Good**: Click if filename is appropriate
   - 👎 **Poor**: Click to see alternatives or provide correct name
   - **Alternatives**: Choose from AI-generated options
   - **Category Review**: Confirm or correct content categorization

### Manual Override
Type on the first line:
```
title: your_custom_filename
```
The agent learns from this override as negative feedback.

### AI Learning Features

#### Filename Generation
The AI uses multiple strategies:
- **Code Identifiers**: Extracts class/function names from code
- **Email Subjects**: Uses subject lines for email content
- **Error Analysis**: Identifies error types and context
- **Metadata Extraction**: Finds timestamps, file references
- **Descriptive Generation**: Creates meaningful names from content analysis

#### Content Categorization
Agent Byte recognizes and learns:
- Code (Python, JavaScript, etc.)
- Email/Communication
- Documentation/Notes
- Error logs/Debug info
- Meeting notes
- Configuration files
- And more through user feedback

#### Dual Learning System
- **Filename Learning**: Reinforces successful naming strategies (+2.0 reward)
- **Category Learning**: Improves content classification (+1.5 reward)
- **User Teaching**: Learns from manual corrections (-1.5 penalty, but gains knowledge)
- **Combined Bonus**: Extra reward when both filename and category are correct (+0.5)

## AI Agent Details

### Agent Byte v1.2 Capabilities
- **Training Steps**: Continuously learning from each interaction
- **Exploration Rate**: Balances trying new strategies vs. using learned ones
- **Win Rate**: Success percentage based on user feedback
- **Knowledge Effectiveness**: Ratio of symbolic vs. neural decision success
- **Strategy Performance**: Tracks which approaches work best for different content types

### Neural Network Architecture
- **Input Size**: 14 features (content analysis)
- **Hidden Layers**: [64, 32, 16] neurons with LeakyReLU activation
- **Output Size**: 5 actions (different filename generation strategies)
- **Dueling DQN**: Separate value and advantage streams for better learning
- **Double DQN**: Reduces overestimation bias in Q-learning

### Adaptive Learning Parameters
- **Gamma**: Discount factor automatically optimized per content type
- **Learning Rate**: Adaptive adjustment based on performance trends
- **Target Updates**: Soft updates with performance-based tau
- **Gradient Clipping**: Adaptive based on reward variance

## User Interface

### Main Window
- **Text Area**: Primary editing space
- **Window Title**: Shows current filename
- **Button Bar**: Undo, Redo, Clear, Open Folder
- **AI Feedback Section**: Good/Poor buttons, Alternatives, Category Review

### Feedback Dialogs
- **Filename Alternatives**: Shows AI-generated options with strategy explanations
- **Category Review**: Displays predicted category with confidence and trigger words
- **Teaching Mode**: Allows users to provide correct examples for learning

### Status Indicators
- **Processing Status**: Shows AI analysis progress and results
- **Category Display**: Current predicted category with trigger words
- **Learning Feedback**: Confirms when agent learns from user input

## Performance and Learning

### Success Metrics
- **Filename Acceptance Rate**: Percentage of AI-generated names users keep
- **Category Accuracy**: Correct content type predictions
- **Learning Speed**: How quickly the agent improves with feedback
- **User Satisfaction**: Reduced need for manual overrides over time

### Learning Data Storage
- **Brain State**: Core neural network weights and training statistics
- **Knowledge Base**: Symbolic strategies, lessons, and performance insights
- **Training Logs**: Recent match history with detailed learning metrics
- **Dual Learning Log**: Comprehensive feedback history for analysis

### Adaptive Features
- **Environment Optimization**: Different learning parameters for different content types
- **Strategy Evolution**: New approaches discovered through successful interactions
- **Pattern Recognition**: Improved text analysis through user feedback
- **Personalization**: Learns individual user preferences and naming conventions

## Technical Architecture

### Modular Design
- **Environment Integration**: Pluggable design for different use cases
- **Dual Brain System**: Combines neural learning with symbolic reasoning
- **Knowledge System**: Stores and applies learned strategies intelligently
- **Feedback Loop**: Continuous improvement through user interaction

### Learning Pipeline
1. **Content Analysis**: Extract features from text
2. **Strategy Selection**: Choose appropriate filename generation approach
3. **Execution**: Generate filename using selected strategy
4. **User Feedback**: Collect approval/rejection signals
5. **Learning Update**: Adjust neural weights and symbolic knowledge
6. **Performance Tracking**: Monitor improvement over time

### Data Flow
```
Text Content → Feature Extraction → Strategy Selection → Filename Generation
     ↓                                                           ↓
User Feedback ← UI Interaction ← Window Update ← File Rename ←──┘
     ↓
Learning Update → Knowledge Base → Future Strategy Selection
```

## Troubleshooting

### Common Issues
- **AI Not Working**: Check if all Python files are in the same directory
- **No Feedback Buttons**: Ensure agent_byte.py and related files are present
- **Poor Filenames**: Provide feedback to help the agent learn your preferences
- **File Conflicts**: Agent automatically handles filename conflicts by renaming

### Debug Information
The application logs detailed information to:
- Console output for real-time debugging
- `dual_learning_log.json` for comprehensive feedback history
- `agent_matches.json` for training session details

### Reset Learning
To reset the AI learning:
1. Delete `agent_brain.json`, `agent_knowledge.json`
2. Delete `agent_main_network.pth`, `agent_target_network.pth`
3. Restart the application

## Development

### Extending the AI
- **New Content Types**: Add patterns to `filename_environment.py`
- **Custom Strategies**: Extend strategy mappings in `knowledge_system.py`
- **Learning Parameters**: Modify adaptive learning in `dual_brain_system.py`

### Adding Features
- **New Feedback Types**: Extend the dual learning feedback system
- **UI Enhancements**: Add new interface elements to `dumpedit.py`
- **Analysis Tools**: Create new content analysis features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project combines text editing functionality with advanced AI learning systems. The Agent Byte AI architecture represents a sophisticated approach to learning from human feedback in real-world applications.

## Contributing

When contributing:
1. Maintain the dual learning architecture
2. Preserve backward compatibility with existing knowledge bases
3. Add comprehensive logging for new features
4. Test with various content types
5. Document new AI capabilities

## Credits

- **Agent Byte v1.2**: Advanced reinforcement learning with dual brain architecture
- **PyTorch Integration**: Modern neural network implementation
- **Symbolic Knowledge System**: Human-interpretable learning and decision making
- **Adaptive Learning**: Environment-specific optimization techniques

---

**DumpEdit with Agent Byte AI** - Where human creativity meets artificial intelligence for intelligent text processing.