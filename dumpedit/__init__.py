# __init__.py - DumpEdit AI Agent Package Initialization
"""
DumpEdit AI Agent Package

This package provides intelligent filename generation for the DumpEdit text editor
using Agent Byte v1.2 - a modular reinforcement learning agent with dual brain
architecture (neural + symbolic knowledge).

Components:
- agent_byte.py: Main Agent Byte v1.2 class with dual brain architecture
- dual_brain_system.py: Core learning engine + symbolic knowledge system
- knowledge_system.py: Symbolic decision making and strategy application
- dueling_network_pytorch.py: PyTorch neural networks with adaptive learning
- filename_environment.py: Environment wrapper for filename generation
- dumpedit_ai.py: Enhanced DumpEdit application with agent integration

Features:
- Neural Network: PyTorch-based Dueling DQN with 15,000+ parameters
- Symbolic Knowledge: Context-aware decision making with environmental adaptation
- Adaptive Learning: Environment-specific parameter optimization (gamma, learning rate)
- Modular Design: Environment-specific behavior integration
- User Feedback Learning: Learns from thumbs up/down and manual overrides
"""

__version__ = "1.2.1"
__author__ = "Agent Byte Development Team"
__description__ = "Intelligent filename generation with reinforcement learning"

# Import core components
try:
    from .agent_byte import AgentByte, MatchLogger
    from .dual_brain_system import DualBrainAgent, AgentBrain, AgentKnowledge
    from .knowledge_system import SymbolicDecisionMaker, KnowledgeInterpreter
    from .filename_environment import FilenameEnvironment

    # Optional PyTorch components (graceful fallback if PyTorch not available)
    try:
        from .dueling_network_pytorch import DuelingNetworkPyTorch, DuelingNetworkCompatibility

        PYTORCH_AVAILABLE = True
    except ImportError:
        print("⚠️ PyTorch not available - using NumPy fallback networks")
        PYTORCH_AVAILABLE = False

    # Main application
    try:
        from .dumpedit_ai import DumpEditWithAgent

        MAIN_APP_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ Main application not available: {e}")
        MAIN_APP_AVAILABLE = False

    AGENT_AVAILABLE = True

except ImportError as e:
    print(f"❌ Agent components not available: {e}")
    AGENT_AVAILABLE = False
    PYTORCH_AVAILABLE = False
    MAIN_APP_AVAILABLE = False

# Package metadata
__all__ = [
    # Core agent components
    'AgentByte',
    'MatchLogger',
    'DualBrainAgent',
    'AgentBrain',
    'AgentKnowledge',
    'SymbolicDecisionMaker',
    'KnowledgeInterpreter',
    'FilenameEnvironment',

    # PyTorch components (if available)
    'DuelingNetworkPyTorch',
    'DuelingNetworkCompatibility',

    # Main application
    'DumpEditWithAgent',

    # Status flags
    'AGENT_AVAILABLE',
    'PYTORCH_AVAILABLE',
    'MAIN_APP_AVAILABLE'
]


def create_agent_for_filename_generation():
    """
    Convenience function to create a properly configured Agent Byte
    for filename generation with all required components.

    Returns:
        tuple: (agent, environment) if successful, (None, None) if failed
    """
    if not AGENT_AVAILABLE:
        print("❌ Agent components not available")
        return None, None

    try:
        # Create filename environment
        env = FilenameEnvironment()

        # Create agent with appropriate dimensions
        agent = AgentByte(
            state_size=env.state_size,
            action_size=env.action_size,
            app_name="filename_generation"
        )

        # Set up environment integration
        agent.set_environment(env)

        # Start agent session with filename generation context
        env_context = env.get_env_context()
        agent.start_new_match("filename_generation", env_context=env_context)

        print("✅ Agent Byte configured for filename generation")
        return agent, env

    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return None, None


def get_package_info():
    """
    Get comprehensive package information including component availability.

    Returns:
        dict: Package information and component status
    """
    info = {
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'components': {
            'agent_available': AGENT_AVAILABLE,
            'pytorch_available': PYTORCH_AVAILABLE,
            'main_app_available': MAIN_APP_AVAILABLE
        }
    }

    if AGENT_AVAILABLE:
        # Get agent capabilities
        try:
            agent, env = create_agent_for_filename_generation()
            if agent and env:
                stats = agent.get_stats()
                info['agent_stats'] = {
                    'architecture': stats.get('architecture', 'Unknown'),
                    'training_steps': stats.get('training_steps', 0),
                    'network_parameters': stats.get('network_parameters', 0),
                    'adaptive_learning': stats.get('learning_parameters_adapted', False),
                    'environment_integrated': stats.get('environment_integrated', False)
                }
                # Clean up
                agent.save_brain()
        except Exception as e:
            info['agent_stats'] = {'error': str(e)}

    return info


def check_dependencies():
    """
    Check all package dependencies and provide installation guidance.

    Returns:
        dict: Dependency status and installation instructions
    """
    dependencies = {
        'required': {
            'numpy': False,
            'json': False,  # Built-in
            'os': False,  # Built-in
            'time': False,  # Built-in
            'random': False,  # Built-in
            'datetime': False,  # Built-in
            'collections': False,  # Built-in
            're': False,  # Built-in
        },
        'optional': {
            'torch': False,
            'tkinter': False
        },
        'installation_commands': {
            'torch': 'pip install torch',
            'numpy': 'pip install numpy',
            'tkinter': 'Usually included with Python, or: sudo apt-get install python3-tk (Linux)'
        }
    }

    # Check required dependencies
    try:
        import numpy
        dependencies['required']['numpy'] = True
    except ImportError:
        pass

    # Built-ins are always available
    for builtin in ['json', 'os', 'time', 'random', 'datetime', 'collections', 're']:
        dependencies['required'][builtin] = True

    # Check optional dependencies
    try:
        import torch
        dependencies['optional']['torch'] = True
    except ImportError:
        pass

    try:
        import tkinter
        dependencies['optional']['tkinter'] = True
    except ImportError:
        pass

    return dependencies


# Initialization message
if AGENT_AVAILABLE:
    print(f"🚀 DumpEdit AI Agent Package v{__version__} loaded successfully!")
    print(f"   🧠 Agent Byte v1.2: {'✅' if AGENT_AVAILABLE else '❌'}")
    print(f"   🔥 PyTorch Networks: {'✅' if PYTORCH_AVAILABLE else '❌'}")
    print(f"   🖥️ GUI Application: {'✅' if MAIN_APP_AVAILABLE else '❌'}")

    if not PYTORCH_AVAILABLE:
        print("   💡 Install PyTorch for GPU acceleration: pip install torch")

    if not MAIN_APP_AVAILABLE:
        print("   💡 Ensure tkinter is available for GUI: python -m tkinter")
else:
    print(f"⚠️ DumpEdit AI Agent Package v{__version__} - Core components unavailable")
    print("   💡 Check dependencies with: check_dependencies()")