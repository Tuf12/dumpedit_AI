# filename_environment.py - Environment wrapper for Agent Byte filename generation
import re
import numpy as np
from datetime import datetime
from collections import Counter
import hashlib
import json   
import os


class FilenameEnvironment:
    """Environment wrapper that interfaces between DumpEdit and Agent Byte for filename generation"""

    def __init__(self):
        self.name = "filename_generation"
        self.state_size = 20  # Feature vector size for agent
        self.action_size = 8  # Number of filename generation strategies

        # Content type patterns for better analysis
        self.content_patterns = {
            'traceback': [
                r'Traceback \(most recent call last\)',
                r'File ".*", line \d+',
                r'\s+raise\s+\w+',
                r'Exception:',
                r'Error:',
                r'at \w+\.\w+\(',
                r'java\.lang\.\w+Exception',
                r'System\.Exception'
            ],
            'log': [
                r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}',
                r'\[(INFO|DEBUG|WARN|ERROR|FATAL)\]',
                r'^\d{2}:\d{2}:\d{2}',
                r'Log started',
                r'Application started',
                r'\d+ ms$',
                r'Exception in thread'
            ],
            'code': [
                r'def\s+\w+\(',
                r'class\s+\w+',
                r'import\s+\w+',
                r'from\s+\w+\s+import',
                r'if\s+__name__\s*==\s*["\']__main__["\']',
                r'function\s+\w+\(',
                r'var\s+\w+\s*=',
                r'public\s+(class|void|static)',
                r'#include\s*<',
                r'console\.log\(',
                r'print\('
            ],
            'email': [
                r'From:.*@.*',
                r'To:.*@.*',
                r'Subject:',
                r'Date:.*\d{4}',
                r'Reply-To:',
                r'Sent from my',
                r'Best regards',
                r'Sincerely',
                r'@\w+\.\w+'
            ],
            'meeting': [
                r'Meeting:',
                r'Agenda:',
                r'Attendees:',
                r'Action items:',
                r'\d{1,2}:\d{2}\s*(AM|PM)',
                r'Next meeting',
                r'Minutes:',
                r'Discussion:',
                r'Follow up'
            ],
            'config': [
                r'\w+\s*=\s*\w+',
                r'\[.*\]',
                r'#.*config',
                r'\.ini$',
                r'\.conf$',
                r'\.properties$',
                r'server\s*=',
                r'port\s*=',
                r'database\s*='
            ],
            'data': [
                r'^\s*\d+[,\s]+\d+',
                r'CSV|csv',
                r',".*",',
                r'^\s*[\d\w]+\s*\|\s*[\d\w]+',
                r'Total:',
                r'Count:',
                r'Sum:',
                r'Average:'
            ]
        }

        # Action mappings to filename strategies
        self.actions = {
            0: self._strategy_extract_title,
            1: self._strategy_error_type,
            2: self._strategy_log_datetime,
            3: self._strategy_code_function,
            4: self._strategy_email_subject,
            5: self._strategy_meeting_topic,
            6: self._strategy_content_keywords,
            7: self._strategy_fallback_summary
        }

        # Stop words for keyword extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'should', 'now', 'this', 'that', 'these', 'those',
            'file', 'line', 'error', 'exception', 'traceback'
        }

    def get_env_context(self):
        """Return environment context for Agent Byte's knowledge system"""
        return {
            'name': 'filename_generation',
            'objective': {
                'primary': 'Generate contextually relevant filenames from text content',
                'secondary': 'Learn user naming preferences through feedback',
                'win_condition': 'User accepts generated filename (thumbs up or file usage)'
            },
            'learning_parameters': {
                'recommended_gamma': 0.95,  # High gamma for immediate feedback learning
                'gamma_rationale': 'Filename feedback is immediate - prioritize recent rewards',
                'recommended_learning_rate': 0.001,
                'temporal_characteristics': {
                    'match_duration': '1-5 seconds',
                    'feedback_immediacy': 'Immediate',
                    'decision_frequency': 'Per content change'
                },
                'environment_complexity': {
                    'state_space': 'Medium (20 features)',
                    'action_space': 'Small (8 strategies)',
                    'reward_clarity': 'High (clear thumbs up/down)'
                }
            },
            'strategic_concepts': {
                'core_skills': [
                    'Content type detection',
                    'Keyword extraction',
                    'Pattern recognition',
                    'Context analysis'
                ],
                'tactical_approaches': [
                    'Extract meaningful titles or headers',
                    'Identify error types in tracebacks',
                    'Parse timestamps from logs',
                    'Detect function names in code',
                    'Extract email subjects',
                    'Summarize meeting topics'
                ],
                'success_patterns': [
                    'Short descriptive names work better than long ones',
                    'Content type context improves relevance',
                    'Timestamp inclusion helps with logs',
                    'Error type identification crucial for tracebacks'
                ]
            },
            'learning_recommendations': {
                'exploration_phase': [
                    'Try different content analysis strategies',
                    'Learn to identify content patterns',
                    'Experiment with filename length and format'
                ],
                'exploitation_phase': [
                    'Focus on proven successful patterns',
                    'Adapt to user preferences',
                    'Optimize for specific content types'
                ]
            },
            'failure_patterns': {
                'generic_naming': {
                    'description': 'Using overly generic names like "file" or "document"',
                    'solution': 'Extract specific content markers for context'
                },
                'too_long': {
                    'description': 'Generating filenames that are too long',
                    'solution': 'Keep names under 50 characters, focus on key terms'
                },
                'no_context': {
                    'description': 'Ignoring content type when generating names',
                    'solution': 'Always consider content patterns before naming'
                }
            },
            'context_metadata': {
                'environment_version': '1.0',
                'generated_at': datetime.now().timestamp(),
                'content_types_supported': list(self.content_patterns.keys())
            }
        }

    def get_environment_specific_constants(self):
        """Return constants for Agent Byte's modular behavior"""
        return {
            'good_filename_reward': 2.0,
            'bad_filename_penalty': -1.0,
            'override_penalty': -2.0,
            'file_usage_bonus': 1.5,
            'content_analysis_bonus': 0.5,
            'filename_length_target': 30,
            'max_filename_length': 50,
            'min_filename_length': 5
        }

    def analyze_content(self, text_content):
        """Convert text content to state representation for Agent Byte"""
        if not text_content or not text_content.strip():
            return np.zeros(self.state_size)

        content = text_content.strip()

        # Feature extraction
        features = []

        # Basic text statistics (0-4)
        features.append(min(len(content) / 1000, 1.0))  # Content length (normalized)
        features.append(len(content.split('\n')) / 100)  # Line count (normalized)
        features.append(len(content.split()) / 500)  # Word count (normalized)
        features.append(len(re.findall(r'[A-Z]', content)) / len(content) if content else 0)  # Uppercase ratio
        features.append(len(re.findall(r'\d', content)) / len(content) if content else 0)  # Digit ratio

        # Content type detection (5-11)
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                score += matches
            # Normalize by content length and number of patterns
            normalized_score = min(score / (len(content.split('\n')) + 1), 1.0)
            features.append(normalized_score)

        # Structure analysis (12-15)
        features.append(1.0 if content.startswith(('Subject:', 'From:', 'To:')) else 0.0)  # Email header
        features.append(
            1.0 if re.search(r'^[^a-z]*[A-Z][^a-z]*$', content.split('\n')[0]) else 0.0)  # Title-like first line
        features.append(
            len([line for line in content.split('\n') if line.strip().startswith('-')]) / 10)  # Bullet points
        features.append(
            1.0 if 'def ' in content or 'function ' in content or 'class ' in content else 0.0)  # Code presence

        # Content quality indicators (16-19)
        first_line = content.split('\n')[0] if content.split('\n') else ""
        features.append(min(len(first_line) / 100, 1.0))  # First line length
        features.append(1.0 if any(
            word in content.lower() for word in ['error', 'exception', 'traceback']) else 0.0)  # Error indicators
        features.append(1.0 if re.search(r'\d{4}-\d{2}-\d{2}', content) else 0.0)  # Date presence
        features.append(len(set(content.lower().split()) - self.stop_words) / 50)  # Unique meaningful words

        return np.array(features[:self.state_size])

    def interpret_reward(self, reward):
        """Provide environment-specific reward interpretation"""
        constants = self.get_environment_specific_constants()

        if reward >= constants['good_filename_reward']:
            return "Excellent filename generated - user approved!"
        elif reward >= constants['file_usage_bonus']:
            return "Good filename - user used the file"
        elif reward >= constants['content_analysis_bonus']:
            return "Decent filename with good content analysis"
        elif reward <= constants['override_penalty']:
            return "Poor filename - user manually overrode"
        elif reward <= constants['bad_filename_penalty']:
            return "Unsatisfactory filename - user rejected"
        else:
            return f"Neutral filename quality (reward: {reward:.1f})"

    def should_generate_lesson(self, reward):
        """Determine if this reward warrants generating a lesson"""
        constants = self.get_environment_specific_constants()
        return abs(reward) >= constants['content_analysis_bonus']

    def generate_lesson_from_reward(self, reward, context=None):
        """Generate learning lesson from reward feedback"""
        constants = self.get_environment_specific_constants()

        if reward >= constants['good_filename_reward']:
            return "High reward indicates successful content analysis and appropriate filename generation"
        elif reward <= constants['override_penalty']:
            return "Manual override suggests filename was irrelevant to content - improve pattern recognition"
        elif reward <= constants['bad_filename_penalty']:
            return "Negative feedback indicates poor content interpretation - analyze different content features"
        else:
            return f"Moderate reward ({reward:.1f}) - filename was acceptable but could be improved"

    def should_generate_strategy(self, reward):
        """Determine if this performance warrants a new strategy"""
        constants = self.get_environment_specific_constants()
        return reward >= constants['good_filename_reward']

    def generate_strategy_from_performance(self, wins, total_attempts, avg_reward):
        """Generate strategic insight from performance metrics"""
        win_rate = (wins / max(1, total_attempts)) * 100

        if win_rate > 80:
            return f"Dominant filename strategy: {win_rate:.1f}% acceptance rate - maintain current content analysis approach"
        elif win_rate > 60:
            return f"Effective filename generation: {win_rate:.1f}% success - focus on consistency and refinement"
        elif avg_reward > 1.0:
            return f"Good content understanding: avg reward {avg_reward:.1f} - expand successful pattern recognition"
        else:
            return f"Developing strategy: {win_rate:.1f}% success rate - need better content type detection"

    def format_user_demo_outcome(self, outcome, reward):
        """Format user demonstration outcome for logging"""
        if outcome == "thumbs_up":
            return f"User approved filename (reward: +{reward:.1f})"
        elif outcome == "thumbs_down":
            return f"User rejected filename (reward: {reward:.1f})"
        elif outcome == "override":
            return f"User provided manual override (reward: {reward:.1f})"
        elif outcome == "file_usage":
            return f"User used file with generated name (reward: +{reward:.1f})"
        else:
            return f"Unknown outcome: {outcome} (reward: {reward:.1f})"

    def get_performance_feedback_phrase(self, metric_type, value):
        """Get performance feedback phrase for UI"""
        if metric_type == "filename_quality":
            if value > 90:
                return "Excellent naming accuracy"
            elif value > 75:
                return "Good filename generation"
            elif value > 60:
                return "Decent naming performance"
            elif value > 40:
                return "Improving filename quality"
            else:
                return "Learning filename patterns"
        return f"Performance: {value:.1f}%"

    # Filename generation strategies
    def _strategy_extract_title(self, content, state):
        """Strategy: Extract title from first line or headers"""
        lines = content.split('\n')
        if not lines:
            return "untitled", 0.1

        first_line = lines[0].strip()

        # Check for email subject
        if first_line.lower().startswith('subject:'):
            title = first_line[8:].strip()
            return self._clean_filename(title), 0.9

        # Check for title-like first line
        if (len(first_line) < 80 and
                not first_line.endswith('.') and
                len(first_line.split()) <= 12):
            return self._clean_filename(first_line), 0.8

        # Check for headers marked with === or ---
        for i, line in enumerate(lines[:5]):
            if line.strip() and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('===') or next_line.startswith('---'):
                    return self._clean_filename(line.strip()), 0.8

        return None, 0.0

    def _strategy_error_type(self, content, state):
        """Strategy: Extract error type from tracebacks"""
        # Look for common exception patterns
        error_patterns = [
            r'(\w+Error): (.+)',
            r'(\w+Exception): (.+)',
            r'Exception in thread "([^"]+)"',
            r'Fatal error: (.+)',
            r'ERROR: (.+)'
        ]

        for pattern in error_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    error_type = match.group(1)
                    error_msg = match.group(2)[:30]  # Limit message length
                    filename = f"{error_type}_{error_msg}".replace(' ', '_')
                    return self._clean_filename(filename), 0.9
                else:
                    return self._clean_filename(f"error_{match.group(1)}"), 0.7

        # Look for traceback
        if 'Traceback (most recent call last)' in content:
            # Try to find the file where error occurred
            file_match = re.search(r'File "([^"]+)"', content)
            if file_match:
                filename = os.path.basename(file_match.group(1))
                return self._clean_filename(f"traceback_{filename}"), 0.8
            return "traceback_error", 0.6

        return None, 0.0

    def _strategy_log_datetime(self, content, state):
        """Strategy: Extract datetime from logs"""
        # Look for timestamps
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})',
            r'(\d{2}:\d{2}:\d{2})',
            r'(\d{4}/\d{2}/\d{2})',
            r'(\w{3} \d{1,2} \d{2}:\d{2}:\d{2})'
        ]

        for pattern in timestamp_patterns:
            match = re.search(pattern, content)
            if match:
                timestamp = match.group(1).replace(':', '').replace('-', '').replace('/', '').replace(' ', '_')

                # Look for log level
                log_level_match = re.search(r'\[(INFO|DEBUG|WARN|ERROR|FATAL)\]', content, re.IGNORECASE)
                if log_level_match:
                    level = log_level_match.group(1).lower()
                    return f"log_{level}_{timestamp}", 0.8

                return f"log_{timestamp}", 0.7

        return None, 0.0

    def _strategy_code_function(self, content, state):
        """Strategy: Extract function or class names from code"""
        # Look for function definitions
        func_patterns = [
            r'def\s+(\w+)\s*\(',
            r'function\s+(\w+)\s*\(',
            r'public\s+(?:static\s+)?(?:void\s+)?(\w+)\s*\(',
            r'class\s+(\w+)',
            r'interface\s+(\w+)'
        ]

        functions = []
        for pattern in func_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            functions.extend(matches)

        if functions:
            # Use the most common function name or the first one
            if len(functions) == 1:
                return self._clean_filename(f"code_{functions[0]}"), 0.8
            else:
                # Get most common
                func_counter = Counter(functions)
                most_common = func_counter.most_common(1)[0][0]
                return self._clean_filename(f"code_{most_common}"), 0.8

        # Look for imports to determine language
        if 'import ' in content:
            if 'numpy' in content or 'pandas' in content:
                return "python_script", 0.6
            elif 'java.' in content:
                return "java_code", 0.6
            else:
                return "code_file", 0.5

        return None, 0.0

    def _strategy_email_subject(self, content, state):
        """Strategy: Extract email subject or sender"""
        lines = content.split('\n')

        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.lower().startswith('subject:'):
                subject = line[8:].strip()
                return self._clean_filename(f"email_{subject}"), 0.9
            elif line.lower().startswith('from:'):
                # Extract sender name or email
                sender = line[5:].strip()
                # Remove email brackets
                sender = re.sub(r'<[^>]+>', '', sender).strip()
                # Get first name if possible
                sender_name = sender.split()[0] if sender.split() else sender
                return self._clean_filename(f"email_from_{sender_name}"), 0.8

        return None, 0.0

    def _strategy_meeting_topic(self, content, state):
        """Strategy: Extract meeting topic or agenda"""
        lines = content.split('\n')

        # Look for meeting-related headers
        meeting_keywords = ['meeting:', 'agenda:', 'discussion:', 'minutes:']

        for line in lines[:10]:
            line_lower = line.strip().lower()
            for keyword in meeting_keywords:
                if line_lower.startswith(keyword):
                    topic = line.strip()[len(keyword):].strip()
                    if topic:
                        return self._clean_filename(f"meeting_{topic}"), 0.8

        # Look for time patterns that might indicate meeting
        time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM))'
        if re.search(time_pattern, content, re.IGNORECASE):
            # Extract first meaningful line as topic
            for line in lines:
                if line.strip() and len(line.strip()) > 10:
                    topic = line.strip()[:30]
                    return self._clean_filename(f"meeting_{topic}"), 0.6

        return None, 0.0

    def _strategy_content_keywords(self, content, state):
        """Strategy: Extract important keywords"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())

        # Remove stop words
        meaningful_words = [w for w in words if w not in self.stop_words]

        if not meaningful_words:
            return None, 0.0

        # Get most common words
        word_counts = Counter(meaningful_words)
        top_words = [word for word, count in word_counts.most_common(3)]

        if top_words:
            filename = '_'.join(top_words)
            return self._clean_filename(filename), 0.5

        return None, 0.0

    def _strategy_fallback_summary(self, content, state):
        """Strategy: Fallback summary approach"""
        # Get first few meaningful words
        words = content.split()[:10]
        meaningful = [w for w in words if len(w) > 2 and w.lower() not in self.stop_words]

        if meaningful:
            # Take first 3 meaningful words
            summary_words = meaningful[:3]
            filename = '_'.join(summary_words)
            return self._clean_filename(filename), 0.3

        # Ultimate fallback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"content_{timestamp}", 0.1

    def _clean_filename(self, filename):
        """Clean and sanitize filename"""
        if not filename:
            return "unnamed"

        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', str(filename))
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')

        # Limit length
        if len(filename) > 45:
            filename = filename[:45]

        # Ensure minimum length
        if len(filename) < 3:
            filename = f"file_{filename}"

        return filename.lower()

    def generate_filename(self, content):
        """Main method to generate filename using best strategy"""
        if not content or not content.strip():
            return "empty_file"

        state = self.analyze_content(content)

        # Try strategies in order of preference based on content type
        strategy_results = []

        for action_id, strategy_func in self.actions.items():
            try:
                filename, confidence = strategy_func(content, state)
                if filename and confidence > 0.0:
                    strategy_results.append((filename, confidence, action_id))
            except Exception as e:
                print(f"Strategy {action_id} failed: {e}")
                continue

        # Sort by confidence and return best result
        if strategy_results:
            strategy_results.sort(key=lambda x: x[1], reverse=True)
            best_filename, best_confidence, best_action = strategy_results[0]
            return best_filename, best_action, best_confidence

        # Ultimate fallback
        return f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 7, 0.1


# Test the environment
if __name__ == "__main__":
    print("🧪 Testing Filename Environment...")

    env = FilenameEnvironment()

    # Test content types
    test_contents = [
        # Traceback
        """Traceback (most recent call last):
  File "test.py", line 23, in main
    result = divide_by_zero()
  File "test.py", line 15, in divide_by_zero
    return 10 / 0
ZeroDivisionError: division by zero""",

        # Log
        """2024-08-04 14:32:15 [INFO] Application started
2024-08-04 14:32:16 [DEBUG] Loading configuration
2024-08-04 14:32:17 [ERROR] Database connection failed
2024-08-04 14:32:18 [WARN] Retrying connection in 5 seconds""",

        # Email
        """From: john.doe@company.com
To: team@company.com
Subject: Weekly Team Meeting Notes
Date: Mon, 4 Aug 2024 14:30:00 -0500

Here are the notes from today's meeting...""",

        # Code
        """def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class FibonacciCalculator:
    def __init__(self):
        self.cache = {}""",

        # Meeting
        """Meeting: Q4 Planning Session
Date: August 4, 2024
Time: 2:00 PM - 3:30 PM
Attendees: Alice, Bob, Carol

Agenda:
- Review Q3 results
- Set Q4 objectives"""
    ]

    for i, content in enumerate(test_contents):
        print(f"\n--- Test {i + 1} ---")
        print(f"Content preview: {content[:50]}...")

        # Test state analysis
        state = env.analyze_content(content)
        print(f"State vector shape: {state.shape}")
        print(f"Content type scores: {state[5:12]}")  # Content type features

        # Test filename generation
        filename, action, confidence = env.generate_filename(content)
        print(f"Generated filename: {filename}")
        print(f"Strategy used: {action}")
        print(f"Confidence: {confidence:.2f}")

    print("\n✅ Filename Environment tests completed!")