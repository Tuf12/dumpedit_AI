import json
import os
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import platform
import re
from datetime import datetime
import threading
import time

import numpy as np

# Import Agent Byte and environment
try:
    from agent_byte import AgentByte
    from filename_environment import FilenameEnvironment

    AGENT_AVAILABLE = True
    print("✅ Agent Byte loaded successfully")
except ImportError as e:
    print(f"⚠️ Agent Byte not available: {e}")
    AGENT_AVAILABLE = False

SAVE_FOLDER = "autosaved_notes"
BACKUP_COUNT = 5

os.makedirs(SAVE_FOLDER, exist_ok=True)


class DumpEditWithAgent:
    def __init__(self, root):
        self.root = root
        self.root.title("DumpEdit")
        self.current_filename = "dumpedit_notes.txt"  # Initial filename

        # Initialize Agent Byte if available
        self.agent = None
        self.filename_env = None
        self.agent_enabled = False

        if AGENT_AVAILABLE:
            # Add missing import for JSON logging
            import json
            self.filename_env = FilenameEnvironment()
            self.agent = AgentByte(
                state_size=self.filename_env.state_size,
                action_size=self.filename_env.action_size,
                app_name="filename_generation"
            )

            # Set up the environment for the agent
            self.agent.set_environment(self.filename_env)

            # Start agent session with filename generation context
            env_context = self.filename_env.get_env_context()
            self.agent.start_new_match("filename_generation", env_context=env_context)

            self.agent_enabled = True
            print("🤖 Agent Byte initialized for filename generation")

        else:
            print("❌ Agent Byte not available")
            self.agent_enabled = False

        self.setup_ui()

        # State management
        self.backups = []
        self.current_index = -1
        self.last_content_hash = None
        self.pending_filename_update = False

        # Load existing file
        self.load_file()

        # Update title to show current filename
        self.update_window_title()

    def setup_ui(self):
        """Setup the user interface"""
        # Main text area
        self.text = tk.Text(self.root, wrap=tk.WORD)
        self.text.pack(expand=True, fill=tk.BOTH)

        # Button frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        # Left side buttons
        left_frame = tk.Frame(btn_frame)
        left_frame.pack(side=tk.LEFT)

        self.undo_btn = tk.Button(left_frame, text="Undo", command=self.undo)
        self.undo_btn.pack(side=tk.LEFT, padx=2)

        self.redo_btn = tk.Button(left_frame, text="Redo", command=self.redo)
        self.redo_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = tk.Button(left_frame, text="Clear", command=self.clear_screen)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        # Agent feedback buttons (only if agent is available)
        if self.agent_enabled:
            feedback_frame = tk.Frame(btn_frame)
            feedback_frame.pack(side=tk.LEFT, padx=20)

            tk.Label(feedback_frame, text="Filename:").pack(side=tk.LEFT)

            self.thumbs_up_btn = tk.Button(
                feedback_frame,
                text="👍",
                command=self.thumbs_up_feedback,
                bg="lightgreen",
                font=("Arial", 12)
            )
            self.thumbs_up_btn.pack(side=tk.LEFT, padx=2)

            self.thumbs_down_btn = tk.Button(
                feedback_frame,
                text="👎",
                command=self.thumbs_down_feedback,
                bg="lightcoral",
                font=("Arial", 12)
            )
            self.thumbs_down_btn.pack(side=tk.LEFT, padx=2)

            # Agent status indicator
            self.agent_status_label = tk.Label(
                feedback_frame,
                text="🤖 Ready",
                fg="green"
            )
            self.agent_status_label.pack(side=tk.LEFT, padx=10)

        # Right side buttons
        right_frame = tk.Frame(btn_frame)
        right_frame.pack(side=tk.RIGHT)

        self.open_folder_btn = tk.Button(right_frame, text="Open Folder", command=self.open_folder)
        self.open_folder_btn.pack(side=tk.RIGHT, padx=2)

        # Bind events
        self.text.bind("<KeyRelease>", self.on_content_change)
        self.text.bind("<Control-a>", self.select_all)
        self.text.bind("<Button-3>", self.show_right_click_menu)
        self.text.bind("<Control-v>", self.on_paste)  # Detect paste events

        # Right-click menu
        self.setup_right_click_menu()

    def setup_right_click_menu(self):
        """Setup right-click context menu"""
        self.right_click_menu = tk.Menu(self.root, tearoff=0)
        self.right_click_menu.add_command(label="Cut", command=lambda: self.text.event_generate("<<Cut>>"))
        self.right_click_menu.add_command(label="Copy", command=lambda: self.text.event_generate("<<Copy>>"))
        self.right_click_menu.add_command(label="Paste", command=lambda: self.text.event_generate("<<Paste>>"))
        self.right_click_menu.add_command(label="Select All", command=self.select_all)

        if self.agent_enabled:
            self.right_click_menu.add_separator()
            self.right_click_menu.add_command(label="👍 Good Filename", command=self.thumbs_up_feedback)
            self.right_click_menu.add_command(label="👎 Bad Filename", command=self.thumbs_down_feedback)
            self.right_click_menu.add_command(label="🔄 Regenerate Name", command=self.regenerate_filename)

    def on_paste(self, event=None):
        """Handle paste events - trigger immediate filename update"""
        # Schedule filename update after paste completes
        self.root.after(100, self.on_significant_content_change)

    def on_content_change(self, event=None):
        """Handle content changes"""
        content = self.text.get("1.0", tk.END).strip()

        # Check for manual filename override
        if self.check_manual_override(content):
            return

        # Regular content handling
        if content:
            self.add_backup(content)
            self.save_file(content)

            # Schedule filename update if content changed significantly
            content_hash = hash(content)
            if self.last_content_hash != content_hash:
                self.last_content_hash = content_hash
                if not self.pending_filename_update:
                    self.pending_filename_update = True
                    # Delay filename update to avoid updating on every keystroke
                    self.root.after(2000, self.on_significant_content_change)  # 2 second delay

    def on_significant_content_change(self):
        """Handle significant content changes that should trigger filename update"""
        self.pending_filename_update = False
        content = self.text.get("1.0", tk.END).strip()

        if content and self.agent_enabled and not self.check_manual_override(content):
            self.update_filename_with_agent(content)

    def check_manual_override(self, content):
        """Check for manual filename override in content"""
        if not content:
            return False

        lines = content.split('\n')
        if not lines:
            return False

        first_line = lines[0].strip()

        # Check for title: override (case insensitive)
        if first_line.lower().startswith('title:'):
            # Extract the manual filename
            manual_name = first_line[6:].strip()
            if manual_name:
                # Remove the title line from content
                remaining_content = '\n'.join(lines[1:]).strip()

                # Apply manual filename
                self.apply_manual_filename(manual_name, remaining_content)

                # Give negative reward to agent
                if self.agent_enabled:
                    self.give_agent_feedback(-2.0, "override")

                return True

        return False

    def apply_manual_filename(self, manual_name, remaining_content):
        """Apply manually specified filename"""
        try:
            # Clean the filename
            clean_name = self.filename_env._clean_filename(manual_name) if self.filename_env else manual_name
            clean_name = clean_name.replace(' ', '_')

            # Ensure .txt extension
            if not clean_name.endswith('.txt'):
                clean_name += '.txt'

            old_filepath = os.path.join(SAVE_FOLDER, self.current_filename)
            new_filepath = os.path.join(SAVE_FOLDER, clean_name)  # FIXED: Use clean_name, not new_filename

            # Remove new file if it exists (prevent conflicts)
            if os.path.exists(new_filepath) and old_filepath != new_filepath:
                os.remove(new_filepath)

            # Rename the single file
            if os.path.exists(old_filepath):
                os.rename(old_filepath, new_filepath)

            # Update filename
            self.current_filename = clean_name  # FIXED: Use clean_name, not new_filename

            # Update text content (remove title line)
            self.text.delete("1.0", tk.END)
            self.text.insert("1.0", remaining_content)

            # Save with new name
            self.save_file(remaining_content)
            self.update_window_title()

            if self.agent_enabled:
                self.agent_status_label.config(text="🤖 Manual Override", fg="orange")

            print(f"📝 Manual filename applied: {clean_name}")

        except Exception as e:
            print(f"❌ Error applying manual filename: {e}")

    def update_filename_with_agent(self, content):
        """Use Agent Byte to generate new filename"""
        if not self.agent_enabled or not content.strip():
            return

        try:
            self.agent_status_label.config(text="🤖 Analyzing...", fg="blue")

            # Analyze content and get state
            state = self.filename_env.analyze_content(content)

            # Get agent's action (filename strategy)
            action = self.agent.get_action(state)

            # Generate filename using selected strategy
            new_filename, strategy_used, confidence = self.filename_env.generate_filename(content)

            # Ensure .txt extension
            if not new_filename.endswith('.txt'):
                new_filename += '.txt'

            # Apply the new filename
            self.apply_new_filename(new_filename, strategy_used, confidence)

            # Store action for learning
            self.last_agent_state = state
            self.last_agent_action = action
            self.last_strategy_used = strategy_used
            self.last_confidence = confidence

            self.agent_status_label.config(text="🤖 Generated", fg="green")

        except Exception as e:
            print(f"❌ Agent filename generation error: {e}")
            self.agent_status_label.config(text="🤖 Error", fg="red")

    def apply_new_filename(self, new_filename, strategy_used, confidence):
        """Apply new filename generated by agent - SINGLE FILE VERSION"""
        if new_filename == self.current_filename:
            return  # No change needed

        old_filepath = os.path.join(SAVE_FOLDER, self.current_filename)
        new_filepath = os.path.join(SAVE_FOLDER, new_filename)  # FIXED: Use new_filename parameter

        try:
            # Remove new file if it exists (prevent conflicts)
            if os.path.exists(new_filepath) and old_filepath != new_filepath:
                os.remove(new_filepath)

            # Rename the single file
            if os.path.exists(old_filepath):
                os.rename(old_filepath, new_filepath)
            else:
                # If old file doesn't exist, create new file with current content
                content = self.text.get("1.0", tk.END)
                with open(new_filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            self.current_filename = new_filename  # FIXED: Use new_filename parameter
            self.update_window_title()

            print(
                f"🤖 Agent renamed file: {self.current_filename} (strategy: {strategy_used}, confidence: {confidence:.2f})")

        except Exception as e:
            print(f"❌ Error applying new filename: {e}")

    def apply_filename_choice(self, chosen_filename, choice_type, reward):
        """Apply the user's filename choice and give appropriate reward - SINGLE FILE VERSION"""
        try:
            # Clean the filename
            clean_name = self.filename_env._clean_filename(chosen_filename) if self.filename_env else chosen_filename
            clean_name = clean_name.replace(' ', '_')

            # Ensure .txt extension
            if not clean_name.endswith('.txt'):
                clean_name += '.txt'

            old_filepath = os.path.join(SAVE_FOLDER, self.current_filename)
            new_filepath = os.path.join(SAVE_FOLDER, clean_name)  # FIXED: Use clean_name, not undefined variable

            # Remove new file if it exists (prevent conflicts)
            if os.path.exists(new_filepath) and old_filepath != new_filepath:
                os.remove(new_filepath)

            # Rename the single file
            if os.path.exists(old_filepath):
                os.rename(old_filepath, new_filepath)

            self.current_filename = clean_name  # FIXED: Use clean_name, not undefined variable
            self.update_window_title()

            # Give appropriate reward and feedback
            if choice_type == "agent_suggestion":
                self.give_agent_feedback(reward, "agent_suggestion_chosen")
                self.agent_status_label.config(text="🤖 👍 Suggestion Accepted", fg="green")
                print(f"✅ User chose agent suggestion: {clean_name} (reward: +{reward})")
            elif choice_type == "user_custom":
                # Give negative reward for custom name (agent failed to generate good name)
                self.give_agent_feedback(-1.5, "user_custom_name")
                self.agent_status_label.config(text="🤖 User Override", fg="orange")
                print(f"📝 User provided custom filename: {clean_name} (reward: -1.5 - agent needs improvement)")

        except Exception as e:
            print(f"❌ Error applying filename choice: {e}")
            self.agent_status_label.config(text="🤖 Error", fg="red")

    def thumbs_up_feedback(self):
        """User likes the generated filename"""
        if not self.agent_enabled:
            return

        try:
            self.give_agent_feedback(2.0, "thumbs_up")
            self.agent_status_label.config(text="🤖 👍 Learned", fg="green")
            messagebox.showinfo("Feedback", f"👍 Positive feedback recorded!\nFilename: {self.current_filename}")
        except Exception as e:
            print(f"❌ Thumbs up feedback error: {e}")

    

    def thumbs_down_feedback(self):
        """User wants to see alternative filenames without negative reward"""
        if not self.agent_enabled:
            return

        try:
            self.agent_status_label.config(text="🤖 👎 Generating Options...", fg="orange")

            # Generate 3 alternative filename suggestions
            content = self.text.get("1.0", tk.END).strip()
            suggestions = self.generate_filename_alternatives(content)

            # Show options dialog
            self.show_filename_options_dialog(suggestions)

        except Exception as e:
            print(f"❌ Thumbs down feedback error: {e}")

    def generate_filename_alternatives(self, content):
        """Generate 3 alternative filename suggestions"""
        if not content or not self.filename_env:
            return ["alternative_1", "alternative_2", "alternative_3"]

        alternatives = []
        used_strategies = set()

        # Try to get 3 different strategies
        for _ in range(3):
            best_filename = None
            best_confidence = 0
            best_strategy = None

            # Try all strategies and pick best unused one
            for action_id, strategy_func in self.filename_env.actions.items():
                if action_id in used_strategies:
                    continue

                try:
                    state = self.filename_env.analyze_content(content)
                    filename, confidence = strategy_func(content, state)
                    if filename and confidence > best_confidence:
                        best_filename = filename
                        best_confidence = confidence
                        best_strategy = action_id
                except:
                    continue

            if best_filename and best_strategy is not None:
                alternatives.append(best_filename)
                used_strategies.add(best_strategy)
            else:
                # Fallback
                alternatives.append(f"option_{len(alternatives) + 1}")

        # Ensure we have exactly 3 alternatives
        while len(alternatives) < 3:
            alternatives.append(f"fallback_{len(alternatives) + 1}")

        return alternatives[:3]

    def show_filename_options_dialog(self, suggestions):
        """Show dialog with 3 filename suggestions and custom input"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose Better Filename")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))

        # Header
        header_label = tk.Label(dialog, text="👎 Choose a better filename:", font=("Arial", 12, "bold"))
        header_label.pack(pady=10)

        # Variable to store selection
        self.filename_choice = tk.StringVar()

        # Agent suggestions
        suggestions_frame = tk.Frame(dialog)
        suggestions_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(suggestions_frame, text="🤖 Agent Byte suggestions:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        for i, suggestion in enumerate(suggestions):
            rb = tk.Radiobutton(
                suggestions_frame,
                text=f"{i + 1}. {suggestion}",
                variable=self.filename_choice,
                value=f"suggestion_{i}",
                font=("Arial", 10),
                wraplength=350
            )
            rb.pack(anchor=tk.W, pady=2)

        # Custom input section
        custom_frame = tk.Frame(dialog)
        custom_frame.pack(pady=10, padx=20, fill=tk.X)

        tk.Label(custom_frame, text="✏️ Or type your own:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        custom_input_frame = tk.Frame(custom_frame)
        custom_input_frame.pack(fill=tk.X, pady=5)

        self.custom_filename_entry = tk.Entry(custom_input_frame, font=("Arial", 10))
        self.custom_filename_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Bind Enter key to apply choice
        self.custom_filename_entry.bind("<Return>", lambda event: apply_choice())

        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)

        def apply_choice():
            custom_name = self.custom_filename_entry.get().strip()
            if custom_name:  # Prioritize non-empty custom input
                self.apply_filename_choice(custom_name, "user_custom", 0.0)
                dialog.destroy()
            else:
                choice = self.filename_choice.get()
                if choice.startswith("suggestion_"):
                    # User chose one of agent's suggestions
                    suggestion_index = int(choice.split("_")[1])
                    new_filename = suggestions[suggestion_index]
                    self.apply_filename_choice(new_filename, "agent_suggestion", 1.5)
                    dialog.destroy()
                else:
                    messagebox.showwarning("No Selection", "Please select a suggestion or enter a custom name.")
                    return

        def cancel_choice():
            dialog.destroy()
            self.agent_status_label.config(text="🤖 Ready", fg="green")

        tk.Button(button_frame, text="Apply", command=apply_choice, bg="lightgreen", font=("Arial", 10)).pack(
            side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_choice, bg="lightcoral", font=("Arial", 10)).pack(
            side=tk.LEFT, padx=5)

        # Set focus to custom entry field for immediate typing
        self.custom_filename_entry.focus_set()

        # Set default radiobutton selection
        if suggestions:
            self.filename_choice.set("suggestion_0")

        # Store suggestions for later use
        self.current_suggestions = suggestions

    def regenerate_filename(self):
        """Generate a new filename after negative feedback"""
        if not self.agent_enabled:
            return

        content = self.text.get("1.0", tk.END).strip()
        if content:
            self.agent_status_label.config(text="🤖 Retrying...", fg="blue")
            self.update_filename_with_agent(content)

    def give_agent_feedback(self, reward, feedback_type):
        """Provide feedback to the agent"""
        if not self.agent_enabled or not hasattr(self, 'last_agent_state'):
            return

        try:
            # Create next state (same as current for filename generation)
            next_state = self.last_agent_state

            # Agent learns from the feedback
            self.agent.learn(reward=reward, next_state=next_state, done=True)

            # Log the feedback
            self.log_filename_feedback(feedback_type, reward)

            print(f"🎯 Agent feedback: {feedback_type} (reward: {reward})")

        except Exception as e:
            print(f"❌ Agent feedback error: {e}")

    # Complete enhanced logging combining both versions - add to dumpedit_ai.py

    def log_filename_feedback(self, feedback_type, reward):
        """Enhanced logging with content analysis AND smart memory management"""
        try:
            content = self.text.get("1.0", tk.END).strip()

            # Analyze content type using environment
            state = self.filename_env.analyze_content(content)
            content_type_scores = state[5:12]  # Content type detection features

            # Determine dominant content type
            content_types = list(self.filename_env.content_patterns.keys())
            dominant_type_idx = np.argmax(content_type_scores)
            dominant_type = content_types[dominant_type_idx] if dominant_type_idx < len(content_types) else "unknown"
            confidence = float(content_type_scores[dominant_type_idx])

            # Enhanced log entry with content analysis
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'filename': self.current_filename,
                'feedback_type': feedback_type,
                'reward': reward,
                'strategy_used': getattr(self, 'last_strategy_used', 'unknown'),
                'strategy_confidence': getattr(self, 'last_confidence', 0.0),
                'content_preview': content[:100],  # First 100 chars
                'content_length': len(content),
                'content_lines': len(content.split('\n')),
                'content_words': len(content.split()),

                # Content type analysis
                'content_type_detected': dominant_type,
                'content_type_confidence': confidence,
                'content_type_scores': {
                    content_types[i]: float(score)
                    for i, score in enumerate(content_type_scores)
                    if i < len(content_types)
                },

                # Pattern recognition results
                'patterns_found': self._extract_content_patterns(content),

                # Learning context
                'agent_training_steps': self.agent.training_steps if self.agent else 0,
                'agent_win_rate': (self.agent.wins / max(1, self.agent.games_played)) * 100 if self.agent else 0,
                'session_match_count': self.agent.games_played if self.agent else 0,

                # NEW: Importance scoring for smart memory management
                'importance_score': self._calculate_importance_score(reward, confidence, dominant_type)
            }

            # Load existing logs
            log_file = "filename_training.json"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            # SMART MEMORY MANAGEMENT - Replace simple truncation
            logs = self._smart_memory_curation(logs)

            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)

            # Log to Agent Byte's knowledge system for transfer learning
            self._log_to_agent_knowledge(log_entry)

            print(f"📚 Logged: {dominant_type} content, importance: {log_entry['importance_score']:.2f}")

        except Exception as e:
            print(f"❌ Enhanced logging error: {e}")

    def _calculate_importance_score(self, reward, confidence, content_type):
        """Calculate importance score for memory retention"""
        base_score = 0.5

        # High reward experiences are important (both positive and negative)
        reward_factor = min(abs(reward) / 2.0, 1.0)

        # High confidence detections are important
        confidence_factor = confidence

        # Rare content types are more important to remember
        rarity_factor = self._get_content_type_rarity(content_type)

        # Extreme feedback is important (great success or major failure)
        extreme_factor = 0.0
        if reward > 1.5:  # Great success
            extreme_factor = 0.3
        elif reward < -1.5:  # Major failure
            extreme_factor = 0.4  # Failures slightly more important to remember

        importance = base_score + (reward_factor * 0.3) + (confidence_factor * 0.2) + (
                    rarity_factor * 0.3) + extreme_factor
        return min(importance, 1.0)

    def _get_content_type_rarity(self, content_type):
        """Calculate how rare this content type is in recent history"""
        try:
            log_file = "filename_training.json"
            if not os.path.exists(log_file):
                return 0.8  # New content type is rare

            with open(log_file, 'r') as f:
                logs = json.load(f)

            if not logs:
                return 0.8

            # Count recent content types (last 50 entries)
            recent_logs = logs[-50:]
            content_type_counts = {}

            for entry in recent_logs:
                ct = entry.get('content_type_detected', 'unknown')
                content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

            total_recent = len(recent_logs)
            current_frequency = content_type_counts.get(content_type, 0) / total_recent

            # Rarity = 1.0 - frequency (rare types get higher scores)
            rarity = max(0.0, 1.0 - (current_frequency * 2))  # Scale to make rare types stand out
            return rarity

        except:
            return 0.5  # Default moderate rarity

    def _smart_memory_curation(self, logs):
        """Intelligently curate memory to prevent catastrophic forgetting"""

        if len(logs) <= 150:  # Under limit, keep everything
            return logs

        # Separate into categories
        high_importance = []  # importance >= 0.8
        medium_importance = []  # 0.5 <= importance < 0.8
        low_importance = []  # importance < 0.5

        for entry in logs:
            importance = entry.get('importance_score', 0.5)
            if importance >= 0.8:
                high_importance.append(entry)
            elif importance >= 0.5:
                medium_importance.append(entry)
            else:
                low_importance.append(entry)

        # SMART RETENTION STRATEGY
        curated_logs = []

        # 1. ALWAYS keep high importance entries (critical learnings)
        curated_logs.extend(high_importance[-50:])  # Last 50 high importance

        # 2. Keep diverse medium importance entries
        medium_by_content_type = {}
        for entry in medium_importance:
            content_type = entry.get('content_type_detected', 'unknown')
            if content_type not in medium_by_content_type:
                medium_by_content_type[content_type] = []
            medium_by_content_type[content_type].append(entry)

        # Keep 5 most recent per content type from medium importance
        for content_type, entries in medium_by_content_type.items():
            curated_logs.extend(entries[-5:])

        # 3. Keep some recent low importance for context
        curated_logs.extend(low_importance[-20:])

        # 4. Ensure we have examples of each content type
        content_type_coverage = set()
        for entry in curated_logs:
            content_type_coverage.add(entry.get('content_type_detected', 'unknown'))

        # Add missing content types from anywhere in history
        all_content_types = set()
        for entry in logs:
            all_content_types.add(entry.get('content_type_detected', 'unknown'))

        missing_types = all_content_types - content_type_coverage
        for missing_type in missing_types:
            # Find best example of missing type
            examples = [e for e in logs if e.get('content_type_detected') == missing_type]
            if examples:
                # Get the highest importance example
                best_example = max(examples, key=lambda x: x.get('importance_score', 0))
                curated_logs.append(best_example)

        # Sort by timestamp to maintain chronological order
        curated_logs.sort(key=lambda x: x.get('timestamp', ''))

        # Final limit check
        if len(curated_logs) > 200:
            # Priority: keep recent high importance + diverse content types
            curated_logs = curated_logs[-200:]

        print(f"🧠 Memory curated: {len(logs)} → {len(curated_logs)} entries")
        print(f"   Content types preserved: {len(set(e.get('content_type_detected') for e in curated_logs))}")

        return curated_logs

    def _extract_content_patterns(self, content):
        """Extract specific patterns found in content for learning"""
        patterns_found = {}

        for content_type, patterns in self.filename_env.content_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if found:
                    matches.extend(found[:3])  # Limit to first 3 matches per pattern

            if matches:
                patterns_found[content_type] = matches

        return patterns_found

    def _log_to_agent_knowledge(self, log_entry):
        """Log filename learning to Agent Byte's knowledge system for transfer"""
        if not self.agent or not self.agent.dual_brain:
            return

        try:
            # Create transferable lesson based on success/failure
            content_type = log_entry['content_type_detected']
            reward = log_entry['reward']
            strategy = log_entry['strategy_used']
            filename = log_entry['filename']

            if reward > 1.0:  # Success
                lesson = f"Content type '{content_type}' responds well to strategy {strategy} - generated '{filename}' successfully"
                self.agent.dual_brain.knowledge.add_lesson("filename_generation", lesson)

                # Add strategy if very successful
                if reward > 2.0:
                    strategy_text = f"For {content_type} content, use strategy {strategy} approach for high user satisfaction"
                    self.agent.dual_brain.knowledge.add_strategy("filename_generation", strategy_text)

            elif reward < -1.0:  # Failure
                lesson = f"Avoid using strategy {strategy} for {content_type} content - user rejected '{filename}'"
                self.agent.dual_brain.knowledge.add_lesson("filename_generation", lesson)

            # Log pattern recognition insights
            patterns = log_entry.get('patterns_found', {})
            if patterns:
                for pattern_type, matches in patterns.items():
                    if matches:
                        insight = f"Content with {pattern_type} patterns: {matches[:2]} - useful for filename generation"
                        self.agent.dual_brain.knowledge.add_lesson("content_analysis", insight)

            print(f"📚 Logged transfer learning data: {content_type} content analysis")

        except Exception as e:
            print(f"❌ Transfer learning logging error: {e}")

    def generate_learning_analysis_report(self):
        """Generate analysis of what Agent Byte has learned for transfer to other environments"""
        try:
            log_file = "filename_training.json"
            if not os.path.exists(log_file):
                return "No learning data available"

            with open(log_file, 'r') as f:
                logs = json.load(f)

            if not logs:
                return "No learning entries found"

            # Analyze content type performance
            content_type_performance = {}
            strategy_effectiveness = {}
            pattern_insights = {}
            importance_distribution = {'high': 0, 'medium': 0, 'low': 0}

            for entry in logs:
                content_type = entry.get('content_type_detected', 'unknown')
                reward = entry.get('reward', 0)
                strategy = entry.get('strategy_used', 'unknown')
                patterns = entry.get('patterns_found', {})
                importance = entry.get('importance_score', 0.5)

                # Track importance distribution
                if importance >= 0.8:
                    importance_distribution['high'] += 1
                elif importance >= 0.5:
                    importance_distribution['medium'] += 1
                else:
                    importance_distribution['low'] += 1

                # Track content type success
                if content_type not in content_type_performance:
                    content_type_performance[content_type] = {'rewards': [], 'count': 0}
                content_type_performance[content_type]['rewards'].append(reward)
                content_type_performance[content_type]['count'] += 1

                # Track strategy effectiveness
                strategy_key = f"{strategy}_for_{content_type}"
                if strategy_key not in strategy_effectiveness:
                    strategy_effectiveness[strategy_key] = {'rewards': [], 'count': 0}
                strategy_effectiveness[strategy_key]['rewards'].append(reward)
                strategy_effectiveness[strategy_key]['count'] += 1

                # Track pattern insights
                for pattern_type, matches in patterns.items():
                    if matches and pattern_type not in pattern_insights:
                        pattern_insights[pattern_type] = {'examples': set(), 'success_rate': []}
                    if matches:
                        pattern_insights[pattern_type]['examples'].update(matches[:2])
                        pattern_insights[pattern_type]['success_rate'].append(reward > 0)

            # Generate report
            report = "🧠 AGENT BYTE LEARNING ANALYSIS - TRANSFER READY\n"
            report += "=" * 60 + "\n\n"

            report += "📊 Content Type Performance:\n"
            for content_type, data in content_type_performance.items():
                avg_reward = sum(data['rewards']) / len(data['rewards'])
                success_rate = sum(1 for r in data['rewards'] if r > 0) / len(data['rewards']) * 100
                report += f"   {content_type}: {avg_reward:.2f} avg reward, {success_rate:.1f}% success ({data['count']} samples)\n"

            report += f"\n🎯 Most Effective Strategies:\n"
            sorted_strategies = sorted(strategy_effectiveness.items(),
                                       key=lambda x: sum(x[1]['rewards']) / len(x[1]['rewards']),
                                       reverse=True)
            for strategy_key, data in sorted_strategies[:5]:
                avg_reward = sum(data['rewards']) / len(data['rewards'])
                report += f"   {strategy_key}: {avg_reward:.2f} avg reward ({data['count']} uses)\n"

            report += f"\n🔍 Pattern Recognition Insights:\n"
            for pattern_type, data in pattern_insights.items():
                success_rate = sum(data['success_rate']) / len(data['success_rate']) * 100 if data[
                    'success_rate'] else 0
                examples = list(data['examples'])[:3]
                report += f"   {pattern_type}: {success_rate:.1f}% success, examples: {examples}\n"

            # NEW: Memory management insights
            report += f"\n🧠 Smart Memory Management:\n"
            report += f"   High Importance: {importance_distribution['high']} entries (critical learnings)\n"
            report += f"   Medium Importance: {importance_distribution['medium']} entries (diverse examples)\n"
            report += f"   Low Importance: {importance_distribution['low']} entries (recent context)\n"
            report += f"   Total Entries: {len(logs)} (intelligently curated)\n"

            report += f"\n🚀 Transfer Learning Recommendations:\n"
            report += f"   • Agent has processed {len(logs)} filename generation tasks\n"
            report += f"   • Content analysis skills are transferable to other text processing environments\n"
            report += f"   • Pattern recognition abilities can be applied to document classification\n"
            report += f"   • Strategy selection logic is applicable to other decision-making contexts\n"
            report += f"   • Smart memory prevents catastrophic forgetting across content types\n"

            return report

        except Exception as e:
            return f"Error generating analysis: {e}"

    def update_window_title(self):
        """Update window title to show current filename"""
        self.root.title(f"DumpEdit - {self.current_filename}")

    def load_file(self):
        """Load existing file"""
        filepath = os.path.join(SAVE_FOLDER, self.current_filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                self.text.insert("1.0", content)
                self.backups.append(content)
                self.current_index = 0
                self.last_content_hash = hash(content.strip())

    def add_backup(self, content):
        """Add content to backup history"""
        if self.backups and self.backups[-1] == content:
            return
        if len(self.backups) >= BACKUP_COUNT:
            self.backups.pop(0)
        self.backups.append(content)
        self.current_index = len(self.backups) - 1

    def save_file(self, content):
        """Save content to current file"""
        filepath = os.path.join(SAVE_FOLDER, self.current_filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def undo(self):
        """Undo to previous version"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_backup()

    def redo(self):
        """Redo to next version"""
        if self.current_index < len(self.backups) - 1:
            self.current_index += 1
            self.load_backup()

    def load_backup(self):
        """Load backup content"""
        content = self.backups[self.current_index]
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.save_file(content)

    def clear_screen(self):
        if messagebox.askyesno("Clear Screen", "Are you sure you want to clear everything?"):
            current = self.text.get("1.0", tk.END).strip()
            if current:
                self.add_backup(current)
            self.text.delete("1.0", tk.END)
            self.save_file("")  # Save empty content to current filename
            self.last_content_hash = None
            if self.agent_enabled:
                self.agent_status_label.config(text="🤖 Ready", fg="green")

    def select_all(self, event=None):
        """Select all text"""
        self.text.tag_add("sel", "1.0", "end")
        return "break"

    def show_right_click_menu(self, event):
        """Show right-click context menu"""
        try:
            self.right_click_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.right_click_menu.grab_release()
            self.right_click_menu.unpost()

    def open_folder(self):
        """Open the save folder"""
        if platform.system() == "Windows":
            os.startfile(SAVE_FOLDER)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", SAVE_FOLDER])
        else:  # Linux and others
            subprocess.Popen(["xdg-open", SAVE_FOLDER])

    def on_closing(self):
        """Handle application closing"""
        if self.agent_enabled and self.agent:
            try:
                # End agent session
                content = self.text.get("1.0", tk.END).strip()
                if content:
                    # Determine if session was successful (no recent negative feedback)
                    final_stats = self.agent.get_stats()
                    self.agent.end_match("User", final_stats)

                # Save agent brain
                self.agent.save_brain()
                print("💾 Agent Byte saved successfully")

            except Exception as e:
                print(f"❌ Error saving agent on exit: {e}")

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DumpEditWithAgent(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()