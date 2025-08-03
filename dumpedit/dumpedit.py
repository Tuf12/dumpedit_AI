import os
import tkinter as tk
from tkinter import messagebox
import subprocess
import platform

SAVE_FOLDER = "autosaved_notes"
SAVE_FILE = "dumpedit_notes.txt"
BACKUP_COUNT = 5

os.makedirs(SAVE_FOLDER, exist_ok=True)
FULL_PATH = os.path.join(SAVE_FOLDER, SAVE_FILE)

class DumpEdit:
    def __init__(self, root):
        self.root = root
        self.root.title("DumpEdit")

        self.text = tk.Text(root, wrap=tk.WORD)
        self.text.pack(expand=True, fill=tk.BOTH)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X)

        self.undo_btn = tk.Button(btn_frame, text="Undo", command=self.undo)
        self.undo_btn.pack(side=tk.LEFT)

        self.redo_btn = tk.Button(btn_frame, text="Redo", command=self.redo)
        self.redo_btn.pack(side=tk.LEFT)

        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_screen)
        self.clear_btn.pack(side=tk.LEFT)

        self.open_folder_btn = tk.Button(btn_frame, text="Open Folder", command=self.open_folder)
        self.open_folder_btn.pack(side=tk.RIGHT)

        # State
        self.backups = []
        self.current_index = -1

        # Load file if exists
        self.load_file()

        # Events
        self.text.bind("<KeyRelease>", self.on_key)
        self.text.bind("<Control-a>", self.select_all)
        self.text.bind("<Button-3>", self.show_right_click_menu)

        self.right_click_menu = tk.Menu(self.root, tearoff=0)
        self.right_click_menu.add_command(label="Cut", command=lambda: self.text.event_generate("<<Cut>>"))
        self.right_click_menu.add_command(label="Copy", command=lambda: self.text.event_generate("<<Copy>>"))
        self.right_click_menu.add_command(label="Paste", command=lambda: self.text.event_generate("<<Paste>>"))
        self.right_click_menu.add_command(label="Select All", command=self.select_all)

    def on_key(self, _event=None):
        content = self.text.get("1.0", tk.END).strip()
        if not content:
            return
        self.add_backup(content)
        self.save_file(content)

    def load_file(self):
        if os.path.exists(FULL_PATH):
            with open(FULL_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                self.text.insert("1.0", content)
                self.backups.append(content)
                self.current_index = 0

    def add_backup(self, content):
        if self.backups and self.backups[-1] == content:
            return
        if len(self.backups) >= BACKUP_COUNT:
            self.backups.pop(0)
        self.backups.append(content)
        self.current_index = len(self.backups) - 1

    def save_file(self, content):
        with open(FULL_PATH, "w", encoding="utf-8") as f:
            f.write(content)

    def undo(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_backup()

    def redo(self):
        if self.current_index < len(self.backups) - 1:
            self.current_index += 1
            self.load_backup()

    def load_backup(self):
        content = self.backups[self.current_index]
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.save_file(content)

    def clear_screen(self):
        if messagebox.askyesno("Clear Screen", "Are you sure you want to clear everything?"):
            current = self.text.get("1.0", tk.END).strip()
            self.add_backup(current)
            self.text.delete("1.0", tk.END)
            self.on_key()  # Trigger autosave

    def select_all(self, _event=None):
        self.text.tag_add("sel", "1.0", "end")
        return "break"

    def show_right_click_menu(self, event):
        try:
            self.right_click_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.right_click_menu.grab_release()
            self.right_click_menu.unpost()

    def open_folder(self):
        if platform.system() == "Windows":
            os.startfile(SAVE_FOLDER)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", SAVE_FOLDER])
        else:  # Linux and others
            subprocess.Popen(["xdg-open", SAVE_FOLDER])

if __name__ == "__main__":
    root = tk.Tk()
    app = DumpEdit(root)
    root.mainloop()
