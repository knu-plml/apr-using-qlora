#!/usr/bin/env python3
"""
Plausible Patch Validator GUI

A Tkinter-based GUI for human evaluation of plausible patches.
Allows evaluators to mark patches as correct, incorrect, or pending.
"""

import csv
import re
import tkinter as tk
from tkinter import ttk, messagebox, font
from collections import defaultdict
from pathlib import Path


CSV_PATH = "./evaluations.csv"


class PatchValidatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Plausible Patch Validator")
        self.root.geometry("1400x900")

        # Data storage
        self.data = []  # List of all rows from CSV
        self.tree_items = {}  # Maps tree item id to data index
        self.current_index = None  # Current data index being viewed

        # Hierarchical structure: problem_id -> model -> list of (data_index, patch_index)
        self.hierarchy = defaultdict(lambda: defaultdict(list))

        # Load data
        self.load_data()
        self.build_hierarchy()

        # Build UI
        self.setup_ui()

        # Select first item
        self.select_first_unevaluated()

    def load_data(self):
        """Load CSV data"""
        if not Path(CSV_PATH).exists():
            messagebox.showerror("Error", f"CSV file not found: {CSV_PATH}\nPlease run convert_to_csv.py first.")
            self.root.destroy()
            return

        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)

        if not self.data:
            messagebox.showerror("Error", "CSV file is empty!")
            self.root.destroy()

    def build_hierarchy(self):
        """Build hierarchical structure from data"""
        self.hierarchy.clear()
        for idx, row in enumerate(self.data):
            problem_id = row["problem_id"]
            model = row["model"]
            patch_index = int(row["patch_index"])
            self.hierarchy[problem_id][model].append((idx, patch_index))

        # Sort patches within each model
        for problem_id in self.hierarchy:
            for model in self.hierarchy[problem_id]:
                self.hierarchy[problem_id][model].sort(key=lambda x: x[1])

    def setup_ui(self):
        """Setup the main UI"""
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Navigation
        left_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(left_frame, weight=1)

        # Navigation label
        nav_label = ttk.Label(left_frame, text="Navigation", font=("TkDefaultFont", 12, "bold"))
        nav_label.pack(pady=(5, 5))

        # Progress label
        self.progress_label = ttk.Label(left_frame, text="Progress: 0/0")
        self.progress_label.pack(pady=(0, 5))

        # Filter frame
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var,
                                     values=["all", "unevaluated", "correct", "incorrect", "pending", "skip"],
                                     state="readonly", width=12)
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", self.on_filter_change)

        # Treeview for navigation
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(tree_frame, selectmode="browse")
        self.tree.heading("#0", text="Problems / Models / Patches")

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Right panel - Code viewer and buttons
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        # Info frame
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.info_label = ttk.Label(info_frame, text="Select a patch to view", font=("TkDefaultFont", 11))
        self.info_label.pack(anchor=tk.W)

        self.benchmark_label = ttk.Label(info_frame, text="")
        self.benchmark_label.pack(anchor=tk.W)

        # Code viewer
        code_frame = ttk.LabelFrame(right_frame, text="Code View")
        code_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create text widget with custom fonts
        self.code_text = tk.Text(code_frame, wrap=tk.NONE, font=("Consolas", 11), state=tk.DISABLED)

        code_scroll_y = ttk.Scrollbar(code_frame, orient=tk.VERTICAL, command=self.code_text.yview)
        code_scroll_x = ttk.Scrollbar(code_frame, orient=tk.HORIZONTAL, command=self.code_text.xview)
        self.code_text.configure(yscrollcommand=code_scroll_y.set, xscrollcommand=code_scroll_x.set)

        code_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        code_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.code_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for syntax highlighting
        self.code_text.tag_configure("buggy", foreground="red", font=("Consolas", 11, "bold"))
        self.code_text.tag_configure("gold", foreground="green", font=("Consolas", 11, "bold"))
        self.code_text.tag_configure("patch", foreground="blue", font=("Consolas", 11, "bold"))
        self.code_text.tag_configure("normal", foreground="black")
        self.code_text.tag_configure("label", foreground="gray", font=("Consolas", 10, "italic"))

        # Legend
        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(legend_frame, text="Legend: ", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="Buggy Line", foreground="red").pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="Gold Answer", foreground="green").pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="Model Patch", foreground="blue").pack(side=tk.LEFT, padx=5)

        # Button frame
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        # Status indicator
        self.status_label = ttk.Label(button_frame, text="Current: Not evaluated", font=("TkDefaultFont", 10))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Evaluation buttons
        self.btn_correct = ttk.Button(button_frame, text="Correct (C)", command=lambda: self.evaluate("correct"))
        self.btn_correct.pack(side=tk.RIGHT, padx=5)

        self.btn_incorrect = ttk.Button(button_frame, text="Incorrect (X)", command=lambda: self.evaluate("incorrect"))
        self.btn_incorrect.pack(side=tk.RIGHT, padx=5)

        self.btn_pending = ttk.Button(button_frame, text="Pending (Z)", command=lambda: self.evaluate("pending"))
        self.btn_pending.pack(side=tk.RIGHT, padx=5)

        self.btn_skip = ttk.Button(button_frame, text="Skip (S)", command=lambda: self.evaluate("skip"))
        self.btn_skip.pack(side=tk.RIGHT, padx=5)

        # Keyboard shortcuts
        self.root.bind("c", lambda e: self.evaluate("correct"))
        self.root.bind("x", lambda e: self.evaluate("incorrect"))
        self.root.bind("z", lambda e: self.evaluate("pending"))
        self.root.bind("s", lambda e: self.evaluate("skip"))
        self.root.bind("<Up>", lambda e: self.navigate_prev())
        self.root.bind("<Down>", lambda e: self.navigate_next())
        self.root.bind("<Left>", lambda e: self.navigate_prev())
        self.root.bind("<Right>", lambda e: self.navigate_next())

        # Populate tree
        self.populate_tree()

    def get_status_icon(self, evaluation: str) -> str:
        """Get status icon for evaluation"""
        if evaluation == "correct":
            return "\u2713"  # checkmark
        elif evaluation == "incorrect":
            return "\u2717"  # X
        elif evaluation == "pending":
            return "?"
        elif evaluation == "skip":
            return "\u21b7"  # skip arrow
        else:
            return "\u25cb"  # empty circle

    def populate_tree(self):
        """Populate the treeview with hierarchical data"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree_items.clear()

        filter_value = self.filter_var.get()

        # Sort problem_ids
        problem_ids = sorted(self.hierarchy.keys())

        for problem_id in problem_ids:
            # Check if this problem has any matching patches
            has_matching = False
            for model in self.hierarchy[problem_id]:
                for data_idx, patch_idx in self.hierarchy[problem_id][model]:
                    eval_status = self.data[data_idx]["evaluation"]
                    if self._matches_filter(eval_status, filter_value):
                        has_matching = True
                        break
                if has_matching:
                    break

            if not has_matching:
                continue

            # Truncate long problem_id for display
            display_id = problem_id if len(problem_id) <= 40 else problem_id[:37] + "..."

            # Count evaluated patches for this problem
            total = 0
            evaluated = 0
            for model in self.hierarchy[problem_id]:
                for data_idx, _ in self.hierarchy[problem_id][model]:
                    total += 1
                    if self.data[data_idx]["evaluation"]:
                        evaluated += 1

            problem_node = self.tree.insert("", "end", text=f"{display_id} [{evaluated}/{total}]",
                                            open=False, tags=("problem",))

            # Sort models
            models = sorted(self.hierarchy[problem_id].keys())

            for model in models:
                patches = self.hierarchy[problem_id][model]

                # Check if this model has any matching patches
                model_patches = []
                for data_idx, patch_idx in patches:
                    eval_status = self.data[data_idx]["evaluation"]
                    if self._matches_filter(eval_status, filter_value):
                        model_patches.append((data_idx, patch_idx, eval_status))

                if not model_patches:
                    continue

                # Count evaluated for this model
                model_total = len(patches)
                model_evaluated = sum(1 for d, _, _ in model_patches if self.data[d]["evaluation"])

                model_node = self.tree.insert(problem_node, "end",
                                              text=f"{model} [{model_evaluated}/{model_total}]",
                                              open=False, tags=("model",))

                for data_idx, patch_idx, eval_status in model_patches:
                    icon = self.get_status_icon(eval_status)
                    patch_node = self.tree.insert(model_node, "end",
                                                   text=f"{icon} patch {patch_idx}",
                                                   tags=("patch",))
                    self.tree_items[patch_node] = data_idx

        self.update_progress()

    def _matches_filter(self, evaluation: str, filter_value: str) -> bool:
        """Check if evaluation matches the current filter"""
        if filter_value == "all":
            return True
        elif filter_value == "unevaluated":
            return evaluation == ""
        else:
            return evaluation == filter_value

    def on_filter_change(self, event=None):
        """Handle filter change"""
        self.populate_tree()

    def on_tree_select(self, event=None):
        """Handle tree selection"""
        selection = self.tree.selection()
        if not selection:
            return

        item_id = selection[0]
        if item_id in self.tree_items:
            self.current_index = self.tree_items[item_id]
            self.display_current()

    def select_first_unevaluated(self):
        """Select the first unevaluated item"""
        for idx, row in enumerate(self.data):
            if row["evaluation"] == "":
                self.current_index = idx
                self.display_current()
                self.select_tree_item(idx)
                return

        # If all evaluated, select first item
        if self.data:
            self.current_index = 0
            self.display_current()
            self.select_tree_item(0)

    def select_tree_item(self, data_idx: int):
        """Select the tree item corresponding to data index"""
        for item_id, idx in self.tree_items.items():
            if idx == data_idx:
                # Open parent nodes
                parent = self.tree.parent(item_id)
                while parent:
                    self.tree.item(parent, open=True)
                    parent = self.tree.parent(parent)

                self.tree.selection_set(item_id)
                self.tree.see(item_id)
                return

    def display_current(self):
        """Display the current patch"""
        if self.current_index is None:
            return

        row = self.data[self.current_index]

        # Update info labels
        self.info_label.config(text=f"Problem: {row['problem_id']}")
        self.benchmark_label.config(text=f"Benchmark: {row['benchmark']} | Model: {row['model']} | Patch: {row['patch_index']}")

        # Update status
        eval_status = row["evaluation"]
        if eval_status:
            self.status_label.config(text=f"Current: {eval_status.upper()}")
        else:
            self.status_label.config(text="Current: Not evaluated")

        # Display code
        self.display_code(row)

    def display_code(self, row: dict):
        """Display the code with highlighting"""
        self.code_text.config(state=tk.NORMAL)
        self.code_text.delete("1.0", tk.END)

        input_text = row["input"]
        fixed_line = row["fixed_line"]
        patch = row["patch"]

        # Parse input to extract buggy lines and surrounding code
        lines = input_text.split("\n")

        buggy_start = -1
        buggy_end = -1
        fixed_start = -1
        buggy_lines = []

        for i, line in enumerate(lines):
            if "// buggy lines start:" in line:
                buggy_start = i
            elif "// buggy lines end:" in line:
                buggy_end = i
            elif "// fixed lines:" in line:
                fixed_start = i

        # Display code before buggy section
        for i in range(buggy_start):
            self.code_text.insert(tk.END, lines[i] + "\n", "normal")

        # Display buggy lines with highlighting
        if buggy_start >= 0 and buggy_end >= 0:
            self.code_text.insert(tk.END, "    // === BUGGY LINE (Original) ===\n", "label")
            for i in range(buggy_start + 1, buggy_end):
                self.code_text.insert(tk.END, lines[i] + "\n", "buggy")

            self.code_text.insert(tk.END, "    // === GOLD ANSWER (Expected) ===\n", "label")
            if fixed_line:
                # Remove trailing newlines for cleaner display
                fixed_line = fixed_line.rstrip("\n")
                for fl in fixed_line.split("\n"):
                    self.code_text.insert(tk.END, fl + "\n", "gold")
            else:
                self.code_text.insert(tk.END, "(No fixed_line provided)\n", "label")

            self.code_text.insert(tk.END, "    // === MODEL PATCH (To Evaluate) ===\n", "label")
            if patch:
                # Remove trailing newlines for cleaner display
                patch = patch.rstrip("\n")
                for pl in patch.split("\n"):
                    self.code_text.insert(tk.END, pl + "\n", "patch")
            else:
                self.code_text.insert(tk.END, "(Empty patch)\n", "label")

            self.code_text.insert(tk.END, "    // === END OF COMPARISON ===\n", "label")

        # Display code after buggy section (excluding fixed lines section)
        if buggy_end >= 0:
            end_idx = fixed_start if fixed_start >= 0 else len(lines)
            for i in range(buggy_end + 1, end_idx):
                self.code_text.insert(tk.END, lines[i] + "\n", "normal")

        self.code_text.config(state=tk.DISABLED)

    def evaluate(self, evaluation: str):
        """Evaluate the current patch"""
        if self.current_index is None:
            return

        # Update data
        self.data[self.current_index]["evaluation"] = evaluation

        # If correct, skip remaining patches in same problem/model
        if evaluation == "correct":
            self._skip_remaining_in_same_model()

        # Save to CSV
        self.save_csv()

        # Update tree display
        self.populate_tree()

        # Move to next unevaluated item
        self.navigate_next_unevaluated()

    def _skip_remaining_in_same_model(self):
        """Mark remaining unevaluated patches in same problem/model as skip"""
        if self.current_index is None:
            return

        current_row = self.data[self.current_index]
        current_problem = current_row["problem_id"]
        current_model = current_row["model"]

        # Find all patches in same problem/model and mark unevaluated ones as skip
        for data_idx, patch_idx in self.hierarchy[current_problem][current_model]:
            if data_idx != self.current_index and self.data[data_idx]["evaluation"] == "":
                self.data[data_idx]["evaluation"] = "skip"

    def save_csv(self):
        """Save data back to CSV"""
        fieldnames = ["problem_id", "benchmark", "model", "patch_index", "patch", "input", "fixed_line", "evaluation"]

        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)

    def navigate_next_unevaluated(self):
        """Navigate to the next unevaluated item following the hierarchy"""
        if self.current_index is None:
            return

        current_row = self.data[self.current_index]
        current_problem = current_row["problem_id"]
        current_model = current_row["model"]

        # Try to find next in same problem, same model
        found_current = False
        for data_idx, patch_idx in self.hierarchy[current_problem][current_model]:
            if data_idx == self.current_index:
                found_current = True
                continue
            if found_current and self.data[data_idx]["evaluation"] == "":
                self.current_index = data_idx
                self.display_current()
                self.select_tree_item(data_idx)
                return

        # Try next model in same problem
        models = sorted(self.hierarchy[current_problem].keys())
        found_model = False
        for model in models:
            if model == current_model:
                found_model = True
                continue
            if found_model:
                for data_idx, patch_idx in self.hierarchy[current_problem][model]:
                    if self.data[data_idx]["evaluation"] == "":
                        self.current_index = data_idx
                        self.display_current()
                        self.select_tree_item(data_idx)
                        return

        # Try next problem
        problem_ids = sorted(self.hierarchy.keys())
        found_problem = False
        for problem_id in problem_ids:
            if problem_id == current_problem:
                found_problem = True
                continue
            if found_problem:
                for model in sorted(self.hierarchy[problem_id].keys()):
                    for data_idx, patch_idx in self.hierarchy[problem_id][model]:
                        if self.data[data_idx]["evaluation"] == "":
                            self.current_index = data_idx
                            self.display_current()
                            self.select_tree_item(data_idx)
                            return

        # Wrap around to beginning
        for problem_id in problem_ids:
            for model in sorted(self.hierarchy[problem_id].keys()):
                for data_idx, patch_idx in self.hierarchy[problem_id][model]:
                    if self.data[data_idx]["evaluation"] == "":
                        self.current_index = data_idx
                        self.display_current()
                        self.select_tree_item(data_idx)
                        return

        # All evaluated!
        messagebox.showinfo("Complete", "All patches have been evaluated!")

    def navigate_prev(self):
        """Navigate to previous item"""
        if self.current_index is None or self.current_index == 0:
            return

        # Find previous item in sorted order
        all_indices = []
        for problem_id in sorted(self.hierarchy.keys()):
            for model in sorted(self.hierarchy[problem_id].keys()):
                for data_idx, _ in self.hierarchy[problem_id][model]:
                    all_indices.append(data_idx)

        try:
            current_pos = all_indices.index(self.current_index)
            if current_pos > 0:
                self.current_index = all_indices[current_pos - 1]
                self.display_current()
                self.select_tree_item(self.current_index)
        except ValueError:
            pass

    def navigate_next(self):
        """Navigate to next item"""
        if self.current_index is None:
            return

        # Find next item in sorted order
        all_indices = []
        for problem_id in sorted(self.hierarchy.keys()):
            for model in sorted(self.hierarchy[problem_id].keys()):
                for data_idx, _ in self.hierarchy[problem_id][model]:
                    all_indices.append(data_idx)

        try:
            current_pos = all_indices.index(self.current_index)
            if current_pos < len(all_indices) - 1:
                self.current_index = all_indices[current_pos + 1]
                self.display_current()
                self.select_tree_item(self.current_index)
        except ValueError:
            pass

    def update_progress(self):
        """Update progress label"""
        total = len(self.data)
        evaluated = sum(1 for row in self.data if row["evaluation"])
        correct = sum(1 for row in self.data if row["evaluation"] == "correct")
        incorrect = sum(1 for row in self.data if row["evaluation"] == "incorrect")
        pending = sum(1 for row in self.data if row["evaluation"] == "pending")
        skip = sum(1 for row in self.data if row["evaluation"] == "skip")

        self.progress_label.config(
            text=f"Progress: {evaluated}/{total} | Correct: {correct} | Incorrect: {incorrect} | Pending: {pending} | Skip: {skip}"
        )


def main():
    root = tk.Tk()
    app = PatchValidatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
