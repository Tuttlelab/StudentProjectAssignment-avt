"""Point-and-click front end for Hungarian.py's StudentProjectAssignment.

Packaged with PyInstaller (see Hungarian.spec) into a standalone .exe so
non-Python users can run the assignment without installing anything.
"""
import contextlib
import io
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from Hungarian import StudentProjectAssignment

FILE_TYPES = [("CSV or Excel files", "*.csv;*.xlsx"), ("All files", "*.*")]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Project Assignment")
        self.geometry("720x520")
        self.spa = None

        self._build_file_row("Projects file:", "projects_path", 0)
        self._build_file_row("Student choices file:", "choices_path", 1)

        tk.Button(self, text="Run assignment", command=self.run_assignment).grid(
            row=2, column=0, padx=8, pady=8, sticky="w"
        )
        self.save_button = tk.Button(
            self, text="Save results as...", command=self.save_results, state="disabled"
        )
        self.save_button.grid(row=2, column=1, padx=8, pady=8, sticky="w")

        self.output = scrolledtext.ScrolledText(self, state="disabled", wrap="word")
        self.output.grid(row=3, column=0, columnspan=3, padx=8, pady=8, sticky="nsew")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

    def _build_file_row(self, label, attr_name, row):
        tk.Label(self, text=label).grid(row=row, column=0, padx=8, pady=4, sticky="w")
        var = tk.StringVar()
        setattr(self, attr_name, var)
        tk.Entry(self, textvariable=var, width=60).grid(
            row=row, column=1, padx=4, pady=4, sticky="we"
        )
        tk.Button(
            self, text="Browse...", command=lambda: self._browse(var)
        ).grid(row=row, column=2, padx=8, pady=4)

    def _browse(self, var):
        path = filedialog.askopenfilename(filetypes=FILE_TYPES)
        if path:
            var.set(path)

    def _write_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state="disabled")

    def run_assignment(self):
        projects = self.projects_path.get().strip()
        choices = self.choices_path.get().strip()
        if not projects or not choices:
            messagebox.showerror("Missing files", "Please choose both a projects file and a student choices file.")
            return

        self.save_button.configure(state="disabled")
        self.spa = None
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                spa = StudentProjectAssignment(projects, choices)
                spa.Hungarian()
                spa.summary()
                try:
                    spa.capacity_check()
                except AssertionError:
                    print("\nWarning: a supervisor's total capacity was exceeded — check the capacities in the projects file.")
        except FileNotFoundError as exc:
            self._write_output(captured.getvalue())
            messagebox.showerror("File not found", str(exc))
            return
        except AssertionError:
            self._write_output(captured.getvalue())
            messagebox.showerror(
                "Invalid project capacities",
                "Every project must have a capacity greater than 0 — check the projects file.",
            )
            return
        except ValueError as exc:
            self._write_output(captured.getvalue())
            messagebox.showerror("Could not read file", str(exc))
            return
        except Exception as exc:
            self._write_output(captured.getvalue())
            messagebox.showerror("Something went wrong", str(exc))
            return

        self.spa = spa
        self._write_output(captured.getvalue())
        self.save_button.configure(state="normal")

    def save_results(self):
        if self.spa is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="Project-assignment.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.spa.save(output_path=path)
        except Exception as exc:
            messagebox.showerror("Could not save file", str(exc))
            return
        messagebox.showinfo("Saved", f"Results saved to:\n{path}")


if __name__ == "__main__":
    App().mainloop()
