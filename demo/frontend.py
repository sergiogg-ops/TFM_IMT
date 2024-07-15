import tkinter as tk
from tkinter import messagebox

class TextMarker(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Text Marker Demo")
        self.geometry("400x300")

        self.text = "This is a sample sentence for marking parts."
        self.marked_parts = []

        self.create_widgets()

    def create_widgets(self):
        self.text_widget = tk.Text(self, height=5, width=40, wrap=tk.WORD)
        self.text_widget.pack(pady=10)
        self.text_widget.insert(tk.END, self.text)
        self.text_widget.config(state=tk.DISABLED)

        self.mark_button = tk.Button(self, text="Mark Selection", command=self.mark_selection)
        self.mark_button.pack(pady=5)

        self.activate_button = tk.Button(self, text="Activate Backend", command=self.activate_backend)
        self.activate_button.pack(pady=5)

    def mark_selection(self):
        try:
            start = self.text_widget.index("sel.first")
            end = self.text_widget.index("sel.last")
            selected_text = self.text_widget.get(start, end)
            self.marked_parts.append(selected_text)
            self.text_widget.tag_add("highlight", start, end)
            self.text_widget.tag_config("highlight", background="yellow")
        except tk.TclError:
            messagebox.showinfo("Info", "Please select some text to mark.")

    def activate_backend(self):
        if not self.marked_parts:
            messagebox.showinfo("Info", "Please mark some parts of the text before activating the backend.")
        else:
            # Here you would call your backend function
            messagebox.showinfo("Backend Activated", f"Marked parts: {', '.join(self.marked_parts)}")

if __name__ == "__main__":
    app = TextMarker()
    app.mainloop()