import tkinter as tk
from tkinter import messagebox

class TextMarker(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Text Marker Demo")
        self.geometry("400x300")

        self.text = "This is a sample sentence for marking parts."
        self.marked_parts = []
        self.stored_parts = []

        self.create_widgets()

    def create_widgets(self):
        self.text_widget = tk.Text(self, height=5, width=40, wrap=tk.WORD)
        self.text_widget.pack(pady=10)
        self.text_widget.insert(tk.END, self.text)
        self.text_widget.config(state=tk.NORMAL)  # Allow selection

        # Bind mouse events
        self.text_widget.bind("<ButtonRelease-1>", self.mark_selection)
        self.text_widget.bind("<Button-3>", self.store_marked_text)

        self.activate_button = tk.Button(self, text="Activate Backend", command=self.activate_backend)
        self.activate_button.pack(pady=5)

    def mark_selection(self, event):
        try:
            start = self.text_widget.index("sel.first")
            end = self.text_widget.index("sel.last")
            selected_text = self.text_widget.get(start, end)
            
            # Add new tag
            tag = f"highlight_{len(self.marked_parts)}"
            self.text_widget.tag_add(tag, start, end)
            self.text_widget.tag_config(tag, background="yellow")
            
            self.marked_parts.append((selected_text, start, end, tag))
            
        except tk.TclError:
            pass  # No selection made, do nothing

    def store_marked_text(self, event):
        click_position = self.text_widget.index(f"@{event.x},{event.y}")
        for text, start, end, tag in self.marked_parts:
            if self.text_widget.compare(start, "<=", click_position) and self.text_widget.compare(click_position, "<=", end):
                if text not in self.stored_parts:
                    self.stored_parts.append(text)
                    messagebox.showinfo("Stored", f"Stored text: {text}")
                else:
                    messagebox.showinfo("Info", f"Text '{text}' is already stored.")
                break
        else:
            messagebox.showinfo("Info", "No marked text at this position. Please click on a marked area.")
        return "break"  # Prevent the default context menu

    def activate_backend(self):
        if not self.stored_parts:
            messagebox.showinfo("Info", "No stored parts. Please mark and store some parts of the text before activating the backend.")
        else:
            # Here you would call your backend function
            stored_text = ", ".join(self.stored_parts)
            message = f"Stored parts: {stored_text}"
            messagebox.showinfo("Backend Activated", message)

if __name__ == "__main__":
    app = TextMarker()
    app.mainloop()