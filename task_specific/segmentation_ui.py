import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import PolygonSelector
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import tifffile
import pandas as pd
import numpy as np
import os
import json

class ImageSegmenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fluorescence Image Instance Segmenter")
        self.root.geometry("1200x800")
        
        # Data placeholders
        self.image_path = None
        self.image_data = None
        self.overlay_data = None
        self.lookup_df = None
        self.polygons = [] # List of dicts: {'verts': [...], 'label': '...', 'patch': patch, 'text': text}
        self.current_verts = None
        
        # State
        self.mode = "draw" # "draw" or "overlay_adjust"
        
        # Overlay transform state
        self.overlay_offset_x = 0
        self.overlay_offset_y = 0
        self.overlay_scale = 1.0

        # Custom colormaps
        self.custom_cmaps = {}
        self.create_custom_colormaps()
        
        self.setup_ui()
        
        # Event IDs for dragging
        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.cid_scroll = self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.drag_start_point = None
        self.initial_offset = None

    def create_custom_colormaps(self):
        # Create solid color maps from black -> solid color
        colors_dict = {
            'Red': (1, 0, 0),
            'Green': (0, 1, 0),
            'Blue': (0, 0, 1),
            'Cyan': (0, 1, 1),
            'Magenta': (1, 0, 1),
            'Yellow': (1, 1, 0)
        }
        for name, color in colors_dict.items():
            # Linear fade from transparent/black to solid color
            cmap = LinearSegmentedColormap.from_list(name, [(0, 0, 0), color])
            self.custom_cmaps[name] = cmap

    def setup_ui(self):
        # Main layout
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Side panel for controls
        self.side_panel = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(self.side_panel, weight=0)

        # Matplotlib Figure Area
        self.fig_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.fig_frame, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.ax.axis('off')
        self.selector = None
        self.base_im = None
        self.overlay_im = None

        # Setup side panel tabs
        self.notebook = ttk.Notebook(self.side_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_metadata = ttk.Frame(self.notebook)
        self.tab_display = ttk.Frame(self.notebook)
        self.tab_regions = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_metadata, text='Metadata')
        self.notebook.add(self.tab_display, text='Display')
        self.notebook.add(self.tab_regions, text='Regions')

        self.build_metadata_tab()
        self.build_display_tab()
        self.build_regions_tab()
        self.build_bottom_bar()

        # Keyboard shortcuts
        self.root.bind('<Return>', lambda event: self.confirm_polygon())
        self.root.bind('<Escape>', lambda event: self.clear_current_polygon())
        self.root.bind('<Delete>', lambda event: self.delete_selected_region())

    def build_metadata_tab(self):
        frame = self.tab_metadata
        
        ttk.Label(frame, text="Step 1: Load Base Image", font=('', 10, 'bold')).pack(pady=(10, 5), anchor='w')
        ttk.Button(frame, text="Load .tif Image", command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(frame, text="Sample Info", font=('', 10, 'bold')).pack(pady=5, anchor='w')
        
        ttk.Label(frame, text="Sample ID:").pack(anchor='w', padx=5)
        self.sample_id_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.sample_id_var).pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(frame, text="ROI #:").pack(anchor='w', padx=5)
        self.roi_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.roi_var).pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(frame, text="Last Used Fluor Label:").pack(anchor='w', padx=5)
        self.fluor_label_var = tk.StringVar(value="DAPI")
        self.fluor_combo = ttk.Combobox(frame, textvariable=self.fluor_label_var, values=["DAPI", "GFP", "RFP", "mCherry", "Cy3", "Cy5"])
        self.fluor_combo.pack(fill=tk.X, padx=5, pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(frame, text="Step 3: Load Lookup (Optional)", font=('', 10, 'bold')).pack(pady=5, anchor='w')
        ttk.Button(frame, text="Load Lookup CSV", command=self.load_csv).pack(fill=tk.X, padx=5, pady=5)
        self.csv_label = ttk.Label(frame, text="No CSV loaded", foreground="gray")
        self.csv_label.pack(anchor='w', padx=5)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(frame, text="Step 4: Export Data", font=('', 10, 'bold')).pack(pady=5, anchor='w')
        ttk.Button(frame, text="Export Mask TIF(s)", command=self.export_mask_tif).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(frame, text="Save CSV", command=self.save_data, style='Accent.TButton').pack(fill=tk.X, padx=5, pady=5)
    def build_display_tab(self):
        frame = self.tab_display

        # Base Image Contrast
        ttk.Label(frame, text="Base Image Contrast", font=('', 10, 'bold')).pack(pady=(10, 5), anchor='w')
        
        ttk.Label(frame, text="Min Percentile:").pack(anchor='w', padx=5)
        self.vmin_scale = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_contrast)
        self.vmin_scale.set(0)
        self.vmin_scale.pack(fill=tk.X, padx=5)

        ttk.Label(frame, text="Max Percentile:").pack(anchor='w', padx=5)
        self.vmax_scale = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_contrast)
        self.vmax_scale.set(100)
        self.vmax_scale.pack(fill=tk.X, padx=5)

        ttk.Label(frame, text="Colormap:").pack(anchor='w', padx=5)
        self.cmap_var = tk.StringVar(value="gray")
        cmap_combo = ttk.Combobox(frame, textvariable=self.cmap_var, values=["gray", "inferno", "viridis", "hot", "coolwarm"])
        cmap_combo.pack(fill=tk.X, padx=5)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_contrast())

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Overlay Image
        ttk.Label(frame, text="Overlay Image", font=('', 10, 'bold')).pack(pady=(10, 5), anchor='w')
        ttk.Button(frame, text="Load Overlay .tif", command=self.load_overlay).pack(fill=tk.X, padx=5, pady=5)
        
        # Adjust Mode Toggle Button
        self.adjust_mode_var = tk.StringVar(value="Adjust/Scale Overlay")
        self.adjust_button = ttk.Button(frame, textvariable=self.adjust_mode_var, command=self.toggle_mode, style='Accent.TButton')
        self.adjust_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.overlay_channel_frame = ttk.Frame(frame)
        self.overlay_channel_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(self.overlay_channel_frame, text="Channel:").pack(side=tk.LEFT)
        self.overlay_channel_var = tk.IntVar(value=0)
        self.overlay_channel_spin = ttk.Spinbox(self.overlay_channel_frame, from_=0, to=0, textvariable=self.overlay_channel_var, command=self.update_overlay, width=5)
        self.overlay_channel_spin.pack(side=tk.LEFT, padx=5)

        ttk.Label(frame, text="Opacity:").pack(anchor='w', padx=5)
        self.alpha_scale = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, command=self.update_overlay_display)
        self.alpha_scale.set(0.5)
        self.alpha_scale.pack(fill=tk.X, padx=5)

        ttk.Label(frame, text="Colormap:").pack(anchor='w', padx=5)
        self.overlay_cmap_var = tk.StringVar(value="Green")
        ocmap_combo = ttk.Combobox(frame, textvariable=self.overlay_cmap_var, values=list(self.custom_cmaps.keys()) + ["magma", "hot", "gray", "viridis", "cool", "hsv"])
        ocmap_combo.pack(fill=tk.X, padx=5)
        ocmap_combo.bind("<<ComboboxSelected>>", lambda e: self.update_overlay_display())

        self.show_overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Show Overlay", variable=self.show_overlay_var, command=self.update_overlay_display).pack(anchor='w', padx=5, pady=5)

    def build_regions_tab(self):
        frame = self.tab_regions
        
        ttk.Label(frame, text="Step 2: Draw Regions", font=('', 10, 'bold')).pack(pady=(10, 5), anchor='w')
        ttk.Label(frame, text="Click on image to add vertices.\nPress 'Enter' to confirm region.\nPress 'Esc' to clear current.").pack(anchor='w', padx=5)
        
        ttk.Button(frame, text="Confirm Region (Enter)", command=self.confirm_polygon).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(frame, text="Clear Current (Esc)", command=self.clear_current_polygon).pack(fill=tk.X, padx=5, pady=5)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Region List
        self.tree = ttk.Treeview(frame, columns=('Index', 'Label', 'Vertices'), show='headings', height=10)
        self.tree.heading('Index', text='Idx')
        self.tree.heading('Label', text='Label')
        self.tree.heading('Vertices', text='Verts')
        self.tree.column('Index', width=30)
        self.tree.column('Label', width=100)
        self.tree.column('Vertices', width=50)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(frame, text="Delete Selected (Del)", command=self.delete_selected_region).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(frame, text="Relabel Selected", command=self.relabel_selected_region).pack(fill=tk.X, padx=5, pady=5)

    def build_bottom_bar(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(bottom_frame, text="Ready", font=('', 10))
        self.status_label.pack(side=tk.LEFT)
        
        self.vertex_counter_label = ttk.Label(bottom_frame, text="")
        self.vertex_counter_label.pack(side=tk.LEFT, padx=20)

        # Export buttons moved to Metadata tab
        # Make Save CSV button slightly accented if possible
        style = ttk.Style()
        style.configure('Accent.TButton', font=('', 10, 'bold'))

    def toggle_mode(self):
        if self.mode == "draw":
            self.mode = "overlay_adjust"
            self.adjust_mode_var.set("Finish Adjusting Overlay")
            if self.selector:
                self.selector.set_active(False)
            self.set_status("Overlay Adjust mode: Click+Drag to move, Scroll to scale.", "blue")
        else:
            self.mode = "draw"
            self.adjust_mode_var.set("Adjust/Scale Overlay")
            if self.selector:
                self.selector.set_active(True)
            self.set_status("Draw mode activated.", "blue")

    # --- Mouse Events for Dragging/Scaling ---
    def on_mouse_press(self, event):
        if self.mode != "overlay_adjust" or not self.overlay_im: return
        if event.inaxes != self.ax: return
        if event.button == 1: # Left click to drag
            self.drag_start_point = (event.xdata, event.ydata)
            self.initial_offset = (self.overlay_offset_x, self.overlay_offset_y)

    def on_mouse_drag(self, event):
        if self.mode != "overlay_adjust" or not self.overlay_im or not self.drag_start_point: return
        if event.inaxes != self.ax: return
        
        dx = event.xdata - self.drag_start_point[0]
        dy = event.ydata - self.drag_start_point[1]
        
        self.overlay_offset_x = self.initial_offset[0] + dx
        self.overlay_offset_y = self.initial_offset[1] + dy
        
        self.update_overlay_display()

    def on_mouse_release(self, event):
        if event.button == 1:
            self.drag_start_point = None

    def on_scroll(self, event):
        if self.mode != "overlay_adjust" or not self.overlay_im: return
        if event.inaxes != self.ax: return
        
        scale_factor = 1.1 if event.step > 0 else 1/1.1
        
        # Center zoom around mouse cursor
        x, y = event.xdata, event.ydata
        
        self.overlay_offset_x = x - (x - self.overlay_offset_x) * scale_factor
        self.overlay_offset_y = y - (y - self.overlay_offset_y) * scale_factor
        
        self.overlay_scale *= scale_factor
        self.update_overlay_display()
    # ----------------------------------------

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIFF Images", "*.tif *.tiff")])
        if not file_path:
            return

        try:
            self.image_path = file_path
            self.image_data = tifffile.imread(file_path)
            if self.image_data.ndim > 2:
                self.image_data = self.image_data[0] # Simplification for 2D

            self.ax.clear()
            self.base_im = self.ax.imshow(self.image_data, cmap=self.cmap_var.get())

            self.ax.axis('off')

            self.update_contrast()
            
            # Re-apply overlay if it exists
            if self.overlay_data is not None:
                self.update_overlay_display()

            # Initialize Polygon Selector
            self.selector = PolygonSelector(self.ax, self.on_select, useblit=True, props=dict(color='magenta', linestyle='-', linewidth=2, marker='o', markersize=6))
            if self.mode != "draw":
                self.selector.set_active(False)
                
            self.clear_all_polygons() # Reset polygons for new image
            self.canvas.draw()
            
            # Auto-fill Sample ID from filename if possible
            filename = os.path.basename(file_path).split('.')[0]
            self.sample_id_var.set(filename)
            self.set_status(f"Image loaded: {filename}. Draw region.", "blue")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def load_overlay(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIFF Images", "*.tif *.tiff")])
        if not file_path:
            return

        try:
            full_overlay = tifffile.imread(file_path)
            self.overlay_data_full = full_overlay
            
            # Reset scaling and offset for new overlay
            self.overlay_offset_x = 0
            self.overlay_offset_y = 0
            self.overlay_scale = 1.0
            
            if full_overlay.ndim > 2:
                self.overlay_channel_spin.config(to=full_overlay.shape[0]-1)
                self.overlay_channel_var.set(0)
            else:
                self.overlay_channel_spin.config(to=0)
                self.overlay_channel_var.set(0)
                
            self.update_overlay()
            self.set_status(f"Overlay loaded: {os.path.basename(file_path)}", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load overlay image:\n{str(e)}")

    def update_overlay(self, *args):
        if hasattr(self, 'overlay_data_full'):
            ch = self.overlay_channel_var.get()
            if self.overlay_data_full.ndim > 2:
                self.overlay_data = self.overlay_data_full[ch]
            else:
                self.overlay_data = self.overlay_data_full
            self.update_overlay_display()

    def update_contrast(self, *args):
        if self.base_im and self.image_data is not None:
            vmin_p = self.vmin_scale.get()
            vmax_p = self.vmax_scale.get()
            
            # Ensure vmin < vmax
            if vmin_p >= vmax_p:
                vmax_p = min(100, vmin_p + 1)
                self.vmax_scale.set(vmax_p)
                
            vmin = np.percentile(self.image_data, vmin_p)
            vmax = np.percentile(self.image_data, vmax_p)
            
            self.base_im.set_clim(vmin, vmax)
            self.base_im.set_cmap(self.cmap_var.get())
            self.canvas.draw_idle()

    def update_overlay_display(self, *args):
        if self.overlay_data is not None and self.image_data is not None:
            if self.overlay_im:
                self.overlay_im.remove()
                self.overlay_im = None
                
            if self.show_overlay_var.get():
                off_x = self.overlay_offset_x
                off_y = self.overlay_offset_y
                scale = self.overlay_scale
                
                oh, ow = self.overlay_data.shape
                
                # We need to map the overlay data into the extent of the base image axes
                # taking into account scale and offset
                extent = [off_x, off_x + ow * scale, off_y + oh * scale, off_y] # left, right, bottom, top for origin='upper'
                
                # Mask the lower intensity background so it doesn't occlude the base image
                bg_threshold = np.percentile(self.overlay_data, 5)
                masked_overlay = np.ma.masked_where(self.overlay_data <= bg_threshold, self.overlay_data)

                cmap_name = self.overlay_cmap_var.get()
                cmap = self.custom_cmaps.get(cmap_name, cmap_name)

                self.overlay_im = self.ax.imshow(
                    masked_overlay, 
                    cmap=cmap,
                    alpha=self.alpha_scale.get(),
                    extent=extent,
                    origin='upper'
                )
            self.canvas.draw_idle()

    def on_select(self, verts):
        """Callback for matplotlib PolygonSelector when a shape is completed."""
        self.current_verts = verts
        self.vertex_counter_label.config(text=f"Drawing: {len(verts)} vertices")

    def confirm_polygon(self):
        if not self.current_verts or len(self.current_verts) < 3:
            messagebox.showwarning("Warning", "Please draw a complete polygon with at least 3 vertices first.")
            return
            
        # --- Snap vertices to border logic ---
        if self.image_data is not None:
            h, w = self.image_data.shape
            threshold = 5
            
            snapped_verts = []
            for x, y in self.current_verts:
                nx, ny = x, y
                if nx < threshold: nx = 0
                elif nx > w - 1 - threshold: nx = w - 1
                
                if ny < threshold: ny = 0
                elif ny > h - 1 - threshold: ny = h - 1
                
                snapped_verts.append((nx, ny))
            self.current_verts = snapped_verts
        # -------------------------------------

        default_label = self.fluor_label_var.get()
        label = simpledialog.askstring("Fluorescence Label", "Enter fluorescence label for this region:", initialvalue=default_label)
        
        if label is None:
            return # Cancelled
            
        self.fluor_label_var.set(label) # Update default for next time

        # Save to list
        idx = len(self.polygons) + 1
        
        # Draw permanently on the axes
        poly_patch = patches.Polygon(self.current_verts, closed=True, fill=False, edgecolor='cyan', linewidth=2)
        self.ax.add_patch(poly_patch)
        text_artist = self.ax.text(self.current_verts[0][0], self.current_verts[0][1], f"R{idx}: {label}", color='cyan', fontsize=10)
        
        region_data = {
            'verts': self.current_verts,
            'label': label,
            'patch': poly_patch,
            'text': text_artist,
            'idx': idx
        }
        self.polygons.append(region_data)
        
        # Add to Treeview
        self.tree.insert('', 'end', iid=str(idx), values=(idx, label, len(self.current_verts)))
        
        # Reset the selector for the next polygon
        self.clear_current_polygon()
        self.set_status(f"Region {idx} confirmed.", "green")

    def clear_current_polygon(self):
        if self.selector:
            self.selector.disconnect_events()
            self.selector = PolygonSelector(self.ax, self.on_select, useblit=True, props=dict(color='magenta', linestyle='-', linewidth=2, marker='o', markersize=6))
            if self.mode != "draw":
                self.selector.set_active(False)
            self.current_verts = None
            self.vertex_counter_label.config(text="")
            self.canvas.draw()

    def clear_all_polygons(self):
        for p in self.polygons:
            p['patch'].remove()
            p['text'].remove()
        self.polygons = []
        for item in self.tree.get_children():
            self.tree.delete(item)

    def delete_selected_region(self):
        selected = self.tree.selection()
        if not selected:
            return
            
        for sel in selected:
            idx = int(self.tree.set(sel, 'Index'))
            # Find and remove
            for i, p in enumerate(self.polygons):
                if p['idx'] == idx:
                    p['patch'].remove()
                    p['text'].remove()
                    self.polygons.pop(i)
                    break
            self.tree.delete(sel)
            
        self.canvas.draw()
        self.set_status("Region(s) deleted.", "orange")

    def relabel_selected_region(self):
        selected = self.tree.selection()
        if not selected:
            return
            
        if len(selected) > 1:
            messagebox.showinfo("Info", "Please select only one region to relabel.")
            return
            
        sel = selected[0]
        idx = int(self.tree.set(sel, 'Index'))
        current_label = self.tree.set(sel, 'Label')
        
        new_label = simpledialog.askstring("Relabel Region", "Enter new fluorescence label:", initialvalue=current_label)
        if new_label is None:
            return
            
        for p in self.polygons:
            if p['idx'] == idx:
                p['label'] = new_label
                p['text'].set_text(f"R{idx}: {new_label}")
                break
                
        self.tree.set(sel, 'Label', new_label)
        self.canvas.draw()
        self.set_status(f"Region {idx} relabeled.", "green")


    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        
        try:
            self.lookup_df = pd.read_csv(file_path)
            self.csv_label.config(text=f"Loaded: {os.path.basename(file_path)}", foreground="green")
            self.set_status("CSV Loaded successfully.", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")

    def save_data(self):
        if not self.polygons:
            messagebox.showwarning("Warning", "No regions have been confirmed yet.")
            return
            
        sample_id = self.sample_id_var.get().strip()
        roi_num = self.roi_var.get().strip()
        
        if not sample_id:
            messagebox.showwarning("Warning", "Please enter a Sample ID.")
            return

        # Prepare base dictionary for the DataFrame
        data = []
        for p in self.polygons:
            verts = p['verts']
            # WKT format: POLYGON ((x1 y1, x2 y2, ...))
            wkt_coords = ", ".join([f"{v[0]} {v[1]}" for v in verts])
            # Ensure it's closed in WKT
            if tuple(verts[0]) != tuple(verts[-1]):
                wkt_coords += f", {verts[0][0]} {verts[0][1]}"
            wkt_str = f"POLYGON (({wkt_coords}))"
            
            # JSON format
            json_str = json.dumps([[v[0], v[1]] for v in verts])
            
            row = {
                "Sample_ID": sample_id,
                "ROI_Number": roi_num,
                "Region_Index": p['idx'],
                "Fluorescence_Label": p['label'],
                "Vertex_Count": len(verts),
                "Geometry_WKT": wkt_str,
                "Vertices_JSON": json_str,
                "Image_Path": self.image_path
            }
            data.append(row)
            
        results_df = pd.DataFrame(data)

        # Merge with Lookup CSV if available
        if self.lookup_df is not None:
            merge_col = None
            for col in self.lookup_df.columns:
                if col.lower() in ['sample_id', 'sampleid', 'sample id']:
                    merge_col = col
                    break
            
            if merge_col:
                self.lookup_df[merge_col] = self.lookup_df[merge_col].astype(str)
                results_df["Sample_ID"] = results_df["Sample_ID"].astype(str)
                
                results_df = pd.merge(results_df, self.lookup_df, left_on="Sample_ID", right_on=merge_col, how="left")
                if merge_col != "Sample_ID":
                    results_df.drop(columns=[merge_col], inplace=True)
            else:
                messagebox.showwarning("Warning", "Could not find a 'Sample_ID' column in the lookup CSV. Saving without merging.")

        # Save Output
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            initialfile=f"{sample_id}_regions.csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        
        if save_path:
            results_df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Saved {len(self.polygons)} regions to:\n{save_path}")
            self.set_status("Data saved successfully.", "green")

    def export_mask_tif(self):
        if not self.polygons:
            messagebox.showwarning("Warning", "No regions have been confirmed yet.")
            return
            
        if self.image_data is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return
            
        sample_id = self.sample_id_var.get().strip() or "Sample"
        
        save_dir = filedialog.askdirectory(title="Select Directory to Save Masks")
        if not save_dir:
            return
            
        import cv2
        from collections import defaultdict
        
        h, w = self.image_data.shape
        
        label_groups = defaultdict(list)
        for p in self.polygons:
            label_groups[p['label']].append(p)
            
        saved_files = []
        for label, polys in label_groups.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for i, p in enumerate(polys, start=1):
                verts = np.array(p['verts'], dtype=np.int32)
                cv2.fillPoly(mask, [verts], i)
                
            safe_label = "".join([c for c in label if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
            filename = f"{sample_id}_{safe_label}_mask.tif"
            save_path = os.path.join(save_dir, filename)
            
            tifffile.imwrite(save_path, mask)
            saved_files.append(filename)
            
        messagebox.showinfo("Success", f"Saved {len(saved_files)} mask(s) to:\n{save_dir}\n\n" + "\n".join(saved_files))
        self.set_status("Masks saved successfully.", "green")

    def set_status(self, text, color="black"):
        self.status_label.config(text=text, foreground=color)


if __name__ == "__main__":
    root = tk.Tk()
    # Optional: apply a basic ttk theme
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    app = ImageSegmenterApp(root)
    root.mainloop()