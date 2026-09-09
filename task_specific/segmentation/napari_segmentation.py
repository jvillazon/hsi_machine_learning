import napari
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                            QLineEdit, QFileDialog, QTreeWidget, 
                            QTreeWidgetItem, QHBoxLayout, QMessageBox, QScrollArea,
                            QSlider)
from qtpy.QtCore import Qt
import tifffile
import pandas as pd
import numpy as np
import os
import json
import cv2

class SegmentationWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        
        # State
        self.accumulated_data = [] # List of dicts for CSV
        self.current_image_path = None
        self.current_image_metadata = {}
        
        self.base_img_data = None
        self.overlay_img_data = None
        self.overlay_name = None
        
        self.setup_ui()
        self.setup_layers()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Image Loading
        layout.addWidget(QLabel("<b>1. Load Images</b>"))
        btn_load_base = QPushButton("Load Base Image")
        btn_load_base.clicked.connect(self.load_base_image)
        layout.addWidget(btn_load_base)
        
        self.base_slice_layout = QHBoxLayout()
        self.base_slice_layout.addWidget(QLabel("Base Channel/Z:"))
        self.base_slice_slider = QSlider(Qt.Horizontal)
        self.base_slice_slider.setMinimum(0)
        self.base_slice_slider.setEnabled(False)
        self.base_slice_slider.valueChanged.connect(self.update_base_slice)
        self.base_slice_layout.addWidget(self.base_slice_slider)
        self.base_slice_val_label = QLabel("0")
        self.base_slice_layout.addWidget(self.base_slice_val_label)
        layout.addLayout(self.base_slice_layout)
        
        btn_load_overlay = QPushButton("Load Overlay Image")
        btn_load_overlay.clicked.connect(self.load_overlay_image)
        layout.addWidget(btn_load_overlay)
        
        self.overlay_slice_layout = QHBoxLayout()
        self.overlay_slice_layout.addWidget(QLabel("Overlay Channel/Z:"))
        self.overlay_slice_slider = QSlider(Qt.Horizontal)
        self.overlay_slice_slider.setMinimum(0)
        self.overlay_slice_slider.setEnabled(False)
        self.overlay_slice_slider.valueChanged.connect(self.update_overlay_slice)
        self.overlay_slice_layout.addWidget(self.overlay_slice_slider)
        self.overlay_slice_val_label = QLabel("0")
        self.overlay_slice_layout.addWidget(self.overlay_slice_val_label)
        layout.addLayout(self.overlay_slice_layout)
        
        # Metadata
        layout.addWidget(QLabel("<br><b>2. Image Metadata</b>"))
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Sample ID:"))
        self.sample_id_input = QLineEdit()
        h1.addWidget(self.sample_id_input)
        layout.addLayout(h1)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("ROI #:"))
        self.roi_input = QLineEdit()
        h2.addWidget(self.roi_input)
        layout.addLayout(h2)
        
        # Region properties
        layout.addWidget(QLabel("<br><b>3. Assign Label</b>"))
        self.fluor_input = QLineEdit("DAPI")
        layout.addWidget(self.fluor_input)
        
        # Confirmation
        layout.addWidget(QLabel("<br><b>4. Batch Workflow</b>"))
        layout.addWidget(QLabel("Draw polygons on the 'Regions' layer."))
        
        btn_confirm = QPushButton("Add Label Regions to Batch (Deletes current polygons)")
        btn_confirm.clicked.connect(self.confirm_regions)
        btn_confirm.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; font-size: 14px; padding: 12px; border-radius: 6px;")
        layout.addWidget(btn_confirm)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Sample ID", "ROI", "Regions", "Label"])
        layout.addWidget(self.tree)
        
        btn_clear_batch = QPushButton("Clear Batch")
        btn_clear_batch.clicked.connect(self.clear_batch)
        layout.addWidget(btn_clear_batch)
        
        # Export Buttons
        layout.addWidget(QLabel("<br><b>5. Export Data</b>"))
        btn_export_csv = QPushButton("Export CSV (All Batched Regions)")
        btn_export_csv.clicked.connect(self.export_csv)
        layout.addWidget(btn_export_csv)
        
        btn_export_masks = QPushButton("Export Batch Masks TIF(s)")
        btn_export_masks.clicked.connect(self.export_masks)
        layout.addWidget(btn_export_masks)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(layout)
        scroll.setWidget(container)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def ensure_regions_on_top(self):
        if 'Regions' in self.viewer.layers:
            idx = self.viewer.layers.index('Regions')
            top_idx = len(self.viewer.layers) - 1
            if idx != top_idx:
                self.viewer.layers.move(idx, top_idx)
            self.viewer.layers.selection.active = self.viewer.layers['Regions']

    def setup_layers(self):
        if 'Regions' not in self.viewer.layers:
            self.shapes_layer = self.viewer.add_shapes(
                name='Regions', 
                shape_type='polygon', 
                edge_color='cyan', 
                face_color='transparent', 
                edge_width=2
            )
        else:
            self.shapes_layer = self.viewer.layers['Regions']
        self.shapes_layer.mode = 'add_polygon'
        self.ensure_regions_on_top()

    def _extract_tiff_metadata(self, filepath):
        meta = {}
        try:
            with tifffile.TiffFile(filepath) as tif:
                page = tif.pages[0]
                meta['ImageWidth'] = getattr(page, 'imagewidth', page.shape[-1])
                meta['ImageLength'] = getattr(page, 'imagelength', page.shape[-2])
                
                ij_meta = tif.imagej_metadata if tif.imagej_metadata else {}
                
                pixel_size_x = None
                pixel_size_y = None
                unit = ij_meta.get('unit', '')

                # 1. Try page.resolution (common in ImageJ tiffs)
                if hasattr(page, 'resolution') and page.resolution:
                    rx = page.resolution[0]
                    ry = page.resolution[1]
                    if rx:
                        if isinstance(rx, tuple):
                            pixel_size_x = rx[1] / rx[0] if rx[0] != 0 else None
                        else:
                            pixel_size_x = 1.0 / rx if rx != 0 else None
                    if ry:
                        if isinstance(ry, tuple):
                            pixel_size_y = ry[1] / ry[0] if ry[0] != 0 else None
                        else:
                            pixel_size_y = 1.0 / ry if ry != 0 else None

                # 2. Fallback to TIFF tags
                tags = page.tags
                if pixel_size_x is None and 'XResolution' in tags:
                    res = tags['XResolution'].value
                    if isinstance(res, tuple):
                        val = res[0] / res[1] if res[1] != 0 else res[0]
                    else:
                        val = float(res)
                    if val != 0:
                        pixel_size_x = 1.0 / val

                if pixel_size_y is None and 'YResolution' in tags:
                    res = tags['YResolution'].value
                    if isinstance(res, tuple):
                        val = res[0] / res[1] if res[1] != 0 else res[0]
                    else:
                        val = float(res)
                    if val != 0:
                        pixel_size_y = 1.0 / val

                if not unit and 'ResolutionUnit' in tags:
                    unit_val = tags['ResolutionUnit'].value
                    unit_map = {1: 'NONE', 2: 'INCH', 3: 'CENTIMETER'}
                    unit = unit_map.get(unit_val.value if hasattr(unit_val, 'value') else unit_val, str(unit_val))

                meta['Pixel_Size_X'] = pixel_size_x if pixel_size_x is not None else ''
                meta['Pixel_Size_Y'] = pixel_size_y if pixel_size_y is not None else ''
                meta['ResolutionUnit'] = unit if unit else ''
                meta['XResolution'] = meta['Pixel_Size_X']
                meta['YResolution'] = meta['Pixel_Size_Y']
        except Exception as e:
            print(f"Failed to extract TIFF metadata: {e}")
        return meta

    def load_base_image(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Base Image", "", "TIFF Images (*.tif *.tiff *.ome.tif *.ome.tiff);;All Files (*)")
        if not filepath:
            return
            
        self.current_image_path = filepath
        self.current_image_metadata = self._extract_tiff_metadata(filepath)
        
        self.base_img_data = tifffile.imread(filepath)
        
        # Automatically clear image layers to start fresh, but keep shapes
        for layer in list(self.viewer.layers):
            if isinstance(layer, napari.layers.Image):
                self.viewer.layers.remove(layer)
                
        # Clear the drawing layer too since it's a new image
        if 'Regions' in self.viewer.layers:
            self.viewer.layers['Regions'].data = []
            
        if self.base_img_data.ndim > 2:
            self.base_slice_slider.setEnabled(True)
            self.base_slice_slider.setMaximum(self.base_img_data.shape[0] - 1)
            self.base_slice_slider.setValue(0)
            self.base_slice_val_label.setText("0")
            slice_data = self.base_img_data[0]
        else:
            self.base_slice_slider.setEnabled(False)
            self.base_slice_val_label.setText("0")
            slice_data = self.base_img_data
            
        self.viewer.add_image(slice_data, name='Base Image', colormap='gray')
        
        self.sample_id_input.setText(os.path.basename(filepath).split('.')[0])
        self.setup_layers()
        self.ensure_regions_on_top()

    def update_base_slice(self, value):
        self.base_slice_val_label.setText(str(value))
        if self.base_img_data is not None and self.base_img_data.ndim > 2:
            if 'Base Image' in self.viewer.layers:
                self.viewer.layers['Base Image'].data = self.base_img_data[value]

    def load_overlay_image(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Overlay Image", "", "TIFF Images (*.tif *.tiff *.ome.tif *.ome.tiff);;All Files (*)")
        if not filepath:
            return
            
        self.overlay_img_data = tifffile.imread(filepath)
        self.overlay_name = os.path.basename(filepath)
        
        if self.overlay_img_data.ndim > 2:
            self.overlay_slice_slider.setEnabled(True)
            self.overlay_slice_slider.setMaximum(self.overlay_img_data.shape[0] - 1)
            self.overlay_slice_slider.setValue(0)
            self.overlay_slice_val_label.setText("0")
            slice_data = self.overlay_img_data[0]
        else:
            self.overlay_slice_slider.setEnabled(False)
            self.overlay_slice_val_label.setText("0")
            slice_data = self.overlay_img_data
            
        self.viewer.add_image(slice_data, name=self.overlay_name, colormap='green', blending='additive')
        self.ensure_regions_on_top()

    def update_overlay_slice(self, value):
        self.overlay_slice_val_label.setText(str(value))
        if self.overlay_img_data is not None and self.overlay_img_data.ndim > 2:
            if self.overlay_name in self.viewer.layers:
                self.viewer.layers[self.overlay_name].data = self.overlay_img_data[value]

    def confirm_regions(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No base image loaded!")
            return
            
        sample_id = self.sample_id_input.text().strip()
        roi_num = self.roi_input.text().strip()
        label = self.fluor_input.text().strip()
        
        if not sample_id:
            QMessageBox.warning(self, "Warning", "Please provide a Sample ID.")
            return
            
        if len(self.shapes_layer.data) == 0:
            QMessageBox.warning(self, "Warning", "No shapes drawn on 'Regions' layer.")
            return

        added_count = 0
        
        # Get image bounds for clipping
        h = self.current_image_metadata.get('ImageLength')
        w = self.current_image_metadata.get('ImageWidth')
        if not h or not w:
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.Image) and layer.name == 'Base Image':
                    h, w = layer.data.shape[-2:]
                    break
        
        for shape_data in self.shapes_layer.data:
            if len(shape_data) < 3: continue
            
            # Napari shape coords are (y, x). Convert to (x, y) and clip to image boundaries
            verts = []
            for pt in shape_data:
                # Always take the last two coordinates in case of weird dimensions
                y, x = float(pt[-2]), float(pt[-1])
                if h and w:
                    x = max(0.0, min(x, float(w - 1)))
                    y = max(0.0, min(y, float(h - 1)))
                verts.append((x, y))
            
            wkt_coords = ", ".join([f"{v[0]} {v[1]}" for v in verts])
            if tuple(verts[0]) != tuple(verts[-1]):
                wkt_coords += f", {verts[0][0]} {verts[0][1]}"
            wkt_str = f"POLYGON (({wkt_coords}))"
            
            json_str = json.dumps([[v[0], v[1]] for v in verts])
            
            row = {
                "Sample_ID": sample_id,
                "ROI_Number": roi_num,
                "Region_Index": len(self.accumulated_data) + 1,
                "Label": label,
                "Vertex_Count": len(verts),
                "Geometry_WKT": wkt_str,
                "Vertices_JSON": json_str,
                "Image_Path": os.path.basename(self.current_image_path),
                "Image_Width": self.current_image_metadata.get('ImageWidth', ''),
                "Image_Height": self.current_image_metadata.get('ImageLength', ''),
                "Pixel_Size_X": self.current_image_metadata.get('XResolution', ''),
                "Pixel_Size_Y": self.current_image_metadata.get('YResolution', ''),
                "Resolution_Unit": self.current_image_metadata.get('ResolutionUnit', '')
            }
            self.accumulated_data.append(row)
            added_count += 1
            
        if added_count > 0:
            item = QTreeWidgetItem([sample_id, roi_num, str(added_count), label])
            self.tree.addTopLevelItem(item)
            self.shapes_layer.data = [] # Clear the layer so user can start fresh on next label/image
            QMessageBox.information(self, "Success", f"{added_count} regions added to batch queue.")

    def clear_batch(self):
        self.accumulated_data = []
        self.tree.clear()
        QMessageBox.information(self, "Cleared", "Batch queue has been cleared.")

    def export_csv(self):
        if not self.accumulated_data:
            QMessageBox.warning(self, "Warning", "Batch queue is empty.")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Save CSV", "batched_regions.csv", "CSV Files (*.csv)")
        if filepath:
            df = pd.DataFrame(self.accumulated_data)
            df.to_csv(filepath, index=False)
            QMessageBox.information(self, "Success", f"Exported {len(self.accumulated_data)} total regions to CSV.")

    def export_masks(self):
        if not self.accumulated_data:
            QMessageBox.warning(self, "Warning", "Batch queue is empty.")
            return
            
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Masks")
        if not save_dir:
            return
            
        from collections import defaultdict
        groups = defaultdict(list)
        for row in self.accumulated_data:
            key = (row['Image_Path'], row['Sample_ID'], row['Label'])
            groups[key].append(row)
            
        saved_count = 0
        for (img_path, sample_id, label), rows in groups.items():
            h = rows[0].get('Image_Height')
            w = rows[0].get('Image_Width')
            
            if not h or not w:
                try:
                    with tifffile.TiffFile(img_path) as tif:
                        h = tif.pages[0].imagelength
                        w = tif.pages[0].imagewidth
                except:
                    print(f"Skipping {sample_id} {label} - could not determine image shape.")
                    continue
            
            mask = np.zeros((int(h), int(w)), dtype=np.uint8)
            
            for i, row in enumerate(rows, start=1):
                verts = json.loads(row['Vertices_JSON'])
                verts_arr = np.array(verts, dtype=np.int32)
                cv2.fillPoly(mask, [verts_arr], i)
                
            safe_label = "".join([c for c in label if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
            filename = f"{sample_id}_{safe_label}_mask.tif"
            out_path = os.path.join(save_dir, filename)
            
            tifffile.imwrite(out_path, mask)
            saved_count += 1
            
        QMessageBox.information(self, "Success", f"Saved {saved_count} mask TIFs to:\n{save_dir}")

def main():
    viewer = napari.Viewer()
    widget = SegmentationWidget(viewer)
    viewer.window.add_dock_widget(widget, area='right', name='Segmentation Controls')
    napari.run()

if __name__ == '__main__':
    main()
