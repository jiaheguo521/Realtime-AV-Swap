
import os
import sys
import time
import threading
import struct
import shutil
import json
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageOps
import cv2
import sounddevice as sd
import numpy as np
import multiprocessing.shared_memory

# --- Setup Paths ---
ROOT_DIR = os.getcwd()
AUDIO_DIR = os.path.join(ROOT_DIR, "audio", "seed-vc-realtime")
VIDEO_DIR = os.path.join(ROOT_DIR, "video", "Deep-Live-Cam")

sys.path.append(AUDIO_DIR)
sys.path.append(VIDEO_DIR)

# --- Imports ---
# Audio
from unified_audio import AudioEngine
# Video
import video_modules.globals
import video_modules.metadata
from video_modules.video_capture import VideoCapturer
from video_modules.face_analyser import get_one_face
from video_modules.processors.frame.core import get_frame_processors_modules
from video_modules.utilities import is_image, is_video, resolve_relative_path

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class UnifiedLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Realtime AV Swap - Unified Control")
        self.geometry("1400x900")
        
        # Shared Memory
        self.shm_name = "RAS_SharedMem"
        self.shm = None
        self.init_shared_memory()
        
        # --- Audio Engine Init ---
        class AudioArgs:
            checkpoint_path = None
            config_path = None
            fp16 = True
            gpu = 0
        
        self.audio_engine = AudioEngine(AudioArgs(), shm_name=self.shm_name)
        
        # --- Video State ---
        self.video_running = False
        self.video_cap = None
        self.frame_processors = []
        self.source_face_image = None
        self.fps_frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Init Video Globals
        video_modules.globals.execution_providers = ['CUDAExecutionProvider']
        video_modules.globals.frame_processors = ['face_swapper']
        video_modules.globals.headless = True
        
        # --- GUI Layout ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_live = self.tabview.add("Realtime AV Swap (Live)")
        self.tab_file = self.tabview.add("Video Processing (File)")
        
        self.setup_live_tab()
        self.setup_file_tab()
        
        # Periodic UI Updates
        self.after(100, self.update_status_loop)

    def init_shared_memory(self):
        SHM_SIZE = 24
        try:
            try:
                self.shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name, create=True, size=SHM_SIZE)
            except FileExistsError:
                self.shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name)
            self.shm.buf[:SHM_SIZE] = bytearray([0] * SHM_SIZE)
        except Exception as e:
            print(f"SHM Init Error: {e}")

    # =========================================================================
    # TAB 1: Realtime Live
    # =========================================================================
    def setup_live_tab(self):
        # Layout: Left (Audio), Right (Video)
        self.tab_live.grid_columnconfigure(0, weight=1)
        self.tab_live.grid_columnconfigure(1, weight=1)
        self.tab_live.grid_rowconfigure(0, weight=1)
        
        # --- Audio Section ---
        self.audio_frame = ctk.CTkFrame(self.tab_live)
        self.audio_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.setup_audio_ui(self.audio_frame)
        
        # --- Video Section ---
        self.video_frame = ctk.CTkFrame(self.tab_live)
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.setup_video_live_ui(self.video_frame)

    def setup_audio_ui(self, parent):
        ctk.CTkLabel(parent, text="Audio Control (Seed-VC)", font=("Roboto", 18, "bold")).pack(pady=10)
        
        # Scrollable Settings Area
        settings_scroll = ctk.CTkScrollableFrame(parent, label_text="Settings")
        settings_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 1. Reference Audio
        ctk.CTkLabel(settings_scroll, text="Reference Audio File:").pack(anchor="w", padx=5)
        self.lbl_audio_ref = ctk.CTkLabel(settings_scroll, text="None", text_color="gray")
        self.lbl_audio_ref.pack(anchor="w", padx=5)
        ctk.CTkButton(settings_scroll, text="Select Reference", command=self.select_audio_ref).pack(pady=5)
        
        # 2. Devices
        self.audio_dev_frame = ctk.CTkFrame(settings_scroll)
        self.audio_dev_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(self.audio_dev_frame, text="Input Device:").grid(row=0, column=0, padx=5)
        self.audio_in_var = ctk.StringVar()
        self.combo_audio_in = ctk.CTkOptionMenu(self.audio_dev_frame, variable=self.audio_in_var)
        self.combo_audio_in.grid(row=0, column=1, padx=5, sticky="ew")
        
        ctk.CTkLabel(self.audio_dev_frame, text="Output Device:").grid(row=1, column=0, padx=5)
        self.audio_out_var = ctk.StringVar()
        self.combo_audio_out = ctk.CTkOptionMenu(self.audio_dev_frame, variable=self.audio_out_var)
        self.combo_audio_out.grid(row=1, column=1, padx=5, sticky="ew")
        
        self.refresh_audio_devices()
        ctk.CTkButton(self.audio_dev_frame, text="Refresh Devices", command=self.refresh_audio_devices).grid(row=2, column=0, columnspan=2, pady=5)
        
        # 3. Parameters
        self.create_slider(settings_scroll, "Block Time (s)", "block_time", 0.04, 3.0, 0.5, 0.01)
        self.create_slider(settings_scroll, "Crossfade (s)", "crossfade_time", 0.02, 0.5, 0.05, 0.01)
        self.create_slider(settings_scroll, "Extra Time CE (s)", "extra_time_ce", 0.5, 10.0, 2.5, 0.1)
        self.create_slider(settings_scroll, "Extra Time (s)", "extra_time", 0.5, 10.0, 0.5, 0.1)
        self.create_slider(settings_scroll, "Diffusion Steps", "diffusion_steps", 1, 30, 10, 1)
        self.create_slider(settings_scroll, "Inference CFG Rate", "inference_cfg_rate", 0.0, 1.0, 0.7, 0.1)
        
        # Controls
        self.btn_audio_start = ctk.CTkButton(parent, text="Start Audio Conversion", command=self.toggle_audio, height=40, font=("Roboto", 14, "bold"))
        self.btn_audio_start.pack(pady=10, fill="x", padx=10)
        
        # VC Toggle
        self.var_audio_vc = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(parent, text="Voice Conversion Active", variable=self.var_audio_vc, command=self.update_audio_mode).pack(pady=5)
        
        self.lbl_audio_status = ctk.CTkLabel(parent, text="Status: Stopped", text_color="red")
        self.lbl_audio_status.pack(pady=5)
        
        self.lbl_audio_stats = ctk.CTkLabel(parent, text="Delay: 0ms | Infer: 0ms", text_color="gray")
        self.lbl_audio_stats.pack(pady=5)

    def update_audio_mode(self):
        mode = "vc" if self.var_audio_vc.get() else "im"
        self.audio_engine.config["function"] = mode

    def create_slider(self, parent, label, key, min_val, max_val, default, step):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=2)
        ctk.CTkLabel(frame, text=label).pack(side="left", padx=5)
        lbl_val = ctk.CTkLabel(frame, text=str(default))
        lbl_val.pack(side="right", padx=5)
        
        def update_val(val):
            val = round(float(val), 2)
            lbl_val.configure(text=str(val))
            self.audio_engine.update_config({key: val})

        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, number_of_steps=int((max_val-min_val)/step), command=update_val)
        slider.set(default)
        slider.pack(side="right", fill="x", expand=True, padx=5)
        # Init config
        self.audio_engine.config[key] = default

    def refresh_audio_devices(self):
        try:
            devs = sd.query_devices()
            ins = [d['name'] for d in devs if d['max_input_channels'] > 0]
            outs = [d['name'] for d in devs if d['max_output_channels'] > 0]
            if ins:
                self.combo_audio_in.configure(values=ins)
                self.audio_in_var.set(ins[0])
            if outs:
                self.combo_audio_out.configure(values=outs)
                self.audio_out_var.set(outs[0])
        except: pass

    def select_audio_ref(self):
        path = ctk.filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac")])
        if path:
            if self.audio_engine.load_reference(path):
                self.lbl_audio_ref.configure(text=os.path.basename(path), text_color="green")
            else:
                self.lbl_audio_ref.configure(text="Error loading file", text_color="red")

    def toggle_audio(self):
        if not self.audio_engine.running:
            # Update devices
            self.audio_engine.config["input_device"] = self.audio_in_var.get()
            self.audio_engine.config["output_device"] = self.audio_out_var.get()
            
            self.audio_engine.start_stream()
            if self.audio_engine.running:
                self.btn_audio_start.configure(text="Stop Audio", fg_color="red")
                self.lbl_audio_status.configure(text="Status: Running", text_color="green")
        else:
            self.audio_engine.stop_stream()
            self.btn_audio_start.configure(text="Start Audio Conversion", fg_color=["#3B8ED0", "#1F6AA5"])
            self.lbl_audio_status.configure(text="Status: Stopped", text_color="red")

    # --- Video Live UI ---
    def setup_video_live_ui(self, parent):
        ctk.CTkLabel(parent, text="Video Control (Deep-Live-Cam)", font=("Roboto", 18, "bold")).pack(pady=10)
        
        # Preview
        self.preview_label = ctk.CTkLabel(parent, text="", fg_color="black", width=640, height=480)
        self.preview_label.pack(pady=5)
        
        # FPS Label
        self.lbl_video_fps = ctk.CTkLabel(parent, text="FPS: 0.0", text_color="gray")
        self.lbl_video_fps.pack(pady=2)
        
        # Controls
        ctrl_frame = ctk.CTkFrame(parent)
        ctrl_frame.pack(fill="x", padx=5, pady=5)
        
        # Camera Select
        self.video_cameras = self.get_cameras()
        cam_names = [name for idx, name in self.video_cameras]
        self.cam_var = ctk.StringVar(value=cam_names[0] if cam_names else "No Camera")
        self.combo_cam = ctk.CTkOptionMenu(ctrl_frame, variable=self.cam_var, values=cam_names)
        self.combo_cam.pack(side="left", padx=5)
        
        # Source Face
        ctk.CTkButton(ctrl_frame, text="Select Source Face", command=self.select_source_face).pack(side="left", padx=5)
        self.lbl_source_face = ctk.CTkLabel(ctrl_frame, text="None")
        self.lbl_source_face.pack(side="left", padx=5)
        
        # Toggles Frame
        toggles_frame = ctk.CTkFrame(parent)
        toggles_frame.pack(fill="x", padx=5, pady=5)
        
        self.var_enhancer = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(toggles_frame, text="Face Enhancer", variable=self.var_enhancer, command=self.update_video_config).pack(side="left", padx=10)
        
        self.var_mirror = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(toggles_frame, text="Mirror", variable=self.var_mirror).pack(side="left", padx=10)

        self.var_many_faces = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(toggles_frame, text="Many Faces", variable=self.var_many_faces, command=self.update_video_config).pack(side="left", padx=10)
        
        # Start/Stop
        self.btn_video_start = ctk.CTkButton(parent, text="Start Camera", command=self.toggle_video, height=40, font=("Roboto", 14, "bold"))
        self.btn_video_start.pack(pady=10, fill="x", padx=10)

    def get_cameras(self):
        try:
            import platform
            if platform.system() == "Windows":
                from pygrabber.dshow_graph import FilterGraph
                graph = FilterGraph()
                devices = graph.get_input_devices()
                return [(i, name) for i, name in enumerate(devices)]
        except ImportError:
            pass
            
        # Fallback / Simple camera enumeration
        cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append((i, f"Camera {i}"))
                cap.release()
        return cams if cams else [(0, "Default")]

    def select_source_face(self):
        path = ctk.filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if path:
            video_modules.globals.source_path = path
            try:
                self.source_face_image = get_one_face(cv2.imread(path))
                if self.source_face_image:
                    self.lbl_source_face.configure(text="Loaded", text_color="green")
                else:
                    self.lbl_source_face.configure(text="No Face Found", text_color="red")
            except Exception as e:
                print(e)
                self.lbl_source_face.configure(text="Error", text_color="red")

    def update_video_config(self):
        video_modules.globals.fp_ui["face_enhancer"] = self.var_enhancer.get()
        video_modules.globals.many_faces = self.var_many_faces.get()
        # Re-init processors if running
        if self.video_running:
             self.frame_processors = get_frame_processors_modules(
                ['face_swapper', 'face_enhancer'] if self.var_enhancer.get() else ['face_swapper']
             )

    def toggle_video(self):
        if not self.video_running:
            if not video_modules.globals.source_path:
                messagebox.showwarning("Warning", "Please select a source face first.")
                return

            # Find index
            cam_name = self.cam_var.get()
            idx = 0
            for i, name in self.video_cameras:
                if name == cam_name: idx = i; break
            
            self.video_cap = VideoCapturer(idx)
            if self.video_cap.start(640, 480, 30):
                self.video_running = True
                self.btn_video_start.configure(text="Stop Camera", fg_color="red")
                
                # Init processors
                self.update_video_config()
                if not self.frame_processors:
                     self.frame_processors = get_frame_processors_modules(['face_swapper'])
                
                self.video_loop()
            else:
                messagebox.showerror("Error", "Failed to start camera")
        else:
            self.video_running = False
            if self.video_cap:
                self.video_cap.release()
            self.btn_video_start.configure(text="Start Camera", fg_color=["#3B8ED0", "#1F6AA5"])
            self.preview_label.configure(image=None)

    def video_loop(self):
        if not self.video_running: return
        
        try:
            ret, frame = self.video_cap.read()
            if ret:
                if self.var_mirror.get():
                    frame = cv2.flip(frame, 1)
                
                # Process
                temp_frame = frame.copy()
                if self.source_face_image:
                    for processor in self.frame_processors:
                        if processor.NAME == "DLC.FACE-ENHANCER":
                            if self.var_enhancer.get():
                                 temp_frame = processor.process_frame(None, temp_frame)
                        else:
                            temp_frame = processor.process_frame(self.source_face_image, temp_frame)
                
                # Show
                img_rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_ctk = ctk.CTkImage(img_pil, size=(640, 480))
                self.preview_label.configure(image=img_ctk)
                
                # Update FPS
                self.fps_frame_count += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_frame_count / (time.time() - self.fps_start_time)
                    self.fps_frame_count = 0
                    self.fps_start_time = time.time()
                    self.lbl_video_fps.configure(text=f"FPS: {self.current_fps:.1f}")
                
                # SHM Latency Update (Simulation)
                if self.shm:
                     pass
            else:
                print("Video Capture Read Error: No Frame")
        except Exception as e:
            print(f"Video Loop Error: {e}")
            import traceback
            traceback.print_exc()

        self.after(10, self.video_loop)

    # =========================================================================
    # TAB 2: File Processing
    # =========================================================================
    def setup_file_tab(self):
        # Simplified File Processing UI
        frame = ctk.CTkFrame(self.tab_file)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(frame, text="Video File Processing", font=("Roboto", 20, "bold")).pack(pady=20)
        
        # Source
        self.btn_file_source = ctk.CTkButton(frame, text="Select Source Face Image", command=self.file_select_source)
        self.btn_file_source.pack(pady=10)
        self.lbl_file_source = ctk.CTkLabel(frame, text="None")
        self.lbl_file_source.pack(pady=5)
        
        # Target
        self.btn_file_target = ctk.CTkButton(frame, text="Select Target Video/Image", command=self.file_select_target)
        self.btn_file_target.pack(pady=10)
        self.lbl_file_target = ctk.CTkLabel(frame, text="None")
        self.lbl_file_target.pack(pady=5)
        
        # Output
        self.btn_file_output = ctk.CTkButton(frame, text="Select Output Path", command=self.file_select_output)
        self.btn_file_output.pack(pady=10)
        self.lbl_file_output = ctk.CTkLabel(frame, text="None")
        self.lbl_file_output.pack(pady=5)
        
        # Options
        opts_frame = ctk.CTkFrame(frame)
        opts_frame.pack(fill="x", pady=10)
        
        self.file_enhancer = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opts_frame, text="Face Enhancer", variable=self.file_enhancer).pack(side="left", padx=10)
        
        self.file_many_faces = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opts_frame, text="Many Faces", variable=self.file_many_faces).pack(side="left", padx=10)
        
        self.file_keep_fps = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts_frame, text="Keep FPS", variable=self.file_keep_fps).pack(side="left", padx=10)
        
        self.file_keep_audio = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts_frame, text="Keep Audio", variable=self.file_keep_audio).pack(side="left", padx=10)
        
        # Process Button
        self.btn_file_process = ctk.CTkButton(frame, text="Start Processing", command=self.file_process, fg_color="green", height=50)
        self.btn_file_process.pack(pady=30)
        
        self.lbl_file_status = ctk.CTkLabel(frame, text="Ready")
        self.lbl_file_status.pack(pady=5)

    def file_select_source(self):
        path = ctk.filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if path:
            video_modules.globals.source_path = path
            self.lbl_file_source.configure(text=os.path.basename(path))

    def file_select_target(self):
        path = ctk.filedialog.askopenfilename(filetypes=[("Video/Image", "*.mp4 *.avi *.mov *.jpg *.png")])
        if path:
            video_modules.globals.target_path = path
            self.lbl_file_target.configure(text=os.path.basename(path))

    def file_select_output(self):
        path = ctk.filedialog.asksaveasfilename(defaultextension=".mp4")
        if path:
            video_modules.globals.output_path = path
            self.lbl_file_output.configure(text=os.path.basename(path))

    def file_process(self):
        if not video_modules.globals.source_path or not video_modules.globals.target_path or not video_modules.globals.output_path:
             messagebox.showwarning("Error", "Please select all paths.")
             return
        
        self.lbl_file_status.configure(text="Processing... Please check console for details.")
        self.btn_file_process.configure(state="disabled")
        
        # Run in thread
        threading.Thread(target=self.run_file_process_thread).start()

    def run_file_process_thread(self):
        import video_modules.core
        try:
            # Setup globals
            video_modules.globals.headless = True
            video_modules.globals.execution_providers = ['CUDAExecutionProvider']
            
            # Set options from UI
            video_modules.globals.many_faces = self.file_many_faces.get()
            video_modules.globals.keep_fps = self.file_keep_fps.get()
            video_modules.globals.keep_audio = self.file_keep_audio.get()
            video_modules.globals.fp_ui["face_enhancer"] = self.file_enhancer.get()
            
            video_modules.globals.frame_processors = ['face_swapper']
            if self.file_enhancer.get():
                video_modules.globals.frame_processors.append('face_enhancer')
            
            video_modules.core.start()
            self.lbl_file_status.configure(text="Processing Complete!")
        except Exception as e:
            self.lbl_file_status.configure(text=f"Error: {e}")
            print(e)
        finally:
            self.btn_file_process.configure(state="normal")


    def update_status_loop(self):
        # Update Audio Status
        if self.audio_engine.running:
            delay = int(self.audio_engine.delay_time * 1000)
            infer = int(self.audio_engine.last_inference_time)
            self.lbl_audio_stats.configure(text=f"Delay: {delay}ms | Infer: {infer}ms")
        else:
             self.lbl_audio_stats.configure(text="Delay: 0ms | Infer: 0ms")
             
        self.after(500, self.update_status_loop)

    def on_close(self):
        self.audio_engine.stop_stream()
        self.video_running = False
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        self.destroy()

if __name__ == "__main__":
    app = UnifiedLauncher()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
