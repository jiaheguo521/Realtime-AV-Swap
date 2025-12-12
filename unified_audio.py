
import os
import sys
import time
import numpy as np
import torch
import sounddevice as sd
import librosa
import struct
from multiprocessing import shared_memory
import yaml
import torchaudio
import torchaudio.transforms as tat
import torch.nn.functional as F
from audio_modules.commons import *
from hf_utils import load_custom_model_from_hf
from audio_modules.audio import mel_spectrogram

# Add Audio Path to Sys Path if not already there
AUDIO_DIR = os.path.join(os.getcwd(), "audio", "seed-vc-realtime")
if AUDIO_DIR not in sys.path:
    sys.path.append(AUDIO_DIR)

# --- Custom Infer Function (Ported from real-time-gui.py) ---
# We keep this outside the class to match original structure, or make it static.
# Keeping it essentially as is.

prompt_condition, mel2, style2 = None, None, None
reference_wav_name = ""
prompt_len = 3  # in seconds
ce_dit_difference = 2.0  # 2 seconds

@torch.no_grad()
def custom_infer(model_set,
                 reference_wav,
                 new_reference_wav_name,
                 input_wav_res,
                 block_frame_16k,
                 skip_head,
                 skip_tail,
                 return_length,
                 diffusion_steps,
                 inference_cfg_rate,
                 max_prompt_length,
                 cd_difference=2.0,
                 fp16=True,
                 device=None):
    global prompt_condition, mel2, style2
    global reference_wav_name
    global prompt_len
    global ce_dit_difference
    
    (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args) = model_set
    sr = mel_fn_args["sampling_rate"]
    hop_length = mel_fn_args["hop_size"]
    
    if ce_dit_difference != cd_difference:
        ce_dit_difference = cd_difference
        print(f"Setting ce_dit_difference to {cd_difference} seconds.")
        
    if prompt_condition is None or reference_wav_name != new_reference_wav_name or prompt_len != max_prompt_length:
        prompt_len = max_prompt_length
        print(f"Setting max prompt length to {max_prompt_length} seconds.")
        reference_wav = reference_wav[:int(sr * prompt_len)]
        reference_wav_tensor = torch.from_numpy(reference_wav).to(device)

        ori_waves_16k = torchaudio.functional.resample(reference_wav_tensor, sr, 16000)
        S_ori = semantic_fn(ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        mel2 = to_mel(reference_wav_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = model.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
        )[0]

        reference_wav_name = new_reference_wav_name

    converted_waves_16k = input_wav_res
    
    # Timing events
    if device.type == "mps":
        start_event = torch.mps.event.Event(enable_timing=True)
        end_event = torch.mps.event.Event(enable_timing=True)
        torch.mps.synchronize()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

    start_event.record()
    S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
    end_event.record()
    
    if device.type == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()
    # elapsed_time_ms = start_event.elapsed_time(end_event) # Debug

    ce_dit_frame_difference = int(ce_dit_difference * 50)
    S_alt = S_alt[:, ce_dit_frame_difference:]
    target_lengths = torch.LongTensor([(skip_head + return_length + skip_tail - ce_dit_frame_difference) / 50 * sr // hop_length]).to(S_alt.device)
    
    cond = model.length_regulator(
        S_alt, ylens=target_lengths , n_quantizers=3, f0=None
    )[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    
    with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
        vc_target = model.cfm.inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
            mel2,
            style2,
            None,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = vocoder_fn(vc_target).squeeze()
        
    output_len = return_length * sr // 50
    tail_len = skip_tail * sr // 50
    output = vc_wave[-output_len - tail_len: -tail_len]

    return output

class AudioEngine:
    def __init__(self, args, shm_name=None):
        self.args = args
        self.shm_name = shm_name
        self.shm = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AudioEngine] Device: {self.device}")
        
        self.stream = None
        self.running = False
        
        # Load Models
        self.model_set = self.load_models(args)
        from funasr import AutoModel
        self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
        
        # Internal State
        self.reference_wav = None
        self.delay_queue = []
        self.vad_cache = {}
        self.vad_input_history = np.array([], dtype=np.float32)
        self.last_vad_reset_time = time.time()
        self.vad_speech_detected = False
        self.vad_pos_start = False
        self.last_vad_end_time = 0
        self.input_wav = None
        self.input_wav_res = None
        self.sola_buffer = None
        self.fade_in_window = None
        self.fade_out_window = None
        self.resampler = None
        self.resampler2 = None
        
        self.last_inference_time = 0.0
        
        # Config (Defaults)
        self.config = {
            "reference_audio_path": "",
            "diffusion_steps": 10,
            "block_time": 0.5,
            "crossfade_time": 0.05,
            "extra_time_ce": 2.5,
            "extra_time": 0.5,
            "extra_time_right": 2.0,
            "inference_cfg_rate": 0.7,
            "max_prompt_length": 3.0,
            "samplerate": 44100, # Will be overwritten by device default
            "input_device": None,
            "output_device": None,
            "function": "vc", # "vc" or "im" (input monitor/listening)
            "sr_type": "sr_model" # or "sr_device"
        }
        
        self.connect_shm()

    def connect_shm(self):
        if self.shm_name:
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                print(f"[AudioEngine] Connected to Shared Memory: {self.shm_name}")
            except Exception as e:
                print(f"[AudioEngine] Failed to connect to Shared Memory: {e}")

    def load_models(self, args):
        print(f"[AudioEngine] Loading models...")
        # (This is mostly copied from unified_audio.py / real-time-gui.py logic)
        
        if args.checkpoint_path is None or args.checkpoint_path == "":
             dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                             "DiT_uvit_tat_xlsr_ema.pth",
                                                                             "config_dit_mel_seed_uvit_xlsr_tiny.yml")
        else:
             dit_checkpoint_path = args.checkpoint_path
             dit_config_path = args.config_path
             
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = 'DiT'
        model = build_model(model_params, stage="DiT")
        
        model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path, load_only_params=True, ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Campplus
        from audio_modules.campplus.DTDNN import CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval().to(self.device)
        
        # Vocoder
        vocoder_type = model_params.vocoder.type
        if vocoder_type == 'bigvgan':
            from audio_modules.bigvgan import bigvgan
            bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            bigvgan_model.remove_weight_norm()
            vocoder_fn = bigvgan_model.eval().to(self.device)
        elif vocoder_type == 'hifigan':
             from audio_modules.hifigan.generator import HiFTGenerator
             from audio_modules.hifigan.f0_predictor import ConvRNNF0Predictor
             # Fix path for config
             hifigan_config_path = os.path.join(AUDIO_DIR, 'configs/hifigan.yml')
             if not os.path.exists(hifigan_config_path):
                 hifigan_config_path = 'configs/hifigan.yml' # Try relative if in audio dir
             
             hift_config = yaml.safe_load(open(hifigan_config_path, 'r'))
             hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
             hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
             hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
             vocoder_fn = hift_gen.eval().to(self.device)
        else:
             raise ValueError(f"Unknown vocoder: {vocoder_type}")

        # Speech Tokenizer
        speech_tokenizer_type = model_params.speech_tokenizer.type
        if speech_tokenizer_type == 'whisper':
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
            del whisper_model.decoder
            whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)
            
            def semantic_fn(waves_16k):
                ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True)
                ori_input_features = whisper_model._mask_input_features(ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(self.device)
                with torch.no_grad():
                    ori_outputs = whisper_model.encoder(ori_input_features.to(whisper_model.encoder.dtype), head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori
        else:
             # Just simplified for this file, assuming Whisper default
             semantic_fn = None
             print("Warning: Only Whisper speech tokenizer fully implemented in this unified engine.")

        # Mel Specs
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": config["preprocess_params"]["sr"],
            "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        return (model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args)

    def load_reference(self, path):
        if not os.path.exists(path):
            print(f"[AudioEngine] Reference file not found: {path}")
            return False
        try:
            sr = self.model_set[-1]["sampling_rate"]
            self.reference_wav, _ = librosa.load(path, sr=sr)
            # Don't move to tensor yet, custom_infer handles slicing and moving
            print(f"[AudioEngine] Loaded reference: {path}")
            self.config["reference_audio_path"] = path
            return True
        except Exception as e:
            print(f"[AudioEngine] Failed to load reference: {e}")
            return False

    def update_config(self, new_config):
        """Update configuration. Restarts stream if necessary."""
        restart_needed = False
        critical_keys = ["block_time", "crossfade_time", "extra_time_ce", "extra_time", "extra_time_right", "samplerate", "sr_type"]
        
        for k, v in new_config.items():
            if k in self.config and self.config[k] != v:
                self.config[k] = v
                if k in critical_keys:
                    restart_needed = True
        
        if restart_needed and self.running:
            print("[AudioEngine] Critical config changed, restarting stream...")
            self.stop_stream()
            self.start_stream()

    def start_stream(self):
        if self.running: return

        if self.reference_wav is None and self.config["reference_audio_path"]:
            self.load_reference(self.config["reference_audio_path"])
        
        if self.reference_wav is None:
            print("[AudioEngine] Cannot start: No reference audio loaded.")
            return

        # Setup Devices
        try:
            input_device = self.config["input_device"]
            output_device = self.config["output_device"]
            
            # Determine Sample Rate
            if self.config["sr_type"] == "sr_model":
                self.config["samplerate"] = self.model_set[-1]["sampling_rate"]
            else:
                 # Query device default
                 dev_info = sd.query_devices(input_device)
                 self.config["samplerate"] = int(dev_info["default_samplerate"])
            
            sr = self.config["samplerate"]
            self.zc = sr // 50
            
            # Recalculate Buffers
            self.block_frame = int(np.round(self.config["block_time"] * sr / self.zc)) * self.zc
            self.crossfade_frame = int(np.round(self.config["crossfade_time"] * sr / self.zc)) * self.zc
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = int(np.round(self.config["extra_time_ce"] * sr / self.zc)) * self.zc
            self.extra_frame_right = int(np.round(self.config["extra_time_right"] * sr / self.zc)) * self.zc
            
            # Init Buffers
            self.input_wav = torch.zeros(self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame + self.extra_frame_right, device=self.device, dtype=torch.float32)
            self.input_wav_res = torch.zeros(320 * self.input_wav.shape[0] // self.zc, device=self.device, dtype=torch.float32)
            self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.device, dtype=torch.float32)
            
            self.fade_in_window = (torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.device, dtype=torch.float32)) ** 2)
            self.fade_out_window = 1 - self.fade_in_window
            
            # Resamplers
            self.resampler = tat.Resample(orig_freq=sr, new_freq=16000, dtype=torch.float32).to(self.device)
            if self.model_set[-1]["sampling_rate"] != sr:
                self.resampler2 = tat.Resample(orig_freq=self.model_set[-1]["sampling_rate"], new_freq=sr, dtype=torch.float32).to(self.device)
            else:
                self.resampler2 = None
                
            # Input Denoiser Buffer (if needed)
            self.input_wav_denoise = self.input_wav.clone()

            # Start Stream
            self.stream = sd.Stream(
                device=(input_device, output_device),
                samplerate=sr,
                blocksize=self.block_frame,
                channels=1, # Mono input
                dtype="float32",
                callback=self.callback
            )
            self.stream.start()
            self.running = True
            print(f"[AudioEngine] Stream started. SR={sr}, Block={self.config['block_time']}s")
            
            # Calc Delay for SHM
            latency = self.stream.latency[0] + self.stream.latency[1]
            self.delay_time = latency + self.config["block_time"] + self.config["crossfade_time"] + 0.01 + 0.5 # 500ms manual offset
            
        except Exception as e:
            print(f"[AudioEngine] Failed to start stream: {e}")
            import traceback
            traceback.print_exc()
            self.running = False

    def stop_stream(self):
        if self.stream:
            self.stream.abort()
            self.stream.close()
            self.stream = None
        self.running = False
        # Reset SHM
        if self.shm:
            try:
                self.shm.buf[0:8] = struct.pack('d', 0.0)
            except: pass

    def callback(self, indata, outdata, frames, time_info, status):
        if not self.running: return
        start_time = time.perf_counter()
        
        indata = indata[:, 0] # Mono
        
        # VAD
        indata_16k = librosa.resample(indata, orig_sr=self.config["samplerate"], target_sr=16000)
        
        if time.time() - self.last_vad_reset_time > 10.0:
            self.vad_cache = {}
            if len(self.vad_input_history) > 0:
                warmup = self.vad_input_history[-int(16000 * 1.5):]
                self.vad_model.generate(input=warmup, cache=self.vad_cache, is_final=False, chunk_size=500)
            self.last_vad_reset_time = time.time()
            
        res = self.vad_model.generate(input=indata_16k, cache=self.vad_cache, is_final=False, chunk_size=500)
        self.vad_input_history = np.concatenate((self.vad_input_history, indata_16k))[-16000*6:]
        
        if len(res[0]["value"]) > 0:
            self.last_vad_end_time = time.time()
            self.vad_speech_detected = True
        elif self.vad_speech_detected and (time.time() - self.last_vad_end_time > 0.5):
             self.vad_speech_detected = False
             
        # Prepare Input
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-frames:] = torch.from_numpy(indata).to(self.device)
        
        # Resample for Inference
        resampled = librosa.resample(self.input_wav[-frames - 2 * self.zc:].cpu().numpy(), orig_sr=self.config["samplerate"], target_sr=16000)[320:]
        self.input_wav_res[:-len(resampled)] = self.input_wav_res[len(resampled):].clone()
        self.input_wav_res[-len(resampled):] = torch.from_numpy(resampled).to(self.device)
        
        # Inference
        infer_wav = None
        if self.config["function"] == "vc":
            if self.vad_speech_detected:
                try:
                    # Need to check context sizes
                    ce_diff = self.config["extra_time_ce"] - self.config["extra_time"]
                    if ce_diff < 0: ce_diff = 0
                    
                    infer_wav = custom_infer(
                        self.model_set,
                        self.reference_wav,
                        self.config["reference_audio_path"],
                        self.input_wav_res,
                        320 * self.block_frame // self.zc,
                        self.extra_frame // self.zc,
                        self.extra_frame_right // self.zc,
                        (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc,
                        int(self.config["diffusion_steps"]),
                        self.config["inference_cfg_rate"],
                        self.config["max_prompt_length"],
                        ce_diff,
                        fp16=self.args.fp16,
                        device=self.device
                    )
                    
                    if self.resampler2:
                        infer_wav = self.resampler2(infer_wav)
                except Exception as e:
                    print(f"Inference Error: {e}")
                    infer_wav = torch.zeros_like(self.input_wav[self.extra_frame:])
            else:
                 infer_wav = torch.zeros_like(self.input_wav[self.extra_frame:])
            
            self.last_inference_time = (time.perf_counter() - start_time) * 1000
                 
        else: # Input Monitor
             infer_wav = self.input_wav[self.extra_frame:].clone()
             self.last_inference_time = 0.0
             
        # SOLA
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, torch.ones(1, 1, self.sola_buffer_frame, device=self.device)) + 1e-8)
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        sola_offset = torch.argmax(tensor).item() if tensor.numel() > 0 else 0
        
        infer_wav = infer_wav[sola_offset:]
        infer_wav[:self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[:self.sola_buffer_frame] += (self.sola_buffer * self.fade_out_window)
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        
        final_output = infer_wav[:self.block_frame].cpu().numpy()
        
        # Sync & Output
        if self.shm:
             # Logic to write delay and wait for video if needed
             # (Simplification: Just write delay, wait logic can be added if Video drives sync)
             try:
                 self.shm.buf[0:8] = struct.pack('d', self.delay_time * 1000)
                 # Video delay read
                 target_delay = struct.unpack('d', self.shm.buf[8:16])[0]
                 
                 # Here we could implement the delay queue if we want Audio to wait for Video
                 # For now, let's output directly to ensure audio isn't choppy, 
                 # as video usually lags audio.
                 outdata[:] = final_output.reshape(-1, 1)
             except:
                 outdata[:] = final_output.reshape(-1, 1)
        else:
            outdata[:] = final_output.reshape(-1, 1)

