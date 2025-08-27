#!/usr/bin/env python3
"""
PDF Background Removal and Text Enhancement Tool — GUI Edition

This script provides a Tkinter-based GUI wrapper around a slightly
extended version of the provided PDFTextCleaner class. It keeps CLI
support (run with arguments) and launches the GUI when run without
arguments.

Key GUI features:
- File pickers for input/output PDFs
- Options for DPI, method, contrast, compression, denoise, morphology
- OCR toggle and language field (with optional Tesseract path)
- Determinate progress bars for page processing and OCR
- Live log window (captures messages from the cleaner)
- Cancel/Stop button (best-effort interruption)

Dependencies (same as original tool):
  pip install pillow opencv-python-headless pdf2image PyPDF2 tqdm numpy
Optional OCR:
  pip install pytesseract
Also install the Tesseract engine from: https://tesseract-ocr.github.io/

Note: For pdf2image you need Poppler installed on your system.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import pdf2image
from PyPDF2 import PdfWriter, PdfReader
from io import BytesIO
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional, Callable, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# tqdm for progress bars (we'll still render our own progress in the GUI)
try:
    from tqdm.auto import tqdm
except Exception:  # fallback noop if tqdm isn't installed
    def tqdm(x=None, total=None, **kwargs):
        return x if x is not None else range(total or 0)


class PDFTextCleaner:
    """Main class for cleaning scanned PDFs and performing OCR.

    Minor extensions:
      - Accepts optional progress_callback(dict) and cancel_check() callables
      - Emits structured progress events at key milestones
      - Checks cancel condition inside long loops
    """

    def __init__(self, input_path: str, output_path: str, dpi: int = 300,
                 show_progress: bool = True, compression: str = 'group4',
                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 cancel_check: Optional[Callable[[], bool]] = None,
                 log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the PDF cleaner

        Args:
            input_path: Path to input PDF
            output_path: Path to output PDF (cleaned image-based PDF)
            dpi: DPI for PDF to image conversion
            show_progress: Whether to display tqdm progress bars (CLI)
            compression: Compression method for output PDF ('none', 'lzw', 'group4')
            progress_callback: Optional callback receiving progress dicts
            cancel_check: Optional function returning True when user requested cancel
            log_callback: Optional callback to receive log lines (in addition to print)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.dpi = dpi
        self.temp_dir = tempfile.mkdtemp()
        self.show_progress = show_progress
        self.compression = compression
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.log_callback = log_callback

    # -------------- Helpers --------------
    def _emit(self, **kwargs):
        if self.progress_callback:
            try:
                self.progress_callback(kwargs)
            except Exception:
                pass

    def _log(self, msg: str):
        print(msg)
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass

    def _check_cancel(self):
        if self.cancel_check and self.cancel_check():
            raise KeyboardInterrupt("Cancelled by user")

    # -------------- Public API --------------
    def process(self, method: str = 'adaptive', enhance_contrast: float = 1.5,
                denoise: bool = True, morphology: bool = True,
                perform_ocr: bool = False, ocr_lang: Optional[str] = None) -> None:
        """
        Process the PDF with specified parameters

        Args:
            method: Processing method ('adaptive', 'otsu', 'sauvola', 'combined')
            enhance_contrast: Contrast enhancement factor
            denoise: Apply denoising
            morphology: Apply morphological operations
            perform_ocr: Whether to run Tesseract OCR on the cleaned images
            ocr_lang: Language for Tesseract OCR (e.g., 'eng', 'fra+eng'). None for auto.
        """
        self._emit(stage='start')
        self._log(f"Processing PDF: {self.input_path}")
        self._log(f"Method: {method}, Contrast: {enhance_contrast}, Denoise: {denoise}")
        self._log(f"Output Compression: {self.compression}")

        # Convert PDF pages to images
        pages = self._pdf_to_images()
        total_pages = len(pages)
        self._emit(stage='pages_detected', total_pages=total_pages)

        # Process each page
        processed_pages = []
        page_iter = enumerate(pages, 1)
        if self.show_progress:
            page_iter = tqdm(page_iter, total=len(pages), desc="Processing pages", unit="page")

        for i, page in page_iter:
            self._check_cancel()
            processed = self._process_page(page, method, enhance_contrast,
                                           denoise, morphology)
            processed_pages.append(processed)
            self._emit(stage='page_processed', index=i, total=total_pages)

        # Save the cleaned, image-only PDF
        self._log("Saving cleaned image-based PDF...")
        self._emit(stage='saving_cleaned')
        if self.show_progress:
            with tqdm(total=1, desc="Writing cleaned PDF", unit="file") as bar:
                self._images_to_pdf(processed_pages, self.output_path)
                bar.update(1)
        else:
            self._images_to_pdf(processed_pages, self.output_path)
        self._log(f"Cleaned image-based PDF saved to: {self.output_path}")
        self._emit(stage='cleaned_saved', path=self.output_path)

        # Perform OCR if requested
        if perform_ocr:
            self._log("\nPerforming OCR...")
            self._perform_ocr_and_save(processed_pages, ocr_lang)

        # Cleanup
        self._cleanup()
        self._emit(stage='finished', success=True)

    # -------------- Internals --------------
    def _pdf_to_images(self) -> List[Image.Image]:
        """Convert PDF pages to PIL images (page-by-page with progress when possible)"""
        self._check_cancel()
        total_pages = None
        try:
            info = pdf2image.pdfinfo_from_path(self.input_path, userpw=None, poppler_path=None)
            total_pages = int(info.get("Pages", 0)) or None
        except Exception:
            try:
                total_pages = len(PdfReader(self.input_path).pages)
            except Exception:
                total_pages = None

        pages: List[Image.Image] = []

        if total_pages:
            page_range = range(1, total_pages + 1)
            iterator = page_range
            if self.show_progress:
                iterator = tqdm(page_range, desc="Converting PDF to images", unit="page", total=total_pages)
            for p in iterator:
                self._check_cancel()
                try:
                    imgs = pdf2image.convert_from_path(
                        self.input_path, dpi=self.dpi, output_folder=self.temp_dir,
                        fmt='PNG', first_page=p, last_page=p, thread_count=1,
                        use_pdftocairo=True
                    )
                except Exception:
                    imgs = pdf2image.convert_from_path(
                        self.input_path, dpi=self.dpi, output_folder=self.temp_dir,
                        fmt='PNG', first_page=p, last_page=p
                    )
                pages.extend(imgs)
                self._emit(stage='page_converted', index=p, total=total_pages)
            return pages

        if self.show_progress:
            self._log("Converting PDF to images (page count unknown; progress unavailable)...")
        try:
            pages = pdf2image.convert_from_path(
                self.input_path, dpi=self.dpi, output_folder=self.temp_dir,
                fmt='PNG', thread_count=4, use_pdftocairo=True
            )
        except Exception:
            pages = pdf2image.convert_from_path(
                self.input_path, dpi=self.dpi, output_folder=self.temp_dir, fmt='PNG'
            )
        return pages

    def _process_page(self, page: Image.Image, method: str,
                      contrast_factor: float, denoise: bool,
                      morphology: bool) -> Image.Image:
        """Process a single page"""
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()
        processed = self._preprocess(gray, contrast_factor)
        if denoise:
            processed = self._denoise(processed)

        if method == 'adaptive':
            binary = self._adaptive_threshold(processed)
        elif method == 'otsu':
            binary = self._otsu_threshold(processed)
        elif method == 'sauvola':
            binary = self._sauvola_threshold(processed)
        elif method == 'combined':
            binary = self._combined_threshold(processed)
        else:
            binary = self._adaptive_threshold(processed)

        if morphology:
            binary = self._morphological_cleanup(binary)
        return Image.fromarray(binary)

    def _preprocess(self, img: np.ndarray, contrast_factor: float) -> np.ndarray:
        """Preprocess the image to handle various background colors"""
        normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized.astype(np.uint8))
        if contrast_factor != 1.0:
            pil_img = Image.fromarray(enhanced)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced_pil = enhancer.enhance(contrast_factor)
            enhanced = np.array(enhanced_pil)
        mean_val = np.mean(enhanced)
        if mean_val > 180:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=-50)
        elif mean_val < 100:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=30)
        return enhanced

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply denoising to the image"""
        denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        return denoised

    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        h, w = img.shape
        block_size = min(h, w) // 50
        block_size = max(11, block_size if block_size % 2 == 1 else block_size + 1)
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 10
        )
        return binary

    def _otsu_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding"""
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _sauvola_threshold(self, img: np.ndarray, window_size: int = 25,
                           k: float = 0.2, r: float = 128) -> np.ndarray:
        """Apply Sauvola thresholding"""
        integ = cv2.integral(img)
        integ_sq = cv2.integral(img ** 2)
        h, w = img.shape
        binary = np.zeros_like(img)
        w_half = window_size // 2
        for y in range(h):
            self._check_cancel()
            for x in range(w):
                y1, y2 = max(0, y - w_half), min(h - 1, y + w_half)
                x1, x2 = max(0, x - w_half), min(w - 1, x + w_half)
                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                sum_val = integ[y2 + 1, x2 + 1] - integ[y1, x2 + 1] - integ[y2 + 1, x1] + integ[y1, x1]
                sum_sq = integ_sq[y2 + 1, x2 + 1] - integ_sq[y1, x2 + 1] - integ_sq[y2 + 1, x1] + integ_sq[y1, x1]
                mean = sum_val / area
                std = np.sqrt(max(0, (sum_sq - sum_val ** 2 / area) / area))
                threshold = mean * (1 + k * ((std / r) - 1))
                binary[y, x] = 255 if img[y, x] > threshold else 0
        return binary.astype(np.uint8)

    def _combined_threshold(self, img: np.ndarray) -> np.ndarray:
        """Combine multiple thresholding methods"""
        adaptive = self._adaptive_threshold(img)
        otsu = self._otsu_threshold(img)
        sauvola = self._sauvola_threshold(img)
        vote_sum = (adaptive / 255) + (otsu / 255) + (sauvola / 255)
        combined = np.zeros_like(img)
        combined[vote_sum >= 2] = 255
        return combined.astype(np.uint8)

    def _morphological_cleanup(self, img: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the image"""
        kernel_noise = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_noise)
        kernel_close = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        kernel_erode = np.ones((2, 2), np.uint8)
        cleaned = cv2.erode(cleaned, kernel_erode, iterations=1)
        kernel_dilate = np.ones((2, 2), np.uint8)
        cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)
        return cleaned

    def _images_to_pdf(self, images: List[Image.Image], path: str) -> None:
        """Convert processed images back to PDF with optional compression"""
        if not images:
            raise ValueError("No images to convert")
        bitonal_images = [img.convert('1') for img in images]
        save_kwargs = {
            "resolution": self.dpi,
            "save_all": True,
            "append_images": bitonal_images[1:] if len(images) > 1 else []
        }
        if self.compression and self.compression.lower() != 'none':
            save_kwargs["compression"] = self.compression
        bitonal_images[0].save(path, "PDF", **save_kwargs)

    def _perform_ocr_and_save(self, images: List[Image.Image], lang: Optional[str]) -> None:
        """
        Runs Tesseract OCR on the images and saves the output to a text file
        and a searchable PDF.
        """
        # Import pytesseract lazily so GUI can run without it if OCR is off
        try:
            import pytesseract
        except ImportError as e:
            self._log("Error: pytesseract is not installed. Install with 'pip install pytesseract' and install the Tesseract engine.")
            raise

        # Derive output paths from the main output path
        base_path, _ = os.path.splitext(self.output_path)
        txt_output_path = base_path + ".txt"
        pdf_searchable_output_path = base_path + "_searchable.pdf"

        full_text = ""
        pdf_writer = PdfWriter()

        # Set up iterator with progress bar
        image_iter = enumerate(images, 1)
        total = len(images)
        if self.show_progress:
            image_iter = enumerate(tqdm(images, desc="Running OCR", unit="page"), 1)

        for idx, img in image_iter:
            self._check_cancel()
            # Get text string
            page_text = pytesseract.image_to_string(img, lang=lang)
            full_text += page_text + "\n\f\n"  # Add form feed between pages
            self._emit(stage='ocr_page_processed', index=idx, total=total)

            # Get searchable PDF data
            pdf_page_data = pytesseract.image_to_pdf_or_hocr(img, lang=lang, extension='pdf')
            pdf_page_reader = PdfReader(BytesIO(pdf_page_data))
            pdf_writer.add_page(pdf_page_reader.pages[0])

        # Save the full text to a .txt file
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        self._log(f"OCR text saved to: {txt_output_path}")

        # Save the searchable PDF
        with open(pdf_searchable_output_path, "wb") as f:
            pdf_writer.write(f)
        self._log(f"Searchable PDF saved to: {pdf_searchable_output_path}")
        self._emit(stage='ocr_finished', txt_path=txt_output_path, pdf_path=pdf_searchable_output_path)

    def _cleanup(self) -> None:
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# ------------------------------ GUI ------------------------------
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class CleanerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Text Cleaner & OCR")
        self.geometry("860x650")
        self.minsize(820, 600)

        self._build_ui()
        self.worker_thread: Optional[threading.Thread] = None
        self.msg_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.processing = False

        # UI poller
        self.after(100, self._poll_queue)

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {'padx': 10, 'pady': 6}

        # Top: file selection
        file_frame = ttk.LabelFrame(self, text="Files")
        file_frame.pack(fill='x', **pad)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()

        ttk.Label(file_frame, text="Input PDF:").grid(row=0, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(file_frame, textvariable=self.input_var, width=70).grid(row=0, column=1, sticky='we', padx=6, pady=6)
        ttk.Button(file_frame, text="Browse…", command=self._browse_input).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(file_frame, text="Output PDF:").grid(row=1, column=0, sticky='e', padx=6, pady=6)
        ttk.Entry(file_frame, textvariable=self.output_var, width=70).grid(row=1, column=1, sticky='we', padx=6, pady=6)
        ttk.Button(file_frame, text="Browse…", command=self._browse_output).grid(row=1, column=2, padx=6, pady=6)

        file_frame.columnconfigure(1, weight=1)

        # Options
        opts = ttk.LabelFrame(self, text="Options")
        opts.pack(fill='x', **pad)

        # Row 0
        ttk.Label(opts, text="DPI:").grid(row=0, column=0, sticky='e')
        self.dpi_var = tk.IntVar(value=300)
        ttk.Spinbox(opts, from_=72, to=600, textvariable=self.dpi_var, width=7).grid(row=0, column=1, sticky='w')

        ttk.Label(opts, text="Method:").grid(row=0, column=2, sticky='e')
        self.method_var = tk.StringVar(value='adaptive')
        ttk.Combobox(opts, textvariable=self.method_var, values=['adaptive', 'otsu', 'sauvola', 'combined'], width=12, state='readonly').grid(row=0, column=3, sticky='w')

        ttk.Label(opts, text="Contrast:").grid(row=0, column=4, sticky='e')
        self.contrast_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(opts, from_=0.5, to=3.0, increment=0.1, textvariable=self.contrast_var, width=7).grid(row=0, column=5, sticky='w')

        # Row 1
        ttk.Label(opts, text="Compression:").grid(row=1, column=0, sticky='e')
        self.compression_var = tk.StringVar(value='group4')
        ttk.Combobox(opts, textvariable=self.compression_var, values=['none', 'lzw', 'group4'], width=12, state='readonly').grid(row=1, column=1, sticky='w')

        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text='Denoise', variable=self.denoise_var).grid(row=1, column=2, sticky='w')
        self.morph_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text='Morphology', variable=self.morph_var).grid(row=1, column=3, sticky='w')

        self.ocr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text='Run OCR (Tesseract)', variable=self.ocr_var, command=self._toggle_ocr_deps).grid(row=1, column=4, sticky='w')

        self.ocr_lang_var = tk.StringVar(value='')
        ttk.Label(opts, text="OCR Lang:").grid(row=1, column=5, sticky='e')
        ttk.Entry(opts, textvariable=self.ocr_lang_var, width=10).grid(row=1, column=6, sticky='w')

        # Row 2: Tesseract path (optional)
        ttk.Label(opts, text="Tesseract path (optional):").grid(row=2, column=0, columnspan=2, sticky='e')
        self.tess_path_var = tk.StringVar(value='')
        ttk.Entry(opts, textvariable=self.tess_path_var).grid(row=2, column=2, columnspan=3, sticky='we', padx=6)
        ttk.Button(opts, text="Browse…", command=self._browse_tesseract).grid(row=2, column=5, padx=6)

        for c in range(7):
            opts.columnconfigure(c, weight=1 if c in (2,3,4) else 0)

        # Actions
        act = ttk.Frame(self)
        act.pack(fill='x', **pad)
        self.start_btn = ttk.Button(act, text="Start", command=self._start)
        self.start_btn.pack(side='left')
        self.cancel_btn = ttk.Button(act, text="Cancel", command=self._cancel, state='disabled')
        self.cancel_btn.pack(side='left', padx=8)

        # Progress
        prog = ttk.LabelFrame(self, text="Progress")
        prog.pack(fill='x', **pad)
        ttk.Label(prog, text="Pages:").grid(row=0, column=0, sticky='e', padx=6)
        self.pages_pb = ttk.Progressbar(prog, mode='determinate')
        self.pages_pb.grid(row=0, column=1, sticky='we', padx=6, pady=6)
        ttk.Label(prog, text="OCR:").grid(row=1, column=0, sticky='e', padx=6)
        self.ocr_pb = ttk.Progressbar(prog, mode='determinate')
        self.ocr_pb.grid(row=1, column=1, sticky='we', padx=6, pady=6)
        prog.columnconfigure(1, weight=1)

        # Log
        logf = ttk.LabelFrame(self, text="Log")
        logf.pack(fill='both', expand=True, **pad)
        self.log = tk.Text(logf, height=14, wrap='word')
        self.log.pack(fill='both', expand=True)
        self._log_line("Ready.")

    # -------------- UI Callbacks --------------
    def _browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if path:
            self.input_var.set(path)
            base, ext = os.path.splitext(path)
            self.output_var.set(base + "_cleaned.pdf")

    def _browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path:
            self.output_var.set(path)

    def _browse_tesseract(self):
        path = filedialog.askopenfilename(title="Locate Tesseract executable")
        if path:
            self.tess_path_var.set(path)

    def _toggle_ocr_deps(self):
        # Placeholder: we could verify pytesseract install here
        pass

    def _start(self):
        if self.processing:
            return
        inp = self.input_var.get().strip()
        outp = self.output_var.get().strip()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Please choose a valid input PDF.")
            return
        if not outp:
            messagebox.showerror("Error", "Please choose an output PDF path.")
            return
        out_dir = os.path.dirname(outp)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder: {e}")
                return

        # Reset progress
        self.pages_pb['value'] = 0
        self.pages_pb['maximum'] = 100
        self.ocr_pb['value'] = 0
        self.ocr_pb['maximum'] = 100
        self.log.delete('1.0', 'end')

        self.processing = True
        self.cancel_event.clear()
        self.start_btn.config(state='disabled')
        self.cancel_btn.config(state='normal')

        # Launch worker
        args = dict(
            input_path=inp,
            output_path=outp,
            dpi=self.dpi_var.get(),
            show_progress=False,  # GUI owns progress
            compression=self.compression_var.get()
        )

        method = self.method_var.get()
        contrast = float(self.contrast_var.get())
        denoise = bool(self.denoise_var.get())
        morph = bool(self.morph_var.get())
        do_ocr = bool(self.ocr_var.get())
        ocr_lang = self.ocr_lang_var.get().strip() or None
        tess_cmd = self.tess_path_var.get().strip() or None

        def run():
            try:
                # Optional: set custom Tesseract binary path
                if do_ocr and tess_cmd:
                    try:
                        import pytesseract
                        pytesseract.pytesseract.tesseract_cmd = tess_cmd
                    except Exception:
                        pass

                cleaner = PDFTextCleaner(
                    **args,
                    progress_callback=self._progress_handler,
                    cancel_check=lambda: self.cancel_event.is_set(),
                    log_callback=self._log_line,
                )
                cleaner.process(
                    method=method,
                    enhance_contrast=contrast,
                    denoise=denoise,
                    morphology=morph,
                    perform_ocr=do_ocr,
                    ocr_lang=ocr_lang,
                )
                self.msg_q.put({'type': 'done', 'success': True})
            except KeyboardInterrupt:
                self.msg_q.put({'type': 'done', 'success': False, 'error': 'Cancelled by user'})
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.msg_q.put({'type': 'done', 'success': False, 'error': f"{e}\n{tb}"})

        self.worker_thread = threading.Thread(target=run, daemon=True)
        self.worker_thread.start()
        self._log_line("Started processing…")

    def _cancel(self):
        if self.processing:
            self.cancel_event.set()
            self._log_line("Cancellation requested… (it may take a moment to stop)")

    # -------------- Progress & Logging --------------
    def _progress_handler(self, info: Dict[str, Any]):
        # Push into thread-safe queue to update from main thread
        self.msg_q.put({'type': 'progress', 'info': info})

    def _log_line(self, msg: str):
        # Append to log text widget from main thread if possible
        if threading.current_thread() is threading.main_thread():
            self.log.insert('end', msg + '\n')
            self.log.see('end')
        else:
            self.msg_q.put({'type': 'log', 'msg': msg})

    def _poll_queue(self):
        try:
            while True:
                item = self.msg_q.get_nowait()
                if item['type'] == 'log':
                    self.log.insert('end', item['msg'] + '\n')
                    self.log.see('end')
                elif item['type'] == 'progress':
                    self._apply_progress(item['info'])
                elif item['type'] == 'done':
                    self.processing = False
                    self.start_btn.config(state='normal')
                    self.cancel_btn.config(state='disabled')
                    if item.get('success'):
                        self._log_line("\nProcessing completed successfully!")
                        try:
                            outp = self.output_var.get().strip()
                            if outp:
                                self._log_line(f"Output: {outp}")
                        except Exception:
                            pass
                        messagebox.showinfo("Done", "Processing completed successfully!")
                    else:
                        err = item.get('error', 'Unknown error')
                        self._log_line("\nError during processing:\n" + err)
                        messagebox.showerror("Error", err)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _apply_progress(self, info: Dict[str, Any]):
        stage = info.get('stage')
        if stage == 'pages_detected':
            total = info.get('total_pages', 0) or 0
            if total > 0:
                self.pages_pb['maximum'] = total
                self.pages_pb['value'] = 0
        elif stage == 'page_converted':
            # Conversion stage — not strictly necessary, but we could reflect progress here
            pass
        elif stage == 'page_processed':
            idx, total = info.get('index', 0), info.get('total', 0)
            if total:
                self.pages_pb['maximum'] = total
            if idx:
                self.pages_pb['value'] = idx
        elif stage == 'saving_cleaned':
            self._log_line("Writing cleaned PDF…")
        elif stage == 'cleaned_saved':
            self._log_line("Cleaned PDF saved.")
        elif stage == 'ocr_page_processed':
            idx, total = info.get('index', 0), info.get('total', 0)
            if total:
                self.ocr_pb['maximum'] = total
            if idx:
                self.ocr_pb['value'] = idx
        elif stage == 'ocr_finished':
            self._log_line("OCR finished.")


# ------------------------------ CLI (kept for compatibility) ------------------------------

def main_cli(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description='Clean scanned PDFs and optionally perform OCR.'
    )
    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('output', help='Output path for the cleaned, image-only PDF')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for processing (default: 300)')
    parser.add_argument('--method', choices=['adaptive', 'otsu', 'sauvola', 'combined'],
                        default='adaptive', help='Thresholding method (default: adaptive)')
    parser.add_argument('--contrast', type=float, default=1.5,
                        help='Contrast enhancement factor (default: 1.5)')
    parser.add_argument('--compression', choices=['none', 'lzw', 'group4'],
                        default='group4',
                        help="Compression for output PDF. 'group4' is best for B&W. (default: group4)")
    parser.add_argument('--no-denoise', action='store_true', help='Disable denoising')
    parser.add_argument('--no-morphology', action='store_true', help='Disable morphological operations')
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars')
    parser.add_argument('--ocr', action='store_true',
                        help='Enable OCR to generate a .txt file and a searchable PDF.')
    parser.add_argument('--ocr-lang', type=str, default=None,
                        help="Tesseract language(s), e.g., 'eng' or 'eng+fra'. Omit for auto-detection.")

    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        cleaner = PDFTextCleaner(
            args.input,
            args.output,
            args.dpi,
            show_progress=not args.no_progress,
            compression=args.compression
        )
        cleaner.process(
            method=args.method,
            enhance_contrast=args.contrast,
            denoise=not args.no_denoise,
            morphology=not args.no_morphology,
            perform_ocr=args.ocr,
            ocr_lang=args.ocr_lang
        )
        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        if "Tesseract is not installed" in str(e):
            print("---")
            print("Tesseract executable not found. Please ensure Tesseract is installed on your system")
            print("and that the installation directory is included in your system's PATH environment variable.")
            print("Installation guide: https://tesseract-ocr.github.io/tessdoc/Installation.html")
            print("---")
        sys.exit(1)


# ------------------------------ Entry point ------------------------------
if __name__ == "__main__":
    # If run with no arguments: launch the GUI.
    # If run with CLI args: behave like the original CLI.
    if len(sys.argv) == 1:
        app = CleanerGUI()
        app.mainloop()
    else:
        main_cli()
