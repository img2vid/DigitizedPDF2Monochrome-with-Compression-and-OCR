# 📄 PDF Text Cleaner & OCR (GUI + CLI)

**PDF Background Removal and Text Enhancement Tool --- GUI Edition**

This project provides a powerful **desktop GUI** and **command-line
tool** for cleaning scanned PDFs, removing noisy or colored backgrounds,
enhancing text visibility, and optionally running **OCR** (Optical
Character Recognition) to create searchable PDFs and text files.

Whether you're working with old, low-quality scanned documents or modern
PDFs with poor text contrast, this tool helps restore readability and
generate machine-searchable outputs.

------------------------------------------------------------------------

## ✨ Features

### 🖥️ Graphical User Interface (GUI)

-   Built using **Tkinter**, providing an intuitive and cross-platform
    interface.
-   File picker dialogs for selecting input/output PDFs.
-   Adjustable processing options:
    -   **DPI** (resolution for image conversion).
    -   **Thresholding methods**: `adaptive`, `otsu`, `sauvola`, or
        `combined`.
    -   **Contrast enhancement** slider.
    -   **Compression options**: `none`, `lzw`, `group4` (ideal for B&W
        text).
    -   **Denoise** and **Morphology** checkboxes for finer control.
-   OCR toggle with **language selection** and optional **Tesseract
    binary path**.
-   Determinate **progress bars** for page cleaning and OCR.
-   **Live logging window** for real-time feedback.
-   **Cancel/Stop button** (best-effort interruption of long jobs).

### 💻 Command-Line Interface (CLI)

For automation and scripting, the CLI interface is preserved:

``` bash
python pdf_text_cleaner_gui.py input.pdf output_cleaned.pdf --dpi 300 --method adaptive --contrast 1.5 --ocr --ocr-lang eng
```

### 🔧 Processing Pipeline

-   Converts each PDF page into high-resolution images using
    **pdf2image + Poppler**.
-   Applies **contrast normalization**, **CLAHE (adaptive histogram
    equalization)**, and custom heuristics for light/dark backgrounds.
-   **Noise reduction** with OpenCV (non-local means & bilateral
    filtering).
-   Multiple **thresholding algorithms**:
    -   *Adaptive Gaussian Thresholding*
    -   *Otsu's Thresholding*
    -   *Sauvola Thresholding* (sliding-window technique for uneven
        lighting)
    -   *Combined voting method* (uses majority decision from the
        above).
-   **Morphological cleanup**: open/close/erode/dilate to sharpen text
    and remove artifacts.
-   Generates a **cleaned, image-only PDF** with optional compression.
-   If OCR is enabled:
    -   Runs **Tesseract OCR** on every page.
    -   Produces:
        -   A `.txt` file with extracted text.
        -   A **searchable PDF** with selectable and searchable text
            overlay.

------------------------------------------------------------------------

## 📦 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/yourusername/pdf-text-cleaner-gui.git
cd pdf-text-cleaner-gui
```

### 2. Install Python dependencies

``` bash
pip install pillow opencv-python-headless pdf2image PyPDF2 tqdm numpy pytesseract
```

### 3. Install system dependencies

This project requires **Tesseract OCR** and **Poppler** to be installed
and available in your system `PATH`.

-   **Tesseract OCR**: [Installation
    Guide](https://digi.bib.uni-mannheim.de/tesseract/)\
-   **Poppler**:
    -   Linux: available in most package managers
        (`apt install poppler-utils`, `brew install poppler` on macOS).\
    -   Windows: download [Poppler for
        Windows](https://github.com/oschwartz10612/poppler-windows/releases/tag/v25.07.0-0) and add
        the `bin/` folder to PATH.

------------------------------------------------------------------------

## 🚀 Usage

### GUI Mode

Simply run without arguments:

``` bash
python pdf_text_cleaner_gui.py
```

This opens the Tkinter-based GUI, allowing interactive PDF cleaning.

### CLI Mode

Run with arguments to process directly from the terminal:

``` bash
python pdf_text_cleaner_gui.py input.pdf output_cleaned.pdf --dpi 300 --method sauvola --contrast 2.0 --ocr --ocr-lang eng+fra
```

### CLI Options

-   `--dpi`: Resolution for conversion (default: 300).\
-   `--method`: Thresholding method (`adaptive`, `otsu`, `sauvola`,
    `combined`).\
-   `--contrast`: Contrast enhancement factor (default: 1.5).\
-   `--compression`: PDF compression (`none`, `lzw`, `group4`).\
-   `--no-denoise`: Disable noise reduction.\
-   `--no-morphology`: Disable morphological cleanup.\
-   `--no-progress`: Hide progress bars.\
-   `--ocr`: Run OCR to generate `.txt` and searchable PDF.\
-   `--ocr-lang`: Tesseract language(s) (e.g., `eng`, `eng+deu`).

------------------------------------------------------------------------

## ⚙️ Example Workflows

### Clean a noisy scanned PDF without OCR:

``` bash
python pdf_text_cleaner_gui.py scans/input.pdf output_cleaned.pdf --method combined --contrast 1.8
```

### Generate a searchable PDF in English and French:

``` bash
python pdf_text_cleaner_gui.py scans/input.pdf output_cleaned.pdf --ocr --ocr-lang eng+fra
```

### Use the GUI for manual configuration:

``` bash
python pdf_text_cleaner_gui.py
```

------------------------------------------------------------------------

## 🧰 Dependencies

Python packages: - `pillow` - `opencv-python-headless` - `pdf2image` -
`PyPDF2` - `tqdm` - `numpy` - `pytesseract`

System dependencies: - **Tesseract OCR** (must be installed and in
PATH).\
- **Poppler** (must be installed and in PATH for `pdf2image` to work).

------------------------------------------------------------------------

## 🖼️ Screenshots (Optional)

*Add screenshots of GUI interface here.*

------------------------------------------------------------------------

## 🛠️ Roadmap / Future Enhancements

-   Batch processing of multiple PDFs.
-   Configurable OCR post-processing (spell-check, language
    auto-detect).
-   Improved cancellation for long OCR jobs.
-   Dark mode GUI support.
-   Portable packaged executables (PyInstaller).

------------------------------------------------------------------------

## 📜 License

[MIT License](LICENSE) -- Free to use, modify, and distribute.
