# FiTS: Filtered, Time-Sensitive Sentiment Modeling

This project implements the "Filtered, Time-Sensitive" (FiTS) sentiment model, as described in the accompanying project proposal, to predict the next-day direction of the Dow Jones Industrial Average (DJIA).

The core hypothesis is that a *smarter sentiment feature* (FiTS) provides a better signal for a simple classifier than a naive, daily-average sentiment feature. This project compares three models:

1.  **Baseline 1:** Price-Only (using technical features like returns and volatility)
2.  **Baseline 2:** Price + Naive Sentiment (using a simple average of all news for the day)
3.  **Proposed Model:** Price + FiTS (using filtered, time-weighted sentiment)

---

## Setup & Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/YourUsername/Financial_Project.git](https://github.com/YourUsername/Financial_Project.git)
    cd Financial_Project
    ```

2.  **Get the Data:**
    * Download the financial news data from the [original author's repository](https://github.com/felixdrinkall/financial-news-dataset).
    * Decompress all `.xz` files (e.g., using 7-Zip or `xz -d *.xz`).
    * Place all the `..._processed.json` files into the `data/` folder.

3.  **Install Dependencies:**
    ```bash
    pip install pandas yfinance ijson lightgbm scikit-learn
    ```

4.  **Run Preprocessing:**
    ```bash
    python run_preprocessing.py
    ```
    This will create a new file named `final_preprocessed_data.csv` (which is also ignored by Git).

5.  **Run Model:**
    ```bash
    python model.py
    ```

## License

This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, allowing for academic sharing and adaptation while prohibiting commercial use. Researchers may use the dataset under Fair Use/Dealing law, as it is intended for non-commercial research and study, aligning with legal exemptions for academic purposes. By applying this license, we ensure open academic access and maintain compliance with Fair Use/Dealing provisions. Fair Use/Dealing permits the use of copyrighted material for academic purposes because it serves the public interest by enabling research, study, education, and transformative analysis without unfairly impacting the original work's commercial value. 

Fair Use (US Law):

https://www.copyright.gov/fair-use/

Fair Dealing (UK Law):

https://www.gov.uk/guidance/exceptions-to-copyright

## Citation

```bibtex
@misc{drinkall2025financialregression,
      title={When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks}, 
      author={Felix Drinkall and Janet B. Pierrehumbert and Stefan Zohren},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```
