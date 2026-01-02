from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
import os


def fetch_btc_history(window_size: int = 31) -> pd.DataFrame:
    """
    CoinMarketCap'ten BTC historical data tablosunu çeker.

    window_size:
        - Tablodan en fazla kaç satır alınacağını belirler.
        - DataFrame'teki satır sayısı en fazla window_size olur.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver_path = os.path.join(os.path.dirname(__file__), "chromedriver.exe")
    service = Service(driver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 20)

    url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/"

    try:
        driver.get(url)

        # --- history div'ini bul ---
        try:
            history_div = wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "div.history"
            )))
        except TimeoutException:
            print("❌ history div bulunamadı: Sayfa yapısı değişmiş olabilir veya bot kontrolü olabilir.")
            return pd.DataFrame()

        time.sleep(1)

        # --- history div altındaki table ---
        try:
            table = history_div.find_element(By.TAG_NAME, "table")
        except NoSuchElementException:
            print("❌ history div bulundu ama table yok.")
            return pd.DataFrame()

        # --- Satırları oku ---
        try:
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            if not rows:
                print("⚠️ Tablo bulundu ama satır bulunamadı (tbody boş olabilir).")
                return pd.DataFrame()
        except NoSuchElementException:
            print("⚠️ Satırlar okunamadı (tbody bulunamadı).")
            return pd.DataFrame()

        data = []
        for i, r in enumerate(rows):
            if i >= window_size:
                break
            vals = [td.text.strip() for td in r.find_elements(By.TAG_NAME, "td")]
            if len(vals) >= 6 and vals[0]:
                # CMC genelde 7 kolon döndürür: Date, Open, High, Low, Close, Volume, Market Cap
                data.append(vals[:7])

        df = pd.DataFrame(
            data,
            columns=["Date", "Open", "High", "Low", "Close*", "Volume", "Market Cap"]
        )

        return df

    except Exception as e:
        print(f"❌ Beklenmeyen hata oluştu: {e}")
        return pd.DataFrame()

    finally:
        driver.quit()


if __name__ == "__main__":
    df = fetch_btc_history()
    if df.empty:
        print("Veri alınamadı.")
    else:
        print(df)
